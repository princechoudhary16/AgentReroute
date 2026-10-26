"""Running a constraint's per-step judge calls concurrently must not change a single verdict.

check_constraint makes one model call per guarded step, each with its own prefix(trajectory, offset),
and on a long trace that is the dominant cost of --use-llm-synthesis. The calls share nothing but the
provider, so they can overlap. What makes that safe is that the provider's last_raw/last_telemetry are
thread-local (a plain attribute let one thread record another's raw response against its own cache
entry) and that cache writes are atomic and content-addressed.

These tests pin the property that matters: identical rows whatever the worker count, in offset order,
with every guarded step evaluated exactly once.
"""
import threading
import time
import unittest
from unittest import mock

from agentrx import perspectives
from agentrx.perspectives import run_perspective_checks

from .test_mp_rl import CONTEXT, tau_trace


def nl_constraint(trigger=None, criteria=2):
    return {'assertion_name': 'judged', 'taxonomy_targets': ['InvalidInvocation'],
            'constraint_type': 'PROTOCOL', 'event_trigger': trigger or {}, 'check_hint': 'judge it',
            'check_type': 'nl_check', 'origin': 'llm',
            'nl_check': {'judge_rubric': [f'criterion {i}' for i in range(criteria)]}}


class RecordingJudge:
    """Counts overlap so a test can tell real concurrency from a sequential loop."""
    identity = {'provider': 'fixture', 'model': 'concurrency-test'}
    measurement_kind = 'offline_fixture'

    def __init__(self, delay=0.05):
        self.delay, self.lock = delay, threading.Lock()
        self.offsets, self.live, self.peak = [], 0, 0

    def complete(self, stage, payload):
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
            self.offsets.append(payload['current_offset'])
        try:
            time.sleep(self.delay)
            rubric = payload['constraint']['nl_check']['judge_rubric']
            return {'rubric_results': [{'criterion_index': i, 'evaluation': 'CLEAR_PASS',
                                        'reasoning': f'step {payload["current_offset"]}'}
                                       for i in range(len(rubric))]}
        finally:
            with self.lock:
                self.live -= 1


def check(workers, delay=0.05):
    judge = RecordingJudge(delay)
    with mock.patch.object(perspectives, 'CHECK_WORKERS', workers):
        started = time.monotonic()
        rows, _ = run_perspective_checks('P1_INTENT_PLAN', [nl_constraint()], tau_trace(), judge, CONTEXT)
    return rows, judge, time.monotonic() - started


class CheckConcurrencyTest(unittest.TestCase):
    def test_concurrent_and_sequential_produce_identical_rows(self):
        serial, _, _ = check(1)
        parallel, _, _ = check(4)
        self.assertEqual([r['verdict'] for r in serial], [r['verdict'] for r in parallel])
        self.assertEqual([r['offset'] for r in serial], [r['offset'] for r in parallel])
        self.assertEqual([r.get('rubric_results') for r in serial],
                         [r.get('rubric_results') for r in parallel])

    def test_rows_stay_in_offset_order_despite_completion_order(self):
        rows, _, _ = check(4)
        self.assertEqual([r['offset'] for r in rows], list(range(len(rows))))

    def test_every_guarded_step_is_evaluated_exactly_once(self):
        _, judge, _ = check(4)
        self.assertEqual(sorted(judge.offsets), sorted(set(judge.offsets)), 'a step was judged twice')
        serial, serial_judge, _ = check(1)
        self.assertEqual(sorted(judge.offsets), sorted(serial_judge.offsets))

    def test_calls_actually_overlap(self):
        _, judge, _ = check(4)
        self.assertGreater(judge.peak, 1, 'no two judge calls were ever in flight together')
        self.assertLessEqual(judge.peak, 4)

    def test_one_worker_never_overlaps(self):
        _, judge, _ = check(1)
        self.assertEqual(judge.peak, 1)

    def test_unowned_steps_are_never_sent_to_the_judge(self):
        """User and system turns carry no agent owner, so their verdicts are discarded downstream."""
        _, judge, _ = check(4)
        trajectory = tau_trace()
        unowned = {o for o in range(len(trajectory['steps'])) if perspectives._unowned(trajectory, o)}
        self.assertTrue(unowned, 'fixture has no unowned steps, so this asserts nothing')
        self.assertFalse(unowned & set(judge.offsets), 'paid for a verdict that gets thrown away')


if __name__ == '__main__':
    unittest.main()
