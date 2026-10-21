import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from repair_strategies.memory import lessons, store, tool_outcomes as tom

from .helpers import RunBuilder, authenticated


class StoreTest(unittest.TestCase):
    def test_atomic_write_and_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nested' / 'm.json'
            self.assertEqual(store.read_json(path, {'x': 1}), {'x': 1})
            store.write_json_atomic(path, {'a': [1, 2]})
            self.assertEqual(store.read_json(path, None), {'a': [1, 2]})
            self.assertEqual(sorted(os.listdir(path.parent)), ['m.json'])

    def test_failed_write_keeps_old_file_and_no_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.json'
            store.write_json_atomic(path, {'old': True})
            with mock.patch('repair_strategies.memory.store.os.replace', side_effect=OSError('disk')):
                with self.assertRaises(OSError):
                    store.write_json_atomic(path, {'new': True})
            self.assertEqual(json.loads(path.read_text()), {'old': True})
            self.assertEqual(os.listdir(tmp), ['m.json'])


class ToolOutcomeTest(unittest.TestCase):
    def test_posterior_and_observe_once(self):
        b = authenticated(RunBuilder('run-1'))
        b.tool('order_agent', 'get_order_details', 'Error: 503 service unavailable', order_id='#W1')
        memory = tom.ToolOutcomeMemory()
        self.assertEqual(memory.observe(b.trace()), 3)
        self.assertEqual(memory.observe(b.trace()), 0)            # same trajectory is not double counted
        self.assertAlmostEqual(memory.success_rate('find_user_id_by_email'), 2 / 3)
        self.assertAlmostEqual(memory.success_rate('get_order_details'), 1 / 3)
        self.assertEqual(memory.stats['get_order_details'].transient, 1)
        self.assertEqual(memory.success_rate('never_seen'), 0.5)

    def test_routing(self):
        memory = tom.ToolOutcomeMemory(max_retries=2, min_retry_rate=0.3, base_backoff_s=1.0)
        for _ in range(4):
            memory.record('find_user_id_by_name_zip', True)
        decision = memory.decide('find_user_id_by_email', 'Error: timed out', ['find_user_id_by_name_zip'])
        self.assertEqual((decision.action, decision.backoff_s), (tom.RETRY, 1.0))
        self.assertEqual(memory.decide('find_user_id_by_email', 'Error: timed out', (), attempts=1).backoff_s, 2.0)
        decision = memory.decide('find_user_id_by_email', 'Error: timed out', ['find_user_id_by_name_zip'], 2)
        self.assertEqual((decision.action, decision.tool), (tom.SWITCH, 'find_user_id_by_name_zip'))
        decision = memory.decide('find_user_id_by_email', 'Error: endpoint deprecated', ['find_user_id_by_name_zip'])
        self.assertEqual(decision.action, tom.SWITCH)
        self.assertEqual(memory.decide('get_order_details', 'Error: endpoint deprecated').action, tom.ABSTAIN)
        self.assertEqual(memory.decide('get_order_details', 'Error: timed out', (), attempts=2).action, tom.ABSTAIN)
        for _ in range(10):
            memory.record('flaky', False, 'Error: 503')
        self.assertEqual(memory.decide('flaky', 'Error: 503').action, tom.ABSTAIN)   # too unreliable to retry

    def test_copy_is_independent(self):
        memory = tom.ToolOutcomeMemory()
        memory.record('a', True)
        memory.observed.add('d/1')
        clone = memory.copy()
        clone.record('a', False)
        clone.observed.add('d/2')
        self.assertEqual((memory.stats['a'].failures, memory.observed), (0, {'d/1'}))
        self.assertEqual(clone.stats['a'].successes, 1)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tools.json'
            memory = tom.ToolOutcomeMemory(path)
            memory.record('a', True)
            memory.record('a', False, 'Error: deprecated')
            memory.observed.add('d/1')
            memory.save()
            again = tom.ToolOutcomeMemory(path)
            self.assertEqual(again.to_dict(), memory.to_dict())
        with self.assertRaises(ValueError):
            tom.ToolOutcomeMemory().save()


class LessonTest(unittest.TestCase):
    def test_verified_lessons_and_retrieval(self):
        memory = lessons.LessonMemory()
        a = memory.add_verified('SystemFailure', 'Re-query after a 503 before replying.', 'route',
                                tools=['get_order_details'], agents=['order_agent'])
        again = memory.add_verified('SystemFailure', 'Re-query after a 503 before replying. ', 'route',
                                    tools=['get_user_details'])
        self.assertIs(a, again)
        self.assertEqual((a.verified, a.tools), (2, ('get_order_details', 'get_user_details')))
        memory.add_verified('SystemFailure', 'Unrelated tool lesson.', 'route', tools=['calculate'])
        found = memory.retrieve('SystemFailure', tools=['get_order_details'])
        self.assertEqual([lesson.lesson_id for lesson in found], [a.lesson_id])
        self.assertEqual(memory.retrieve('Cycle', tools=['get_order_details']), [])
        memory.record_failure(a.lesson_id)
        memory.record_failure(a.lesson_id)
        self.assertEqual(memory.retrieve('SystemFailure', tools=['get_order_details']), [])
        memory.record_failure('missing-id')
        with self.assertRaises(ValueError):
            memory.add_verified('SystemFailure', '  ', 's')

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lessons.json'
            memory = lessons.LessonMemory(path)
            memory.add_verified('Cycle', 'Stop after two identical results.', 'loop_nudge', agents=['planner'])
            memory.save()
            self.assertEqual(lessons.LessonMemory(path).to_dict(), memory.to_dict())


if __name__ == '__main__':
    unittest.main()
