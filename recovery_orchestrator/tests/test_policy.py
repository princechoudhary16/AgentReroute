import unittest

from recovery_orchestrator.policy import (ABSTAINED, ESCALATED, INTERVENE, BaselinePolicy, EVConfig,
                                          EVPolicy, UNMODIFIED_STRATEGY, UnmodifiedReplayPolicy)
from recovery_orchestrator.rca import Candidate, RCAPayload, load_payload
from recovery_orchestrator.registry import mock_registry
from recovery_orchestrator.tau_retail import WRITE_TOOLS
from recovery_orchestrator.trace import Step, ToolCall

from .helpers import SYN_ID, SYN_RUN, syn_trace

SIM = EVConfig(respect_irreversible=False)


def payload(*cands, inconclusive=0.0):
    return RCAPayload(SYN_ID, 'tau', 'ok', [Candidate(s, 'assistant', c, 1, share) for s, c, share in cands],
                      len(cands), len(cands), inconclusive)


def ev(config=SIM, **kw):
    return EVPolicy(mock_registry(), config, write_tools=WRITE_TOOLS, **kw)


class EVPolicyTest(unittest.TestCase):
    def setUp(self):
        self.trace = syn_trace()
        self.fa = load_payload(SYN_RUN, SYN_ID, 'tau')

    def test_best_option_then_bayes_shift(self):
        policy = ev()
        guard = policy.make_guard()
        first = policy.decide(self.fa, self.trace, guard)
        self.assertEqual(first.action, INTERVENE)
        self.assertEqual((first.option.rollback_step, first.option.strategy.id), (7, 'reground_tool_output'))
        self.assertEqual([c.step_index for c in first.option.covered], [7, 9])
        self.assertAlmostEqual(first.option.p_success, 0.5 * 0.45)
        policy.observe_failure(first.option)
        guard.record(7, 'reground_tool_output', 0.2, 0.2)
        self.assertAlmostEqual(policy.belief(self.fa.candidates[0]), 0.5 * 0.55 / (1 - 0.225))
        second = policy.decide(self.fa, self.trace, guard)
        self.assertEqual((second.option.rollback_step, second.option.strategy.id), (9, 'policy_reminder'))

    def test_never_repeats_strategy_on_step(self):
        policy = ev()
        guard = policy.make_guard()
        seen = set()
        for _ in range(10):
            d = policy.decide(self.fa, self.trace, guard)
            if d.action != INTERVENE:
                break
            key = (d.option.rollback_step, d.option.strategy.id)
            self.assertNotIn(key, seen)
            seen.add(key)
            policy.observe_failure(d.option)
            guard.record(*key, 0.2, 0.2)
        self.assertEqual(d.action, ABSTAINED)

    def test_near_tie_rolls_back_to_earliest(self):
        p = payload((9, 'MisinterpretationOfToolOutput', 0.4), (7, 'MisinterpretationOfToolOutput', 0.4))
        d = ev().decide(p, self.trace, ev().make_guard())
        self.assertEqual(d.option.rollback_step, 7)
        self.assertEqual({c.step_index for c in d.option.covered}, {7, 9})

    def test_irreversible_downstream_escalates_by_default(self):
        d = ev(EVConfig()).decide(self.fa, self.trace, ev().make_guard())
        self.assertEqual(d.action, ESCALATED)
        self.assertIn('irreversible', d.reason)

    def test_abstain_and_escalate_rules(self):
        trace = self.trace
        self.assertEqual(ev().decide(payload(), trace, ev().make_guard()).action, ABSTAINED)
        p = payload((7, 'MisinterpretationOfToolOutput', 0.3), inconclusive=0.6)
        self.assertEqual(ev().decide(p, trace, ev().make_guard()).action, ABSTAINED)
        p = payload((7, 'IntentNotSupported', 0.8))
        self.assertEqual(ev().decide(p, trace, ev().make_guard()).action, ESCALATED)
        costly = EVConfig(respect_irreversible=False, cost_per_1k_tokens=1.0)
        d = ev(costly).decide(self.fa, trace, ev().make_guard())
        self.assertEqual(d.action, ABSTAINED)
        self.assertIn('EV', d.reason)

    def test_unrepairable_low_confidence_top_falls_through(self):
        p = payload((7, 'GuardrailsTriggered', 0.3), (9, 'InvalidInvocation', 0.3))
        d = ev().decide(p, self.trace, ev().make_guard())
        self.assertEqual((d.action, d.option.primary.category), (INTERVENE, 'InvalidInvocation'))

    def test_calibrator(self):
        cfg = EVConfig(respect_irreversible=False, calibrator=lambda s: s / 2)
        d = ev(cfg).decide(self.fa, self.trace, ev().make_guard())
        self.assertAlmostEqual(d.option.p_success, 0.25 * 0.45)

    def test_detector_candidate_when_fa_misses_loop(self):
        trace = syn_trace()
        trace.steps = trace.steps[:6]
        i = 7
        for _ in range(3):
            trace.steps += [Step(i, 'assistant', '', (ToolCall('get_order_details', {'order_id': '#W1'}),)),
                            Step(i + 1, 'tool', 'same', name='get_order_details')]
            i += 2
        p = payload((5, 'InvalidInvocation', 0.05))
        options, _ = ev().options(p, trace, ev().make_guard())
        self.assertIn('break_loop', [o.strategy.id for o in options])


class BaselineAndControlTest(unittest.TestCase):
    def test_baseline_retries_same_option(self):
        trace, fa = syn_trace(), load_payload(SYN_RUN, SYN_ID, 'tau')
        policy = BaselinePolicy(mock_registry(), retries=2, config=SIM, write_tools=WRITE_TOOLS)
        guard = policy.make_guard()
        for _ in range(2):
            d = policy.decide(fa, trace, guard)
            self.assertEqual((d.option.rollback_step, d.option.strategy.id), (7, 'reground_tool_output'))
            guard.record(7, d.option.strategy.id, 0.1, 0.5)
        self.assertEqual(policy.decide(fa, trace, guard).action, ABSTAINED)

    def test_unmodified_control(self):
        trace, fa = syn_trace(), load_payload(SYN_RUN, SYN_ID, 'tau')
        policy = UnmodifiedReplayPolicy(ev())
        d = policy.decide(fa, trace, policy.make_guard())
        self.assertIs(d.option.strategy, UNMODIFIED_STRATEGY)
        self.assertEqual(d.option.rollback_step, 7)


if __name__ == '__main__':
    unittest.main()
