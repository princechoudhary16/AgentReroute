import unittest

from repair_strategies import lowering
from repair_strategies.context import Hypothesis
from repair_strategies.engine import RepairEngine
from repair_strategies.plan import (Abstain, BanCalls, Compensate, Escalate, GuardCalls, InjectMessage, RepairPlan,
                                    ReplaceEvent, RequeryTool, Resample, Retry, SwitchTool)

from .helpers import retail
from .test_engine import loop_run


def plan(*operations, rollback=5):
    return RepairPlan('s', 'Cycle', rollback, operations)


RULE = {'id': 'confirm', 'kind': 'requires_confirmation', 'tools': ['cancel_pending_order'],
        'message': 'Ask for confirmation before cancelling.'}


class LoweringTest(unittest.TestCase):
    def test_text_only_backend_degrades_and_records_it(self):
        p = plan(RequeryTool('get_order_details', {'order_id': '#W1'}),
                 InjectMessage('tool', 'price = $45.04', name='get_order_details', persistent=True),
                 GuardCalls(rules=(RULE,), validate_schema=True, contract={'goal': 'return the bottle'}),
                 BanCalls(('get_order_details:{}',)), SwitchTool('a', 'b'), ReplaceEvent(3, 2, '{}', 'handoff'),
                 Resample(fresh_context=True))
        low = lowering.lower(p, lowering.BENCHMARK)
        self.assertEqual((low.kind, low.rollback_step, low.resample), (lowering.GUIDANCE, 5, False))
        for snippet in ('[Tool record: get_order_details]', 'Ask for confirmation before cancelling.',
                        'Goal: return the bottle', 'Do not repeat this call', 'Use b instead of a',
                        'Call get_order_details with {"order_id": "#W1"} again', 'Treat step 3 as corrected'):
            self.assertIn(snippet, low.text)
        self.assertEqual(len(low.degraded), 9)
        self.assertTrue(any(d.startswith('resample: backend cannot combine') for d in low.degraded))

    def test_full_backend_keeps_everything_native(self):
        p = plan(RequeryTool('t'), InjectMessage('tool', 'x', name='t', persistent=True), GuardCalls(),
                 BanCalls(('s',)), SwitchTool('a', 'b'), ReplaceEvent(3, 1, '{}'), Resample(True))
        low = lowering.lower(p, lowering.FULL)
        self.assertEqual(low.degraded, ())
        self.assertTrue(low.resample)

    def test_resample_and_retry_without_guidance(self):
        p = plan(Resample(seed_offset=3))
        low = lowering.lower(p, lowering.BENCHMARK)
        self.assertEqual((low.kind, low.seed_offset, low.text), (lowering.RESAMPLE, 3, ''))
        self.assertEqual(lowering.lower(p, lowering.ORCHESTRATOR).kind, lowering.RETRY)
        retry = plan(Retry('get_order_details', {}, backoff_s=2.0))
        self.assertEqual(lowering.lower(retry, lowering.ORCHESTRATOR).kind, lowering.RETRY)
        benchmark = lowering.lower(retry, lowering.BENCHMARK)
        self.assertEqual(benchmark.kind, lowering.RESAMPLE)
        self.assertIn('backoff of 2.0s not applied', benchmark.degraded[0])
        bare = lowering.lower(retry, lowering.Capabilities('text-only'))
        self.assertEqual(bare.kind, lowering.GUIDANCE)
        self.assertIn('Call get_order_details', bare.text)

    def test_terminal_plans(self):
        p = plan(Compensate(({'tool': 'modify_user_address', 'arguments': {'zip': '1'}, 'undoes_step': 4},),
                            ({'tool': 'cancel_pending_order', 'step_index': 6, 'reason': 'no inverse'},)),
                 Escalate('not allowed', 'summary here'), rollback=None)
        low = lowering.lower(p, lowering.ORCHESTRATOR)
        self.assertEqual((low.kind, low.terminal_action, low.rollback_step), (lowering.TERMINAL, 'escalate', None))
        self.assertIn('Undo step 4', low.text)
        self.assertIn('Cannot undo cancel_pending_order', low.text)
        self.assertIn('compensation must be executed by the caller', low.degraded[0])
        abstain = lowering.lower(plan(Abstain('no tool'), rollback=None), lowering.BENCHMARK)
        self.assertEqual(abstain.terminal_action, 'abstain')

    def test_every_real_plan_lowers(self):
        engine = RepairEngine(retail())
        trace = loop_run()
        for category in ('Cycle', 'SystemFailure', 'Instruction/PlanAdherenceFailure', 'IntentPlanMisalignment',
                         'GuardrailsTriggered', 'Inconclusive'):
            p = engine.propose(trace, Hypothesis(8, category))
            for caps in (lowering.BENCHMARK, lowering.ORCHESTRATOR, lowering.FULL):
                low = lowering.lower(p, caps)
                self.assertIn(low.kind, (lowering.GUIDANCE, lowering.RESAMPLE, lowering.RETRY, lowering.TERMINAL))
                self.assertEqual(low.to_dict()['plan_id'], p.plan_id)


if __name__ == '__main__':
    unittest.main()
