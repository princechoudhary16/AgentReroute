import unittest

from repair_strategies.context import AttemptRecord, Hypothesis
from repair_strategies.memory.tool_outcomes import ToolOutcomeMemory
from repair_strategies.plan import (Abstain, AskUser, BanCalls, Compensate, Escalate, GuardCalls, InjectMessage,
                                    Resample, Retry, SwitchTool)
from repair_strategies.strategies import Skip
from repair_strategies.strategies.base import RepairContext

from .helpers import ORDER, RunBuilder, authenticated, retail
from .strategy_helpers import RETURN, STRATEGIES, kinds, op, order_run, proposal, run


class PlanningTest(unittest.TestCase):
    def test_intent_contract_replan(self):
        b = authenticated(RunBuilder())
        wrong = b.tool('order_agent', 'cancel_pending_order', '{"status": "cancelled"}', order_id='#W1000009',
                       reason='no longer needed')[0]
        plan = run('intent_contract_replan', b, wrong, 'drift', rationale='cancelled instead of returning')
        self.assertEqual(plan.rollback_step, 2)                      # first agent turn after the request
        self.assertTrue(op(plan, Resample).fresh_context)
        message = op(plan, InjectMessage).content
        self.assertIn('Goal: Hi, my email is', message)
        self.assertIn('cancel_pending_order', message)                # lesson from the abandoned branch
        self.assertEqual(kinds(plan), ['contract_respected', 'calls_valid', 'rules_pass'])
        self.assertIsNotNone(op(plan, GuardCalls).contract)

    def test_confirm_ambiguous_writes(self):
        b = authenticated(RunBuilder('Cancel my pending order.'))
        step = b.call('order_agent', 'cancel_pending_order', order_id='#W1000009', reason='no longer needed')
        plan = run('confirm_ambiguous_writes', b, step, 'IntentPlanMisalignment')
        self.assertEqual(plan.acceptance[0].params['fields'], ['order_id'])
        b2 = authenticated(RunBuilder())
        step = b2.reply('planner', 'Let me check.')
        self.assertIsInstance(run('confirm_ambiguous_writes', b2, step, 'IntentPlanMisalignment'), Skip)

    def test_policy_gate_enforces_rules_and_tracks_confirmed_actions(self):
        b = authenticated(RunBuilder())
        proposal(b, 'return_delivered_order_items', RETURN)
        step = b.reply('planner', 'Your return is done!')             # claims success without the write
        plan = run('policy_gate', b, step, 'Instruction/PlanAdherenceFailure')
        message = op(plan, InjectMessage)
        self.assertTrue(message.persistent)
        self.assertIn('[PENDING] return_delivered_order_items', message.content)
        self.assertEqual(len(op(plan, GuardCalls).rules), 5)
        self.assertIn('obligations_done', kinds(plan))
        fresh = run('policy_gate_fresh', b, step, 'Instruction/PlanAdherenceFailure')
        self.assertTrue(op(fresh, Resample).fresh_context)

    def test_policy_gate_rolls_back_to_the_violation(self):
        b = RunBuilder()
        b.user('cancel #W1000005')
        bad = b.call('order_agent', 'cancel_pending_order', order_id='#W1000005', reason='no longer needed')
        b.result('cancel_pending_order', '{"status": "cancelled"}')
        step = b.reply('order_agent', 'Cancelled.')
        plan = run('policy_gate', b, step, 'Instruction/PlanAdherenceFailure')
        self.assertEqual(plan.rollback_step, bad)
        self.assertTrue(any('[authenticate_first]' in e for e in plan.evidence))
        self.assertTrue(any('[confirm_before_write]' in e for e in plan.evidence))


class ExecutionTest(unittest.TestCase):
    def test_hard_call_repair(self):
        b = authenticated(RunBuilder())
        step = b.call('returns_agent', 'return_delivered_order_items', order_id='#W1000005',
                      item_id=['2000000001'], payment_method_id='credit_card_4421')
        b.result('return_delivered_order_items', "Error: unexpected keyword argument 'item_id'")
        plan = run('hard_call_repair', b, step + 1, 'InvalidInvocation')
        message = op(plan, InjectMessage)
        self.assertEqual((plan.rollback_step, message.role), (step, 'tool'))
        self.assertIn('did you mean "item_ids"', message.content)
        self.assertIn('required format', message.content)
        self.assertTrue(op(plan, GuardCalls).validate_schema)

    def test_valid_successful_call_is_not_repaired(self):
        b = authenticated(RunBuilder())
        self.assertIsInstance(run('hard_call_repair', b, 2, 'InvalidInvocation'), Skip)

    def test_switch_on_drift_and_unknown_tool(self):
        b = RunBuilder()
        b.user('I am Ava Lee, zip 98101.')
        step = b.tool('planner', 'find_user_id_by_email', 'Error: endpoint deprecated', email='x@y.com')[0]
        plan = run('switch_on_schema_drift', b, step, 'InvalidInvocation')
        self.assertEqual(op(plan, SwitchTool).to_tool, 'find_user_id_by_name_zip')
        b = RunBuilder()
        b.user('order status')
        step = b.call('order_agent', 'get_order_detail', order_id='#W1')
        self.assertEqual(op(run('switch_on_schema_drift', b, step, 'InvalidInvocation'), SwitchTool).to_tool,
                         'get_order_details')
        b = authenticated(RunBuilder())
        step = b.tool('order_agent', 'get_order_details', 'Error: 503', order_id='#W1')[0]
        self.assertIsInstance(run('switch_on_schema_drift', b, step, 'InvalidInvocation'), Skip)

    def loop_run(self):
        b = authenticated(RunBuilder())
        for _ in range(3):
            b.tool('order_agent', 'get_order_details', 'Error: still processing, retry', order_id='#W1000005')
        return b

    def test_cycle_ladder(self):
        b = self.loop_run()
        nudge = run('loop_nudge', b, 8, 'Cycle')
        force = run('loop_force_change', b, 8, 'Cycle')
        replan = run('loop_replan', b, 8, 'Cycle')
        rollback = run('loop_rollback', b, 8, 'Cycle')
        self.assertEqual([p.rollback_step for p in (nudge, force, replan, rollback)], [8, 8, 6, 2])
        self.assertIn('3 times', op(nudge, InjectMessage).content)
        self.assertEqual(op(force, BanCalls).signatures, ('get_order_details:{"order_id": "#W1000005"}',))
        self.assertTrue(op(replan, Resample).fresh_context)
        self.assertEqual(nudge.acceptance[0].params['signatures'], list(op(force, BanCalls).signatures))
        clean = authenticated(RunBuilder())
        self.assertIsInstance(run('loop_nudge', clean, 2, 'Cycle'), Skip)

    def test_loop_rollback_skips_when_it_equals_replan(self):
        b = RunBuilder()
        b.user('status?')
        for _ in range(3):
            b.tool('planner', 'get_user_details', 'Error: still processing', user_id='ava_lee_2203')
        self.assertIsInstance(run('loop_rollback', b, 4, 'Cycle'), Skip)

    def test_tool_outcome_routing(self):
        b = authenticated(RunBuilder())
        step = b.tool('planner', 'find_user_id_by_email', 'Error: request timed out', email='a@b.co')[0]
        plan = run('tool_outcome_routing', b, step, 'SystemFailure')
        self.assertEqual((op(plan, Retry).tool, plan.rollback_step), ('find_user_id_by_email', step))
        b = RunBuilder()
        b.user('hi')
        step = b.tool('planner', 'find_user_id_by_email', 'Error: endpoint deprecated', email='a@b.co')[0]
        self.assertEqual(op(run('tool_outcome_routing', b, step, 'SystemFailure'), SwitchTool).to_tool,
                         'find_user_id_by_name_zip')
        b = RunBuilder()
        b.user('hi')
        step = b.tool('order_agent', 'get_order_details', 'Error: endpoint deprecated', order_id='#W1')[0]
        plan = run('tool_outcome_routing', b, step, 'SystemFailure')
        self.assertTrue(plan.terminal)
        self.assertIsInstance(plan.operations[0], Abstain)

    def test_tool_outcome_routing_uses_memory_without_mutating_it(self):
        memory = ToolOutcomeMemory()
        for _ in range(8):
            memory.record('find_user_id_by_email', False, 'Error: 503')
        b = RunBuilder()
        b.user('hi')
        step = b.tool('planner', 'find_user_id_by_email', 'Error: 503', email='a@b.co')[0]
        plan = run('tool_outcome_routing', b, step, 'SystemFailure', memory=memory)
        self.assertEqual(op(plan, SwitchTool).to_tool, 'find_user_id_by_name_zip')
        self.assertEqual(memory.stats['find_user_id_by_email'].failures, 8)
        self.assertEqual(memory.observed, set())

    def test_silent_failure_requeries(self):
        b, step = order_run('Your order is pending.')
        plan = run('tool_outcome_routing', b, step, 'tool_failure')
        self.assertEqual(op(plan, Retry).tool, 'get_order_details')


class EscalationTest(unittest.TestCase):
    def test_targeted_clarification_asks_only_for_missing_fields(self):
        b = authenticated(RunBuilder(), request='Please change my address to 5 Pine St.')
        step = b.call('order_agent', 'modify_user_address', user_id='ava_lee_2203', address1='5 Pine St',
                      address2='', city='Portland', state='OR', country='USA', zip='97201')
        plan = run('targeted_clarification', b, step, 'UnderspecifiedUserIntent')
        self.assertEqual(op(plan, AskUser).fields, ('city', 'state', 'zip'))
        self.assertIn("address2=''", op(plan, InjectMessage).content)
        self.assertEqual(kinds(plan), ['asked_user', 'rules_pass'])

    def test_fully_specified_request_is_skipped(self):
        b = authenticated(RunBuilder(), request='Cancel order #W1000005, I no longer need it.')
        step = b.call('order_agent', 'cancel_pending_order', order_id='#W1000005', reason='no longer needed')
        self.assertIsInstance(run('targeted_clarification', b, step, 'UnderspecifiedUserIntent'), Skip)

    def test_compensate_and_escalate(self):
        pending = dict(ORDER, status='pending')
        b = authenticated(RunBuilder())
        b.tool('order_agent', 'get_order_details', pending, order_id='#W1000005')
        new = {'order_id': '#W1000005', 'address1': '5 Pine St', 'address2': '', 'city': 'Portland', 'state': 'OR',
               'country': 'USA', 'zip': '97201'}
        b.tool('order_agent', 'modify_pending_order_address', '{"status": "pending"}', **new)
        b.tool('order_agent', 'cancel_pending_order', '{"status": "cancelled"}', order_id='#W1000009',
               reason='no longer needed')
        plan = run('compensate_and_escalate', b, b.last, 'GuardrailsTriggered', rationale='refund to crypto')
        self.assertTrue(plan.terminal)
        compensate = op(plan, Compensate)
        [inverse] = compensate.calls
        self.assertEqual(inverse['arguments']['address1'], '12 Hillcrest Road')
        self.assertEqual(inverse['arguments']['order_id'], '#W1000005')
        self.assertEqual(compensate.irreversible[0]['tool'], 'cancel_pending_order')
        escalate = op(plan, Escalate)
        self.assertEqual(escalate.reason, 'refund to crypto')
        self.assertIn('Cannot be reverted: cancel_pending_order', escalate.summary)

    def test_escalate_without_writes(self):
        b = authenticated(RunBuilder())
        plan = run('compensate_and_escalate', b, 2, 'IntentNotSupported')
        self.assertEqual([o.op for o in plan.operations], ['escalate'])

    def test_inconclusive_gets_one_replay_then_abstains(self):
        b = authenticated(RunBuilder())
        plan = run('verified_unmodified_replay', b, None, 'Inconclusive')
        self.assertEqual((plan.rollback_step, plan.acceptance), (2, ()))
        used = [AttemptRecord('verified_unmodified_replay', 'Inconclusive', 2, 'failed')]
        self.assertIsInstance(run('verified_unmodified_replay', b, None, 'Inconclusive', history=used), Skip)
        self.assertTrue(run('abstain_undiagnosed', b, None, 'Inconclusive').terminal)


class CatalogueTest(unittest.TestCase):
    def test_every_category_has_a_first_rung(self):
        from repair_strategies import taxonomy as tx
        for category in tx.CATEGORIES:
            rungs = [s.spec.rung for s in STRATEGIES.values() if category in s.spec.categories]
            self.assertIn(1, rungs, category)

    def test_every_strategy_survives_a_minimal_trace(self):
        b = RunBuilder()
        b.user('hello')
        b.reply('planner', 'Hi!')
        for strategy in STRATEGIES.values():
            for category in strategy.spec.categories:
                for subtype in strategy.spec.subtypes or (None,):
                    ctx = RepairContext(b.trace(), Hypothesis(2, category, subtype=subtype), retail())
                    result = strategy.plan(ctx)
                    self.assertTrue(isinstance(result, Skip) or result.operations, strategy.spec.id)
                    ctx = RepairContext(b.trace(), Hypothesis(99, category, subtype=subtype), retail())
                    strategy.plan(ctx)


if __name__ == '__main__':
    unittest.main()
