import json
import unittest

from repair_strategies import acceptance
from repair_strategies.plan import (AcceptanceCheck, Abstain, AskUser, BanCalls, Compensate, Escalate, GuardCalls,
                                    InjectMessage, Operation, PlanError, RepairPlan, ReplaceEvent, RequeryTool,
                                    Resample, Retry, SwitchTool)
from repair_strategies.trace import ToolCall

from .helpers import ORDER, RunBuilder, authenticated, retail


def plan(**overrides):
    base = dict(strategy_id='s', category='InvalidInvocation', rollback_step=2,
                operations=(InjectMessage('tool', 'corrected value', name='get_order_details'),))
    base.update(overrides)
    return RepairPlan(**base)


class PlanModelTest(unittest.TestCase):
    def test_round_trip_every_operation(self):
        ops = (InjectMessage('system', 'note', persistent=True), ReplaceEvent(3, 2, '{}', 'complete handoff'),
               RequeryTool('get_order_details', {'order_id': '#W1'}), GuardCalls(({'id': 'r'},), True, {'goal': 'g'}),
               BanCalls(('a:{}',)), SwitchTool('a', 'b'), Resample(True, 2), AskUser('Which zip?', ('zip',)),
               Retry('get_order_details', {'order_id': '#W1'}, 1.5))
        original = plan(operations=ops, acceptance=(AcceptanceCheck('no_repeat', {'signatures': ['a:{}']}),),
                        evidence=('e',), hypothesis={'step_index': 2})
        again = RepairPlan.from_dict(json.loads(original.to_json()))
        self.assertEqual(again, original)
        self.assertEqual(again.plan_id, original.plan_id)
        self.assertEqual(len(original.find(InjectMessage)), 1)

    def test_terminal_plans(self):
        terminal = plan(rollback_step=None, operations=(Compensate(({'tool': 't', 'arguments': {}},)),
                                                       Escalate('policy forbids it')))
        self.assertTrue(terminal.terminal)
        self.assertEqual(RepairPlan.from_dict(terminal.to_dict()), terminal)
        with self.assertRaises(PlanError):
            plan(rollback_step=None)
        self.assertTrue(plan(rollback_step=None, operations=(Abstain('no evidence'),)).terminal)

    def test_validation(self):
        bad = [dict(operations=()), dict(rollback_step=0), dict(rollback_step='3'), dict(expected_success=1.2),
               dict(estimated_tokens=-1), dict(operations=('not an op',))]
        for kwargs in bad:
            with self.assertRaises(PlanError, msg=kwargs):
                plan(**kwargs)
        with self.assertRaises(PlanError):
            InjectMessage('assistant', 'x')
        with self.assertRaises(PlanError):
            InjectMessage('tool', '   ')
        with self.assertRaises(PlanError):
            AcceptanceCheck('looks_good')
        with self.assertRaises(PlanError):
            Operation.from_dict({'op': 'teleport'})
        with self.assertRaises(PlanError):
            Operation.from_dict({'op': 'abstain', 'reason': 'x', 'extra': 1})
        with self.assertRaises(PlanError):
            RepairPlan.from_dict({'strategy_id': 's'})
        data = json.loads(plan().to_json())
        data['rationale'] = 'tampered'
        with self.assertRaises(PlanError):
            RepairPlan.from_dict(data)

    def test_plan_id_is_content_based(self):
        self.assertEqual(plan().plan_id, plan().plan_id)
        self.assertNotEqual(plan().plan_id, plan(rollback_step=3).plan_id)


def check(kind, rollout, start, **params):
    return acceptance.run_check(AcceptanceCheck(kind, params), rollout, retail(), start)


class AcceptanceTest(unittest.TestCase):
    def test_claims_and_values(self):
        b = authenticated(RunBuilder())
        b.tool('returns_agent', 'get_order_details', ORDER, order_id='#W1000005')
        start = b.reply('planner', 'The refund for the water bottle is $45.04.')
        rollout = b.trace()
        self.assertTrue(check('claims_grounded', rollout, start).passed)
        self.assertTrue(check('values_present', rollout, start, values=['$45.04']).passed)
        result = check('values_present', rollout, start, values=['$45.04', '#W1000005'])
        self.assertFalse(result.passed)
        self.assertIn('#W1000005', result.detail)
        b.reply('planner', 'Sorry, it is $54.04.')
        self.assertFalse(check('claims_grounded', b.trace(), start).passed)

    def test_repeats_validity_and_errors(self):
        b = authenticated(RunBuilder())
        start = b.last + 1
        call = ToolCall('get_order_details', {'order_id': '#W1000005'})
        b.tool('order_agent', 'get_order_details', 'Error: still processing', order_id='#W1000005')
        rollout = b.trace()
        self.assertFalse(check('no_repeat', rollout, start, signatures=[call.signature()]).passed)
        self.assertTrue(check('calls_valid', rollout, start).passed)
        self.assertFalse(check('no_error_results', rollout, start).passed)
        self.assertTrue(check('no_error_results', rollout, start, tools=['calculate']).passed)
        b.call('planner', 'get_order_details', order_id='#W1000005')       # planner may not call it
        self.assertFalse(check('calls_valid', b.trace(), start).passed)

    def test_rules_obligations_contract(self):
        b = RunBuilder()
        b.user('cancel #W1000005')
        start = b.call('order_agent', 'cancel_pending_order', order_id='#W1000005', reason='no longer needed')
        rollout = b.trace()
        self.assertFalse(check('rules_pass', rollout, start).passed)
        self.assertTrue(check('rules_pass', rollout, start, rule_ids=['cancel_reason']).passed)
        obligation = {'tool': 'cancel_pending_order', 'arguments': {'order_id': '#W1000005'}}
        self.assertFalse(check('obligations_done', rollout, start, obligations=[obligation]).passed)
        b.result('cancel_pending_order', '{"status": "cancelled"}')
        self.assertTrue(check('obligations_done', b.trace(), start, obligations=[obligation]).passed)
        contract = {'goal': 'cancel', 'entities': ['#w1000009']}
        self.assertFalse(check('contract_respected', b.trace(), start, contract=contract).passed)
        self.assertTrue(check('tool_succeeded', b.trace(), start, tool='cancel_pending_order').passed)
        self.assertFalse(check('tool_not_called', b.trace(), start, tool='cancel_pending_order').passed)

    def test_orphan_results_are_not_validated_as_calls(self):
        b = RunBuilder()
        b.user('hi')
        start = b.result('get_order_details', '{}')
        self.assertTrue(check('calls_valid', b.trace(), start).passed)

    def test_asked_user(self):
        b = RunBuilder()
        b.user('change my address')
        start = b.reply('planner', 'Sure. What is the new zip code?')
        self.assertTrue(check('asked_user', b.trace(), start, fields=['zip']).passed)
        self.assertFalse(check('asked_user', b.trace(), start, fields=['city']).passed)

    def test_evaluate_whole_plan(self):
        b = authenticated(RunBuilder())
        start = b.reply('planner', 'Your order #W1000005 is on its way.')
        p = plan(rollback_step=start, acceptance=(AcceptanceCheck('claims_grounded'),
                                                  AcceptanceCheck('values_present', {'values': ['#W1000005']})))
        report = acceptance.evaluate(p, b.trace(), retail())
        self.assertTrue(report.passed)
        self.assertEqual(len(report.to_dict()['results']), 2)
        terminal = plan(rollback_step=None, operations=(Escalate('x'),))
        self.assertTrue(acceptance.evaluate(terminal, b.trace(), retail()).passed)


if __name__ == '__main__':
    unittest.main()
