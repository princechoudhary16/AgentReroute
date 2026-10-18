import unittest

from repair_strategies.context import AttemptRecord, Hypothesis
from repair_strategies.engine import FALLBACK, RepairEngine
from repair_strategies.plan import Compensate, Escalate
from repair_strategies.strategies import default_strategies

from .helpers import ORDER, USER, RunBuilder, authenticated, retail

NEW_ADDRESS = {'address1': '5 Pine St', 'address2': '', 'city': 'Portland', 'state': 'OR', 'country': 'USA',
               'zip': '97201'}


def run_with_writes():
    """A user-address change before the fault; an order-address change and a cancellation after it."""
    b = authenticated(RunBuilder('fb-1'))
    b.tool('order_agent', 'modify_user_address', '{"status": "ok"}', user_id='ava_lee_2203', **NEW_ADDRESS)
    fault = b.reply('planner', 'Your new order is #W5555555.')                 # invented identifier
    b.tool('order_agent', 'get_order_details', dict(ORDER, status='pending'), order_id='#W1000005')
    b.tool('order_agent', 'modify_pending_order_address', '{"status": "pending"}', order_id='#W1000005',
           **NEW_ADDRESS)
    b.tool('order_agent', 'cancel_pending_order', '{"status": "cancelled"}', order_id='#W1000009',
           reason='no longer needed')
    return b.trace(), fault


class FallbackTest(unittest.TestCase):
    def setUp(self):
        self.engine = RepairEngine(retail())

    def exhausted(self, trace, hypothesis):
        history = []
        while True:
            plan = self.engine.propose(trace, hypothesis, history)
            if plan is None or plan.terminal:
                return plan, history
            history.append(AttemptRecord(plan.strategy_id, plan.category, plan.rollback_step, 'failed'))

    def test_rolls_back_from_the_fault_onward_and_escalates(self):
        trace, fault = run_with_writes()
        plan, history = self.exhausted(trace, Hypothesis(fault, 'InventionOfNewInformation', rationale='made up id'))
        self.assertEqual((plan.strategy_id, plan.rollback_step), ('rollback_compensate_escalate', None))
        self.assertEqual([r.strategy_id for r in history], ['grounded_replacement', 'abstain_or_ask'])
        compensate, escalate = plan.find(Compensate)[0], plan.find(Escalate)[0]
        [inverse] = compensate.calls
        self.assertEqual(inverse['tool'], 'modify_pending_order_address')
        self.assertEqual(inverse['arguments']['address1'], '12 Hillcrest Road')       # value read before the change
        self.assertEqual([i['tool'] for i in compensate.irreversible], ['cancel_pending_order'])
        self.assertIn('after 2 attempt(s) for InventionOfNewInformation', escalate.reason)
        self.assertIn('Kept (before the fault): modify_user_address', escalate.summary)
        self.assertIn('Diagnosis: made up id', escalate.summary)
        self.assertIsNone(self.engine.propose(trace, Hypothesis(fault, 'InventionOfNewInformation'),
                                              history + [AttemptRecord(plan.strategy_id, plan.category, None,
                                                                       'failed')]))

    def test_applies_immediately_when_no_strategy_applies(self):
        b = RunBuilder('fb-2')
        b.user('hello')
        step = b.reply('planner', 'Hi, how can I help?')
        plan = self.engine.propose(b.trace(), Hypothesis(step, 'UnderspecifiedUserIntent'))
        self.assertEqual(plan.strategy_id, 'rollback_compensate_escalate')
        self.assertEqual([o.op for o in plan.operations], ['escalate'])
        self.assertIn('No repair strategy applies to UnderspecifiedUserIntent', plan.operations[0].reason)
        self.assertIn('No data was changed.', plan.operations[0].summary)

    def test_is_listed_last_and_never_in_a_ladder(self):
        trace, fault = run_with_writes()
        hypothesis = Hypothesis(fault, 'InventionOfNewInformation')
        self.assertNotIn('rollback_compensate_escalate', [s.spec.id for s in self.engine.ladder(hypothesis)])
        last = self.engine.evaluate(trace, hypothesis)[-1]
        self.assertEqual((last.strategy_id, last.status), ('rollback_compensate_escalate', FALLBACK))
        self.assertTrue(self.engine.is_fallback(self.engine.fallback(trace, hypothesis)))
        self.assertFalse(self.engine.is_fallback(None))

    def test_engine_without_a_fallback_returns_none(self):
        strategies = tuple(s for s in default_strategies() if not s.spec.fallback)
        engine = RepairEngine(retail(), strategies=strategies)
        b = RunBuilder('fb-3')
        b.user('hello')
        self.assertIsNone(engine.propose(b.trace(), Hypothesis(1, 'UnderspecifiedUserIntent')))
        with self.assertRaises(ValueError):
            fallback = next(s for s in default_strategies() if s.spec.fallback)
            RepairEngine(retail(), strategies=(fallback, type(fallback)()))

    def test_unknown_fault_location_compensates_every_write(self):
        trace, _ = run_with_writes()
        plan = self.engine.fallback(trace, Hypothesis(None, 'Inconclusive'))
        compensate = plan.find(Compensate)[0]
        self.assertEqual(sorted(c['tool'] for c in compensate.calls),
                         ['modify_pending_order_address', 'modify_user_address'])
        self.assertEqual(USER['address']['address1'],
                         next(c for c in compensate.calls if c['tool'] == 'modify_user_address')['arguments']['address1'])


if __name__ == '__main__':
    unittest.main()
