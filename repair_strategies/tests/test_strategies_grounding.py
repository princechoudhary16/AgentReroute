import json
import unittest

from repair_strategies.llm import ScriptedLLM
from repair_strategies.plan import AskUser, GuardCalls, InjectMessage, ReplaceEvent, RequeryTool
from repair_strategies.strategies import Skip

from .helpers import ORDER, RunBuilder, authenticated
from .strategy_helpers import RETURN, STRATEGIES, VARIANTS, kinds, op, order_run, proposal, run


class GroundingTest(unittest.TestCase):
    def test_grounded_replacement_delivers_the_verified_value_as_a_tool_message(self):
        b, step = order_run('The refund for the water bottle is $54.04.')
        plan = run('grounded_replacement', b, step, 'InventionOfNewInformation')
        message = op(plan, InjectMessage)
        self.assertEqual((plan.rollback_step, message.role), (step, 'tool'))
        self.assertIn('items.0.price = $45.04', message.content)
        self.assertEqual(kinds(plan), ['claims_grounded'])

    def test_grounded_replacement_requeries_the_source_of_an_ungrounded_identifier(self):
        b = authenticated(RunBuilder())
        step = b.reply('planner', 'Your order #W9999999 ships tomorrow.')
        plan = run('grounded_replacement', b, step, 'InventionOfNewInformation')
        self.assertEqual(op(plan, RequeryTool).tool, 'get_user_details')

    def test_grounded_replacement_skips_clean_turns(self):
        b, step = order_run('The refund for the water bottle is $45.04.')
        self.assertIsInstance(run('grounded_replacement', b, step, 'InventionOfNewInformation'), Skip)

    def test_omitted_details_are_supplied(self):
        b, step = order_run('Your return has been submitted.')
        plan = run('complete_omitted_details', b, step, 'missing_details')
        self.assertEqual(plan.acceptance[0].params['values'], ['$45.04'])
        self.assertIn('refund: items.0.price = $45.04', op(plan, InjectMessage).content)
        b, step = order_run('Your return is submitted; the refund is $45.04.')
        self.assertIsInstance(run('complete_omitted_details', b, step, 'missing_details'), Skip)

    def test_abstain_or_ask(self):
        b = RunBuilder()
        b.user('Please update my address.')
        step = b.call('planner', 'find_user_id_by_email', email='someone@example.com')
        plan = run('abstain_or_ask', b, step, 'InventionOfNewInformation')
        self.assertEqual(op(plan, AskUser).fields, ('email',))
        self.assertIn('asked_user', kinds(plan))
        b, step = order_run('Your order #W1234567 is delayed.')
        plan = run('abstain_or_ask', b, step, 'InventionOfNewInformation')
        self.assertFalse(plan.find(AskUser))
        self.assertIn('not available', op(plan, InjectMessage).content)
        b, step = order_run('Everything is fine.')
        self.assertIsInstance(run('abstain_or_ask', b, step, 'InventionOfNewInformation'), Skip)
        plan = run('abstain_or_ask', b, step, 'InventionOfNewInformation', rationale='claimed a free upgrade')
        self.assertLess(plan.expected_success, STRATEGIES['abstain_or_ask'].spec.prior_success)


class InterpretationTest(unittest.TestCase):
    def handoff_run(self, context):
        b = authenticated(RunBuilder())
        proposal(b, 'return_delivered_order_items', RETURN)
        handoff = b.handoff('planner', 'returns_agent', 'Process the confirmed return.', context)
        receiver = b.tool('returns_agent', 'get_order_details', ORDER, order_id='#W1000005')[0]
        return b, handoff, receiver

    def test_handoff_is_patched_and_only_the_receiver_reruns(self):
        b, handoff, receiver = self.handoff_run({'order_id': '#W1000005'})
        plan = run('repair_handoff', b, handoff, 'context_propagation')
        patch = op(plan, ReplaceEvent)
        context = json.loads(patch.content)['context']
        self.assertEqual((patch.step_index, patch.sub_index, plan.rollback_step), (handoff, 2, receiver))
        self.assertEqual(context['item_ids'], ['2000000001'])
        self.assertEqual(context['payment_method_id'], 'credit_card_4421')
        self.assertEqual(op(plan, GuardCalls).rules[0]['id'], 'ids_from_evidence')

    def test_complete_handoff_is_skipped(self):
        b, handoff, _ = self.handoff_run(dict(RETURN))
        self.assertIsInstance(run('repair_handoff', b, handoff, 'context_propagation'), Skip)

    def test_unrecoverable_fields_roll_back_to_the_handoff(self):
        b = authenticated(RunBuilder())
        handoff = b.handoff('planner', 'returns_agent', 'Return something.', {})
        b.reply('returns_agent', 'Which order?')
        plan = run('repair_handoff', b, handoff, 'context_propagation')
        self.assertEqual(plan.rollback_step, handoff)
        self.assertIn('must include order_id', op(plan, InjectMessage).content)

    def test_structured_reading_reports_computed_counts(self):
        b = RunBuilder()
        b.user('How many T-shirt options are available?')
        b.tool('product_agent', 'get_product_details', VARIANTS, product_id='9523456873')
        step = b.reply('product_agent', 'There are 3 available options.')
        plan = run('structured_reading', b, step, 'MisinterpretationOfToolOutput')
        message = op(plan, InjectMessage)
        self.assertEqual((message.role, message.name), ('tool', 'get_product_details'))
        self.assertIn('2 entries of variants have available=true', message.content)
        self.assertIn('Corrections', message.content)

    def test_orphan_tool_results_can_be_read_but_not_requeried(self):
        b = RunBuilder()
        b.user('How many T-shirt options are available?')
        b.result('get_product_details', VARIANTS)                       # logged without its call
        step = b.reply('assistant', 'There are 3 available T-shirt options.')
        [orphan] = b.trace().tool_exchanges()
        self.assertTrue(orphan.orphan)
        plan = run('structured_reading', b, step, 'MisinterpretationOfToolOutput')
        self.assertIn('2 entries of variants have available=true', op(plan, InjectMessage).content)
        self.assertIsInstance(run('cross_source_check', b, step, 'MisinterpretationOfToolOutput'), Skip)

    def test_independent_reading_keeps_only_confirmed_values(self):
        b = RunBuilder()
        b.user('How many T-shirt options are available?')
        b.tool('product_agent', 'get_product_details', VARIANTS, product_id='9523456873')
        step = b.reply('product_agent', 'There are 3 available options.')
        llm = ScriptedLLM(['{"values": {"available": 2, "warehouse": "Oslo"}, "answer": "2"}'])
        plan = run('independent_reading', b, step, 'MisinterpretationOfToolOutput', llm=llm)
        self.assertIn('available = 2', op(plan, InjectMessage).content)
        self.assertIn('discarded (not in output): warehouse = "Oslo"', plan.evidence)
        self.assertIsInstance(run('independent_reading', b, step, 'MisinterpretationOfToolOutput'), Skip)
        broken = ScriptedLLM(['not json'])
        self.assertIsInstance(run('independent_reading', b, step, 'MisinterpretationOfToolOutput', llm=broken), Skip)

    def test_cross_source_check_requeries(self):
        b, step = order_run('Status: delivered yesterday.')
        plan = run('cross_source_check', b, step, 'MisinterpretationOfToolOutput')
        self.assertEqual(op(plan, RequeryTool).arguments, {'order_id': '#W1000005'})


if __name__ == '__main__':
    unittest.main()
