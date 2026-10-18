import unittest
from decimal import Decimal

from repair_strategies.evidence import claims as cl
from repair_strategies.evidence import cycles, schema
from repair_strategies.evidence import tool_output as to
from repair_strategies.trace import ToolCall, Trace

from .helpers import ORDER, RunBuilder, authenticated, retail

VARIANTS = {'variants': {'a': {'available': True, 'price': 10}, 'b': {'available': False, 'price': 12},
                         'c': {'available': True, 'price': 14}}}


class ToolOutputTest(unittest.TestCase):
    def test_parse_kinds(self):
        self.assertEqual(to.parse_output('{"a": 1}').kind, to.JSON)
        self.assertEqual(to.parse_output('Error: order not found').kind, to.ERROR)
        self.assertEqual(to.parse_output('{"error": "boom"}').kind, to.ERROR)
        self.assertEqual(to.parse_output('ava_lee_2203').kind, to.TEXT)
        self.assertEqual(to.parse_output('{not json').kind, to.TEXT)
        self.assertEqual(to.parse_output('  ').kind, to.EMPTY)
        self.assertTrue(to.parse_output('{"error": "x", "a": 1, "b": 2, "c": 3}').ok)

    def test_error_classes(self):
        self.assertEqual(to.error_class('Error: 503 service unavailable'), to.TRANSIENT)
        self.assertEqual(to.error_class('Error: request timed out'), to.TRANSIENT)
        self.assertEqual(to.error_class('Error: endpoint deprecated, use v2'), to.PERSISTENT)
        self.assertEqual(to.error_class('Error: something odd'), to.UNKNOWN)
        self.assertTrue(to.indicates_drift("Error: unexpected keyword argument 'orderId'"))
        self.assertFalse(to.indicates_drift('Error: 503'))

    def test_paths_and_aggregates(self):
        self.assertEqual(to.get_path(ORDER, 'items.1.price'), 120.5)
        self.assertEqual(to.get_path(ORDER, 'payment_history.0.payment_method_id'), 'credit_card_4421')
        self.assertIsNone(to.get_path(ORDER, 'items.9.price'))
        found = {(a.kind, a.key): a.value for a in to.aggregates(VARIANTS)}
        self.assertEqual(found[('size', '')], 3)
        self.assertEqual(found[('count_true', 'available')], 2)
        self.assertEqual(found[('sum', 'price')], 36)
        self.assertIn(Decimal('24'), to.subset_sums([Decimal(10), Decimal(14), Decimal(12)]))
        self.assertEqual(to.to_decimal('$1,204.50'), Decimal('1204.50'))
        self.assertIsNone(to.to_decimal(True))
        self.assertEqual(to.key_tokens('payment_method_ids'), {'payment', 'method', 'id'})
        self.assertEqual([to.singular(w) for w in ('status', 'address', 'entries', 'items')],
                         ['status', 'address', 'entry', 'item'])


def _variants_run(claim):
    b = RunBuilder(instruction='How many options are available?')
    b.user('How many T-shirt options are available?')
    b.tool('product_agent', 'get_product_details', VARIANTS, product_id='9523456873')
    step = b.reply('product_agent', claim)
    return b.trace(), step


class ProvenanceTest(unittest.TestCase):
    def test_count_contradicted_by_computed_flag_count(self):
        trace, step = _variants_run('There are 3 available T-shirt options.')
        [check] = cl.provenance(trace, step).checks
        self.assertEqual(check.status, cl.CONTRADICTED)
        self.assertIn('2 entries of variants have available=true = 2', check.expected[0])

    def test_correct_count_is_grounded(self):
        trace, step = _variants_run('There are 2 available T-shirt options.')
        self.assertTrue(cl.provenance(trace, step).clean)

    def test_amounts_single_and_summed(self):
        b = authenticated(RunBuilder())
        b.tool('returns_agent', 'get_order_details', ORDER, order_id='#W1000005')
        ok = b.reply('returns_agent', 'The refund for the water bottle is $45.04.')
        both = b.reply('returns_agent', 'Returning both items refunds $165.54.')
        wrong = b.reply('returns_agent', 'The refund for the water bottle is $54.04.')
        trace, synonyms = b.trace(), retail().fact_keywords
        self.assertTrue(cl.provenance(trace, ok, synonyms=synonyms).clean)
        self.assertTrue(cl.provenance(trace, both, synonyms=synonyms).clean)
        [check] = cl.provenance(trace, wrong, synonyms=synonyms).faulty
        self.assertEqual(check.status, cl.CONTRADICTED)
        self.assertEqual(check.expected, ('items.0.price = $45.04',))

    def test_amount_without_any_matching_field_is_ungrounded(self):
        b = authenticated(RunBuilder())
        step = b.reply('planner', 'Shipping is $7.25.')
        [check] = cl.provenance(b.trace(), step).faulty
        self.assertEqual(check.status, cl.UNGROUNDED)

    def test_invented_identifiers_and_arguments(self):
        b = authenticated(RunBuilder())
        bad_reply = b.reply('planner', 'I found your order #W9999999.')
        bad_call = b.call('returns_agent', 'return_delivered_order_items', order_id='#W1000005',
                          item_ids=['2000000777'], payment_method_id='credit_card_4421')
        trace = b.trace()
        [check] = cl.provenance(trace, bad_reply).faulty
        self.assertEqual((check.status, check.claim.value), (cl.UNGROUNDED, '#w9999999'))
        faulty = cl.provenance(trace, bad_call).faulty
        self.assertEqual([(c.claim.argument, c.claim.raw) for c in faulty], [('item_ids', '2000000777')])
        self.assertTrue(cl.provenance(trace, bad_call, include_arguments=False).clean)

    def test_own_claims_do_not_ground_later_claims(self):
        b = RunBuilder()
        b.user('status of my order?')
        b.reply('planner', 'Your order is #W5550001.')
        again = b.reply('planner', 'Again, the order is #W5550001.')
        self.assertFalse(cl.provenance(b.trace(), again).clean)

    def test_handoff_context_and_email_are_evidence(self):
        b = RunBuilder()
        b.user('my email is ava.lee5521@example.com')
        b.handoff('planner', 'order_agent', 'cancel it', {'order_id': '#W1000005'})
        step = b.reply('order_agent', 'Cancelling #W1000005 for ava.lee5521@example.com.')
        self.assertTrue(cl.provenance(b.trace(), step).clean)

    def test_bare_numbers_and_list_markers(self):
        found = cl.extract_claims('1. Check the order\n2. Wait 30 minutes', 5)
        self.assertEqual([c.raw for c in found], ['30'])
        trace, step = _variants_run('This takes 30 minutes.')
        [check] = cl.provenance(trace, step).checks
        self.assertEqual(check.status, cl.UNVERIFIABLE)
        self.assertFalse(check.faulty)

    def test_handoff_context_claims_are_checked(self):
        b = RunBuilder()
        b.user('cancel order #W1000005')
        step = b.handoff('planner', 'order_agent', 'cancel', {'order_id': '#W1000006'})
        [check] = cl.provenance(b.trace(), step).faulty
        self.assertEqual(check.claim.argument, 'context.order_id')

    def test_provenance_from_covers_every_later_agent_step(self):
        trace, step = _variants_run('There are 3 available T-shirt options.')
        reports = cl.provenance_from(trace, 1)
        self.assertEqual([r.step_index for r in reports], [2, step])


class SchemaTest(unittest.TestCase):
    def setUp(self):
        self.task = retail()

    def codes(self, call, **kw):
        return [(i.code, i.suggestion) for i in schema.validate_call(call, self.task, **kw)]

    def test_valid_call(self):
        call = ToolCall('cancel_pending_order', {'order_id': '#W1', 'reason': 'no longer needed'})
        self.assertEqual(self.codes(call), [])

    def test_missing_unexpected_and_types(self):
        call = ToolCall('return_delivered_order_items', {'order_id': 5, 'item_id': ['1'], 'payment_method_id': 'x'})
        self.assertEqual(sorted(self.codes(call)), sorted([
            (schema.MISSING, None), (schema.UNEXPECTED, 'item_ids'), (schema.TYPE, None)]))
        call = ToolCall('return_delivered_order_items', {'order_id': '#W1', 'item_ids': [1], 'payment_method_id': 'x'})
        self.assertEqual(self.codes(call), [(schema.TYPE, None)])

    def test_unknown_tool_restricted_tools_and_malformed(self):
        self.assertEqual(self.codes(ToolCall('get_order_detail', {})), [(schema.UNKNOWN_TOOL, 'get_order_details')])
        call = ToolCall('cancel_pending_order', {'order_id': '#W1', 'reason': 'no longer needed'})
        self.assertEqual(self.codes(call, allowed=('get_order_details',))[0][0], schema.UNKNOWN_TOOL)
        self.assertEqual(self.codes(ToolCall('get_order_details', {'__raw__': '{'})), [(schema.MALFORMED, None)])

    def test_enum_and_examples(self):
        issues = schema.validate('no longer neded', {'type': 'string', 'enum': ['no longer needed', 'ordered by mistake']})
        self.assertEqual(issues[0].suggestion, 'no longer needed')
        self.assertIn('did you mean', issues[0].describe())
        example = schema.example_arguments(self.task.tool('exchange_delivered_order_items'))
        self.assertEqual(example['item_ids'], ['<string>'])
        self.assertEqual(schema.validate(True, {'type': 'integer'})[0].code, schema.TYPE)


class CycleTest(unittest.TestCase):
    def loop_run(self):
        b = authenticated(RunBuilder())
        for _ in range(3):
            b.tool('order_agent', 'get_order_details', 'Error: still processing, retry', order_id='#W1000005')
        return b

    def test_repeated_call_with_same_result(self):
        [loop] = cycles.detect(self.loop_run().trace())
        self.assertEqual((loop.kind, loop.steps, loop.first_repeat), (cycles.CALL, (6, 8, 10), 8))
        self.assertIn('3 times', loop.describe())

    def test_changing_results_are_not_a_loop(self):
        b = authenticated(RunBuilder())
        for i in range(3):
            b.tool('order_agent', 'get_order_details', f'{{"n": {i}}}', order_id='#W1000005')
        self.assertEqual(cycles.detect(b.trace()), ())

    def test_write_resets_but_keeps_completed_loops(self):
        b = self.loop_run()
        b.tool('order_agent', 'cancel_pending_order', '{"status": "cancelled"}', order_id='#W1000005',
               reason='no longer needed')
        b.tool('order_agent', 'get_order_details', 'Error: still processing, retry', order_id='#W1000005')
        loops = cycles.detect(b.trace(), write_tools=retail().write_tools)
        self.assertEqual([loop.steps for loop in loops], [(6, 8, 10)])

    def test_handoff_and_message_loops(self):
        b = RunBuilder()
        b.user('help')
        for _ in range(3):
            b.handoff('planner', 'order_agent', 'do it', {'order_id': '#W1'})
            b.reply('order_agent', 'I need more information.')
        kinds = sorted(loop.kind for loop in cycles.detect(b.trace()))
        self.assertEqual(kinds, [cycles.HANDOFF, cycles.MESSAGE])
        self.assertEqual(cycles.loop_at(b.trace(), 4).kind, cycles.HANDOFF)
        self.assertIsNone(cycles.loop_at(Trace.from_messages([{'role': 'user', 'content': 'x'}]), None))
        with self.assertRaises(ValueError):
            cycles.detect(b.trace(), min_repeats=1)


if __name__ == '__main__':
    unittest.main()
