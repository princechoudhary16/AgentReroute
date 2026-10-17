import json
import unittest

from benchmark.env.retail import RetailDB, gold_state_hash
from benchmark.env.tasks import load_fixture, parse_tau_tasks


class RetailToolsTest(unittest.TestCase):
    def setUp(self):
        self.domain = load_fixture()
        self.db = RetailDB(self.domain.data)

    def test_cancel_refunds_and_rejects_bad_reason(self):
        self.assertTrue(self.db.call('cancel_pending_order', {'order_id': '#W1000001', 'reason': 'bored'})
                        .startswith('Error'))
        order = json.loads(self.db.call('cancel_pending_order', {'order_id': '#W1000001', 'reason': 'no longer needed'}))
        self.assertEqual(order['status'], 'cancelled')
        self.assertEqual(order['payment_history'][-1]['transaction_type'], 'refund')
        self.assertTrue(self.db.call('cancel_pending_order', {'order_id': '#W1000001', 'reason': 'no longer needed'})
                        .startswith('Error'))

    def test_return_requires_original_or_gift_card(self):
        bad = self.db.call('return_delivered_order_items', {'order_id': '#W1000005', 'item_ids': ['2000000001'],
                                                            'payment_method_id': 'nope'})
        self.assertTrue(bad.startswith('Error'))
        ok = json.loads(self.db.call('return_delivered_order_items', {
            'order_id': '#W1000005', 'item_ids': ['2000000001'], 'payment_method_id': 'gift_card_8801234'}))
        self.assertEqual(ok['status'], 'return requested')

    def test_modify_items_charges_difference_once(self):
        args = {'order_id': '#W1000001', 'item_ids': ['2000000001'], 'new_item_ids': ['2000000002'],
                'payment_method_id': 'gift_card_3036421'}
        order = json.loads(self.db.call('modify_pending_order_items', args))
        self.assertEqual(order['status'], 'pending (item modified)')
        self.assertEqual(self.db.users['mia_garcia_4516']['payment_methods']['gift_card_3036421']['balance'], 112.0)
        self.assertTrue(self.db.call('modify_pending_order_items', args).startswith('Error'))

    def test_unavailable_variant_and_argument_validation(self):
        self.assertIn('not found or available', self.db.call('exchange_delivered_order_items', {
            'order_id': '#W1000003', 'item_ids': ['1000000002'], 'new_item_ids': ['1000000003'],
            'payment_method_id': 'paypal_5727330'}))
        self.assertIn('missing', self.db.call('get_order_details', {}))
        self.assertEqual(self.db.call('calculate', {'expression': '(2+3)*4/2'}), '10.0')
        self.assertTrue(self.db.call('calculate', {'expression': '__import__("os")'}).startswith('Error'))

    def test_gold_hash_is_deterministic_and_reads_do_not_change_state(self):
        task = self.domain.task('fixture-cancel')
        self.assertEqual(gold_state_hash(self.domain.data, task['actions']),
                         gold_state_hash(self.domain.data, task['actions']))
        before = self.db.state_hash()
        self.db.call('get_user_details', {'user_id': 'mia_garcia_4516'})
        self.assertEqual(before, self.db.state_hash())


class TauTaskParsingTest(unittest.TestCase):
    def test_parses_without_executing(self):
        source = '''from tau_bench.types import Task, Action
TASKS_TEST = [
    Task(annotator="0", user_id="u_1", instruction="Cancel it.",
         actions=[Action(name="cancel_pending_order", kwargs={"order_id": "#W1", "reason": "no longer needed"})],
         outputs=[]),
    Task(annotator="1", user_id="u_2", instruction="Ask.", actions=[], outputs=["42"]),
]
'''
        tasks = parse_tau_tasks(source)
        self.assertEqual([t['task_id'] for t in tasks], ['retail-000', 'retail-001'])
        self.assertEqual(tasks[0]['actions'][0]['kwargs']['order_id'], '#W1')
        self.assertEqual(tasks[1]['outputs'], ['42'])
        with self.assertRaises(ValueError):
            parse_tau_tasks('X = [os.system("echo hi")]')


if __name__ == '__main__':
    unittest.main()
