import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from repair_strategies.llm import LLMError, OllamaClient, ScriptedLLM, ollama_url, parse_json_object
from repair_strategies.policy import intent, plan_state, rules
from repair_strategies.trace import ToolCall

from .helpers import ORDER, RunBuilder, authenticated, retail

RETURN = {'order_id': '#W1000005', 'item_ids': ['2000000001'], 'payment_method_id': 'credit_card_4421'}


def confirmed_return():
    b = authenticated(RunBuilder())
    b.reply('planner', 'Please confirm the following action: return_delivered_order_items '
            + json.dumps(RETURN))
    b.user('yes')
    b.handoff('planner', 'returns_agent', 'User confirmed the return.', dict(RETURN, user_id='ava_lee_2203'))
    b.tool('returns_agent', 'get_order_details', ORDER, order_id='#W1000005')
    return b


class RuleCompileTest(unittest.TestCase):
    def test_retail_rules_compile(self):
        gate = rules.RuleGate.for_task(retail())
        self.assertEqual(len(gate.rules), 5)
        confirm = next(r for r in gate.rules if r.kind == rules.CONFIRM)
        self.assertNotIn('transfer_to_human_agents', confirm.tools)
        self.assertEqual(rules.compile_rules([confirm.to_dict()], retail())[0], confirm)

    def test_write_alias_and_invalid_specs(self):
        [rule] = rules.compile_rules([{'id': 'w', 'kind': rules.FORBIDDEN, 'tools': ['@write']}], retail())
        self.assertEqual(rule.tools, retail().write_tools)
        bad = [
            [{'id': 'x', 'kind': 'nope', 'tools': ['calculate']}],
            [{'id': 'x', 'kind': rules.FORBIDDEN, 'tools': ['no_such_tool']}],
            [{'id': 'x', 'kind': rules.ENUM, 'tool': 'cancel_pending_order', 'argument': 'reason'}],
            [{'id': 'x', 'kind': rules.FORBIDDEN, 'tools': []}],
            [{'id': 'x', 'kind': rules.FORBIDDEN, 'tools': ['calculate']}] * 2,
            [{'id': 'x', 'kind': rules.ONCE, 'tools': ['calculate', 'get_order_details'], 'key': 'a'}],
            ['not an object'],
        ]
        for specs in bad:
            with self.assertRaises(rules.RuleError, msg=specs):
                rules.compile_rules(specs, retail())


class RuleGateTest(unittest.TestCase):
    def setUp(self):
        self.gate = rules.RuleGate.for_task(retail())

    def ids(self, trace, call, step=None):
        return sorted(v.rule_id for v in self.gate.check(trace, step or trace.last_index + 1, call))

    def test_confirmed_grounded_return_passes(self):
        trace = confirmed_return().trace()
        self.assertEqual(self.ids(trace, ToolCall('return_delivered_order_items', RETURN)), [])

    def test_missing_confirmation_and_authentication(self):
        b = RunBuilder()
        b.user('Return the water bottle from #W1000005, card credit_card_4421, item 2000000001.')
        self.assertEqual(self.ids(b.trace(), ToolCall('return_delivered_order_items', RETURN)),
                         ['authenticate_first', 'confirm_before_write'])

    def test_confirmation_must_answer_a_proposal_of_this_action(self):
        b = authenticated(RunBuilder())
        b.reply('planner', 'Shall I look up your orders?')
        b.user('yes')
        self.assertIn('confirm_before_write', self.ids(b.trace(), ToolCall('return_delivered_order_items', RETURN)))

    def test_confirming_a_lookup_does_not_confirm_a_write(self):
        b = authenticated(RunBuilder())
        b.reply('planner', 'Shall I look up order #W1000005?')
        b.user('yes')
        cancel = ToolCall('cancel_pending_order', {'order_id': '#W1000005', 'reason': 'no longer needed'})
        self.assertIn('confirm_before_write', self.ids(b.trace(), cancel))
        b.reply('planner', 'Shall I cancel order #W1000005 because you no longer need it?')
        b.user('yes please')
        self.assertNotIn('confirm_before_write', self.ids(b.trace(), cancel))
        address = ToolCall('modify_pending_order_address', {'order_id': '#W1000005', 'address1': '5 Pine St'})
        b.reply('planner', 'Shall I modify order #W1000005?')
        b.user('yes')
        self.assertIn('confirm_before_write', self.ids(b.trace(), address))     # "modify" without "address"

    def test_enum_once_and_grounding(self):
        trace = confirmed_return().trace()
        cancel = ToolCall('cancel_pending_order', {'order_id': '#W1000005', 'reason': 'changed mind'})
        self.assertIn('cancel_reason', self.ids(trace, cancel))
        invented = ToolCall('return_delivered_order_items', dict(RETURN, item_ids=['2000000999']))
        self.assertEqual(self.ids(trace, invented), ['ids_from_evidence'])
        b = confirmed_return()
        args = {'order_id': '#W1000005', 'item_ids': ['2000000001'], 'new_item_ids': ['2000000001'],
                'payment_method_id': 'credit_card_4421'}
        b.reply('planner', 'Please confirm: modify_pending_order_items ' + json.dumps(args))
        b.user('yes')
        b.tool('order_agent', 'modify_pending_order_items', '{"status": "pending"}', **args)
        self.assertIn('modify_items_once', self.ids(b.trace(), ToolCall('modify_pending_order_items', args)))

    def test_audit_checks_each_call_against_its_prefix(self):
        b = RunBuilder()
        b.user('cancel #W1000005')
        b.call('order_agent', 'cancel_pending_order', order_id='#W1000005', reason='no longer needed')
        found = self.gate.audit(b.trace())
        self.assertEqual(sorted(v.rule_id for v in found), ['authenticate_first', 'confirm_before_write'])
        self.assertIn('step 2', found[0].describe())

    def test_affirmatives(self):
        for text in ('Yes', 'yes please', 'Sure, go ahead', "That's correct", 'ok'):
            self.assertTrue(rules.is_affirmative(text), text)
        for text in ('no', 'not yet', 'why?', 'yesterday was fine'):
            self.assertFalse(rules.is_affirmative(text), text)


class PlanStateTest(unittest.TestCase):
    def test_confirmed_obligations_and_tracking(self):
        b = confirmed_return()
        [obligation] = plan_state.confirmed_obligations(b.trace(), retail())
        self.assertEqual((obligation.tool, obligation.source), ('return_delivered_order_items', 'confirmed'))
        state = plan_state.track(b.trace(), [obligation])
        self.assertEqual([p.status for p in state.items], [plan_state.PENDING])
        self.assertIn('[PENDING]', state.render())
        b.tool('returns_agent', 'return_delivered_order_items', 'Error: payment method not found', **RETURN)
        self.assertEqual(plan_state.track(b.trace(), [obligation]).items[0].status, plan_state.FAILED)
        b.tool('returns_agent', 'return_delivered_order_items', '{"status": "return requested"}',
               **dict(RETURN, item_ids=['2000000001']))
        state = plan_state.track(b.trace(), [obligation])
        self.assertTrue(state.complete)
        self.assertEqual(plan_state.Obligation.from_dict(obligation.to_dict()), obligation)

    def test_unanswered_or_declined_proposals_are_not_obligations(self):
        b = authenticated(RunBuilder())
        b.reply('planner', 'Please confirm: cancel_pending_order {"order_id": "#W1000005"}')
        b.user('no, keep it')
        self.assertEqual(plan_state.confirmed_obligations(b.trace(), retail()), ())

    def test_list_arguments_match_in_any_order(self):
        o = plan_state.Obligation('return_delivered_order_items', {'item_ids': ['1', '2']})
        self.assertTrue(o.matches(ToolCall('return_delivered_order_items', {'item_ids': ['2', '1']})))
        self.assertFalse(o.matches(ToolCall('return_delivered_order_items', {'item_ids': ['1']})))


class IntentTest(unittest.TestCase):
    def test_rule_based_contract(self):
        b = confirmed_return()
        contract = intent.RuleBasedExtractor().extract(b.trace(), retail(), b.last + 1)
        self.assertIn('#w1000005', contract.entities)
        self.assertIn('credit_card_4421', contract.entities)          # reached through get_user_details
        self.assertEqual([o.tool for o in contract.required], ['return_delivered_order_items'])
        self.assertEqual(contract.validate(ToolCall('return_delivered_order_items', RETURN), retail()), [])
        issues = contract.validate(ToolCall('cancel_pending_order', {'order_id': '#W1000009',
                                                                     'reason': 'no longer needed'}), retail())
        self.assertEqual(len(issues), 1)
        issues = contract.validate(ToolCall('return_delivered_order_items', dict(RETURN, order_id='#W7777777')),
                                   retail())
        self.assertIn('outside the user', issues[0])
        self.assertEqual(intent.IntentContract.from_dict(contract.to_dict()), contract)
        self.assertIn('Goal:', contract.render())

    def test_llm_contract_is_validated_and_merged(self):
        reply = json.dumps({'goal': 'Return the bottle', 'forbidden_tools': ['cancel_pending_order', 'fly'],
                            'required_actions': [{'tool': 'teleport'}, {'tool': 'return_delivered_order_items'}],
                            'constraints': ['refund to the original card']})
        b = confirmed_return()
        contract = intent.LLMContractExtractor(ScriptedLLM([f'Sure: {reply}'])).extract(b.trace(), retail(), b.last + 1)
        self.assertEqual(contract.source, 'llm')
        self.assertEqual(contract.forbidden_tools, {'cancel_pending_order'})
        self.assertEqual(len(contract.required), 1)                     # merged with the confirmed one
        self.assertIn("dropped unknown tool 'teleport'", contract.notes)
        self.assertTrue(contract.validate(ToolCall('cancel_pending_order', {}), retail()))

    def test_llm_failure_falls_back_to_rules(self):
        b = confirmed_return()
        contract = intent.LLMContractExtractor(ScriptedLLM(['no json here'])).extract(b.trace(), retail(), b.last + 1)
        self.assertEqual(contract.source, 'rules')
        self.assertTrue(contract.notes[0].startswith('LLM contract failed'))


class _Handler(BaseHTTPRequestHandler):
    reply = {}
    status = 200
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        type(self).seen.append((self.path, self.headers.get('Authorization'), body))
        data = json.dumps(type(self).reply).encode()
        self.send_response(type(self).status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class OllamaClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(('127.0.0.1', 0), _Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f'http://127.0.0.1:{cls.server.server_address[1]}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def reply(self, status=200, **payload):
        _Handler.status, _Handler.seen = status, []
        _Handler.reply = payload

    def test_native_chat_request_and_usage(self):
        self.reply(message={'role': 'assistant', 'content': '{"a": 1}'}, done=True, done_reason='stop',
                   prompt_eval_count=7, eval_count=3)
        client = OllamaClient('qwen2.5:7b', base_url=self.url, num_ctx=8192)
        reply = client.complete([{'role': 'user', 'content': 'hi'}], purpose='test')
        self.assertEqual((reply.text, reply.tokens), ('{"a": 1}', 10))
        path, auth, body = _Handler.seen[0]
        self.assertEqual((path, auth), ('/api/chat', None))
        self.assertEqual((body['model'], body['stream'], body['format']), ('qwen2.5:7b', False, 'json'))
        self.assertEqual(body['options'], {'num_ctx': 8192, 'temperature': 0.0})
        self.assertEqual((body['truncate'], body['shift']), (False, False))

    def test_errors_explain_what_to_do(self):
        client = OllamaClient('missing-model', base_url=self.url)
        self.reply(404, error='model "missing-model" not found')
        with self.assertRaisesRegex(LLMError, 'ollama pull missing-model'):
            client.complete([], purpose='t')
        self.reply(message={'content': 'x'}, done=True, done_reason='length')
        with self.assertRaisesRegex(LLMError, 'raise num_ctx'):
            client.complete([], purpose='t')
        self.reply(error='out of memory')
        with self.assertRaisesRegex(LLMError, 'out of memory'):
            client.complete([], purpose='t')
        self.reply(message={'content': 'x'}, done=False)
        with self.assertRaisesRegex(LLMError, 'incomplete'):
            client.complete([], purpose='t')
        self.reply(unexpected=True)
        with self.assertRaisesRegex(LLMError, 'shape'):
            client.complete([], purpose='t')
        with self.assertRaisesRegex(LLMError, 'ollama serve'):
            OllamaClient('m', base_url='http://127.0.0.1:9', timeout=2).complete([], purpose='t')

    def test_server_url_resolution(self):
        with mock.patch.dict(os.environ, {'OLLAMA_HOST': 'gpu-box:11434'}):
            self.assertEqual(ollama_url(), 'http://gpu-box:11434')
            self.assertEqual(ollama_url('https://lab.example/ollama/'), 'https://lab.example/ollama')
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ollama_url(), 'http://127.0.0.1:11434')
        with self.assertRaises(ValueError):
            ollama_url('ftp://x')
        with self.assertRaises(ValueError):
            OllamaClient('')

    def test_scripted_and_json_parsing(self):
        llm = ScriptedLLM(lambda messages, purpose: purpose)
        self.assertEqual(llm.complete([], purpose='p').text, 'p')
        with self.assertRaises(LLMError):
            ScriptedLLM([]).complete([], purpose='p')
        self.assertEqual(parse_json_object('```json\n{"a": [1]}\n```'), {'a': [1]})
        for bad in ('none', '{bad}', ''):
            with self.assertRaises(LLMError):
                parse_json_object(bad)


if __name__ == '__main__':
    unittest.main()
