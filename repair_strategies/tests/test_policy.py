import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from repair_strategies.llm import LLMError, OpenAICompatibleClient, ScriptedLLM, parse_json_object
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


class LLMClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(('127.0.0.1', 0), _Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f'http://127.0.0.1:{cls.server.server_address[1]}/v1'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_request_and_usage(self):
        _Handler.status, _Handler.seen = 200, []
        _Handler.reply = {'choices': [{'message': {'content': '{"a": 1}'}, 'finish_reason': 'stop'}],
                          'usage': {'prompt_tokens': 7, 'completion_tokens': 3}}
        with mock.patch.dict(os.environ, {'TEST_KEY': 'k-123'}):
            client = OpenAICompatibleClient(self.url, 'qwen2.5:7b', api_key_env='TEST_KEY', json_mode=True)
        reply = client.complete([{'role': 'user', 'content': 'hi'}], purpose='test')
        self.assertEqual((reply.text, reply.tokens), ('{"a": 1}', 10))
        path, auth, body = _Handler.seen[0]
        self.assertEqual((path, auth, body['model']), ('/v1/chat/completions', 'Bearer k-123', 'qwen2.5:7b'))
        self.assertEqual(body['response_format'], {'type': 'json_object'})

    def test_errors(self):
        client = OpenAICompatibleClient(self.url, 'm')
        _Handler.status, _Handler.reply = 500, {'error': 'boom'}
        with self.assertRaisesRegex(LLMError, 'HTTP 500'):
            client.complete([], purpose='t')
        _Handler.status, _Handler.reply = 200, {'choices': [{'message': {'content': 'x'}, 'finish_reason': 'length'}]}
        with self.assertRaisesRegex(LLMError, 'truncated'):
            client.complete([], purpose='t')
        _Handler.reply = {'unexpected': True}
        with self.assertRaisesRegex(LLMError, 'shape'):
            client.complete([], purpose='t')
        with self.assertRaisesRegex(LLMError, 'cannot reach'):
            OpenAICompatibleClient('http://127.0.0.1:9', 'm', timeout=2).complete([], purpose='t')
        with self.assertRaises(LLMError):
            OpenAICompatibleClient(self.url, 'm', api_key_env='SURELY_UNSET_VARIABLE_123')
        with self.assertRaises(ValueError):
            OpenAICompatibleClient('ftp://x', 'm')

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
