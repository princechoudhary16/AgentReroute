import io
import json
import threading
import unittest
from email.message import Message
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock
from unittest.mock import patch
from urllib import error, request
from agentrx.providers import HTTPProvider, OllamaProvider, ProviderError, _parse_json_object, env_key


class Handler(BaseHTTPRequestHandler):
    records = []
    fail_once = False
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.records.append({'body': body, 'path': self.path, 'auth': self.headers.get('Authorization')})
        if self.fail_once and len(self.records) == 1:
            self.send_response(429); self.end_headers(); return
        data = json.dumps({'id':'response-id', 'model':'actual-model', 'usage': {'total_tokens': 22},
                'choices':[{'finish_reason':'stop', 'message': {'content': '{"constraints": []}'}}]}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(data)
    def log_message(self, *args): pass


class OllamaHandler(BaseHTTPRequestHandler):
    records = []
    prompt_tokens = 120
    status_code = 200
    error_body = b''
    model_digest = 'a' * 64
    def do_GET(self):
        data = ({'version':'test-ollama'} if self.path == '/api/version' else
                {'models':[{'name':'model:latest','model':'model:latest','digest':self.model_digest,
                            'details':{'quantization_level':'Q4_K_M'}}]})
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(data).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.records.append({'body': body, 'path': self.path})
        if self.status_code != 200:
            self.send_response(self.status_code); self.end_headers(); self.wfile.write(self.error_body); return
        data = json.dumps({'model': body['model'], 'done': True, 'done_reason': 'stop', 'prompt_eval_count': self.prompt_tokens,
                           'eval_count': 8, 'eval_duration': 1_000_000,
                           'message': {'role': 'assistant', 'content': '{"constraints": []}'}}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(data)
    def log_message(self, *args): pass


class ProviderTests(unittest.TestCase):
    def setUp(self):
        Handler.records = []; Handler.fail_once = False
        self.server = HTTPServer(('127.0.0.1',0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'
    def tearDown(self): self.server.shutdown(); self.server.server_close(); self.thread.join()

    def test_http_client_sends_json_and_retains_usage_and_actual_model(self):
        p = HTTPProvider('gpt-5', self.url, 'test-key', retries=0)
        self.assertEqual(p.complete('global', {'policy': '', 'tools': {}}), {'constraints': []})
        self.assertEqual(Handler.records[0]['path'], '/v1/chat/completions')
        self.assertEqual(Handler.records[0]['auth'], 'Bearer test-key')
        self.assertNotIn('temperature', Handler.records[0]['body'])
        self.assertEqual(p.last_telemetry['model'], 'actual-model')
        self.assertEqual(p.last_telemetry['usage']['total_tokens'], 22)

    def test_transient_http_error_is_retried(self):
        Handler.fail_once = True
        p = HTTPProvider('gpt-5', self.url, 'key', retries=1, retry_delay=0)
        self.assertEqual(p.complete('global', {'policy': '', 'tools': {}}), {'constraints': []})
        self.assertEqual(len(Handler.records), 2)

    def test_missing_credentials_fail_preflight(self):
        # sys.platform: no HKCU\Environment fallback, which would find a real user-level key on Windows.
        with patch.dict('os.environ', {}, clear=True), patch('agentrx.providers.sys.platform', 'linux'):
            with self.assertRaisesRegex(ProviderError, 'OPENAI_API_KEY'):
                HTTPProvider.from_config({'provider': 'openai', 'model': 'gpt-5'})

    def test_gemini_provider_uses_openai_compatible_endpoint_and_key(self):
        p = HTTPProvider.from_config({'provider': 'gemini', 'api_key': 'gem-key', 'base_url': self.url, 'retries': 0})
        self.assertEqual(p.complete('global', {'policy': '', 'tools': {}}), {'constraints': []})
        self.assertEqual(Handler.records[0]['path'], '/v1/chat/completions')
        self.assertEqual(Handler.records[0]['auth'], 'Bearer gem-key')

    def test_gemini_missing_credentials_fail(self):
        with patch.dict('os.environ', {}, clear=True), patch('agentrx.providers.sys.platform', 'linux'):
            with self.assertRaisesRegex(ProviderError, 'GEMINI_API_KEY'):
                HTTPProvider.from_config({'provider': 'gemini'})


class OllamaProviderTests(unittest.TestCase):
    def setUp(self):
        OllamaHandler.records = []; OllamaHandler.prompt_tokens = 120
        OllamaHandler.status_code = 200; OllamaHandler.error_body = b''
        OllamaHandler.model_digest = 'a' * 64
        self.server = HTTPServer(('127.0.0.1',0), OllamaHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self): self.server.shutdown(); self.server.server_close(); self.thread.join()

    def test_native_chat_sets_context_json_and_usage_without_credentials(self):
        with patch.dict('os.environ', {}, clear=True):
            p = HTTPProvider.from_config({'provider': 'ollama', 'model': 'qwen3:8b', 'base_url': self.url,
                                          'num_ctx': 8192, 'think': False})
        self.assertIsInstance(p, OllamaProvider)
        self.assertEqual(p.complete('global', {'policy': '', 'tools': {}}), {'constraints': []})
        sent = OllamaHandler.records[0]
        self.assertEqual(sent['path'], '/api/chat')
        self.assertEqual(sent['body']['format'], 'json')
        self.assertEqual(sent['body']['options']['num_ctx'], 8192)
        self.assertIs(sent['body']['think'], False)
        self.assertEqual(p.last_telemetry['usage']['total_tokens'], 128)

    def test_ranked_judge_uses_native_schema_and_legacy_generation_uses_json(self):
        from tests.test_track_b import trace
        from agentrx.prompts import ranked_response_schema
        p=OllamaProvider('model',self.url,num_ctx=8192)
        payload={'trajectory':trace(),'max_candidates':2,'boundary':{},'checklist':True}
        p.complete('ranked_judge',payload)
        self.assertEqual(OllamaHandler.records[-1]['body']['format'],ranked_response_schema(payload))
        p.complete('global',{'policy':'','tools':{}})
        self.assertEqual(OllamaHandler.records[-1]['body']['format'],'json')

    def test_context_overflow_is_an_error_not_silent_truncation(self):
        OllamaHandler.prompt_tokens = 4096
        p = OllamaProvider('qwen2.5-coder:7b', self.url, num_ctx=4096)
        with self.assertRaisesRegex(ProviderError, 'Context window'):
            p.complete('global', {'policy': '', 'tools': {}})

    def test_remote_plain_http_is_rejected(self):
        with self.assertRaisesRegex(ProviderError, 'HTTPS'):
            OllamaProvider('qwen3:8b', 'http://example.com:11434')

    def test_native_chat_disables_input_truncation_and_context_shifting(self):
        p=OllamaProvider('model',self.url,num_ctx=8192)
        p.complete('global',{'policy':'','tools':{}})
        sent=OllamaHandler.records[0]['body']
        self.assertIs(sent.get('truncate'),False)
        self.assertIs(sent.get('shift'),False)

    def test_nonfinal_native_response_is_not_accepted(self):
        p=OllamaProvider('model',self.url)
        p._post=lambda body:{'done':False,'message':{'content':'{"constraints": []}'}}
        with self.assertRaisesRegex(ProviderError,'Incomplete'):
            p.complete('global',{'policy':'','tools':{}})

    def test_native_malformed_envelope_is_a_provider_error(self):
        for response in ([],{'done':True,'prompt_eval_count':'bad','message':{'content':'{}'}}):
            with self.subTest(response=response):
                p=OllamaProvider('model',self.url)
                p._post=lambda body:response
                with self.assertRaises(ProviderError):
                    p.complete('global',{'policy':'','tools':{}})

    def test_model_and_endpoint_environment_overrides_are_respected(self):
        with patch.dict('os.environ',{'TEST_MODEL':'local-version','TEST_URL':self.url,'TEST_KEY':'test'}):
            p=HTTPProvider.from_config({'provider':'openai-compatible','model':'default',
                'model_env':'TEST_MODEL','base_url':'https://example.com/v1',
                'base_url_env':'TEST_URL','api_key_env':'TEST_KEY'})
        self.assertEqual(p.model,'local-version')
        self.assertEqual(p.identity['base_url'],self.url)

    def test_explicit_ollama_endpoint_environment_must_exist(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(ProviderError,'CUSTOM_OLLAMA_URL'):
                HTTPProvider.from_config({'provider':'ollama','base_url_env':'CUSTOM_OLLAMA_URL'})

    def test_server_context_rejection_is_actionable_and_not_retried(self):
        OllamaHandler.status_code=400
        OllamaHandler.error_body=b'{"error":"exceed_context_size_error: request exceeds available context size"}'
        p=OllamaProvider('model',self.url,retries=1,retry_delay=0)
        with self.assertRaisesRegex(ProviderError,'context.*num_ctx'):
            p.complete('global',{'policy':'','tools':{}})
        self.assertEqual(len(OllamaHandler.records),1)

    def test_changed_ollama_weights_do_not_reuse_tag_only_cached_outputs(self):
        import tempfile
        from agentrx.providers import CachedProvider
        with tempfile.TemporaryDirectory() as d:
            p=OllamaProvider('model',self.url)
            cached=CachedProvider(p,d,{'rep':0})
            cached.complete('global',{'policy':'','tools':{}})
            self.assertEqual(cached.identity['model_digest'],'a'*64)
            self.assertEqual(cached.identity['ollama_version'],'test-ollama')
            CachedProvider(OllamaProvider('model',self.url),d,{'rep':0}).complete('global',{'policy':'','tools':{}})
            self.assertEqual(len(OllamaHandler.records),1)
            OllamaHandler.model_digest='b'*64
            CachedProvider(OllamaProvider('model',self.url),d,{'rep':0}).complete('global',{'policy':'','tools':{}})
            self.assertEqual(len(OllamaHandler.records),2)

    def test_generic_json_chat_uses_custom_messages_and_existing_native_safeguards(self):
        p=OllamaProvider('model',self.url,num_ctx=8192)
        custom=[{'role':'system','content':'Return dependency JSON'}, {'role':'user','content':'{"tools":[]}'}]
        self.assertEqual(p.complete_messages(custom),{'constraints':[]})
        body=OllamaHandler.records[0]['body']
        self.assertEqual(body['messages'],custom)
        self.assertIs(body['truncate'],False)
        self.assertIs(body['shift'],False)
        self.assertEqual(body['format'],'json')

    def test_native_chat_can_constrain_the_json_response_shape(self):
        p=OllamaProvider('model',self.url,num_ctx=8192)
        schema={'type':'object','properties':{'rules':{'type':'array'}},'required':['rules']}
        p.complete_messages([{'role':'user','content':'Return rules'}],response_schema=schema)
        self.assertEqual(OllamaHandler.records[0]['body']['format'],schema)


# OpenRouter hardening of HTTPProvider (offline: scripted opener, no network)
BASE_URL = 'https://openrouter.ai/api/v1'
MODEL = 'minimax/minimax-m3.1-flash:free'
ANSWER = {'step_index': 2, 'agent': 'planner', 'category': 'InvalidInvocation', 'rationale': 'ok'}


def http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers['Retry-After'] = retry_after
    return error.HTTPError(BASE_URL + '/chat/completions', code, 'error', headers, io.BytesIO(b'{}'))


def completion(content):
    return io.BytesIO(json.dumps({'id': 'gen-1', 'model': MODEL, 'usage': {'total_tokens': 9},
                                  'choices': [{'finish_reason': 'stop',
                                               'message': {'content': content}}]}).encode())


class FakeOpener:
    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def open(self, req, timeout=None):
        self.requests.append(req)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class OpenRouterHardeningTests(unittest.TestCase):
    def test_strips_think_tags_and_markdown_fences(self):
        raw = json.dumps(ANSWER)
        self.assertEqual(_parse_json_object(raw), ANSWER)
        self.assertEqual(_parse_json_object('<think>reasoning</think>\n```json\n' + raw + '\n```'), ANSWER)
        self.assertEqual(_parse_json_object('<think>a\n{"x": 1}\nb</think>```\n' + raw + '```'), ANSWER)
        self.assertEqual(_parse_json_object('Here is my verdict:\n' + raw + '\nHope this helps.'), ANSWER)
        with self.assertRaisesRegex(ProviderError, r'Malformed model JSON response \(JSONDecodeError\)'):
            _parse_json_object('<think>only thoughts</think> no json here')
        with self.assertRaisesRegex(ProviderError, 'Model must return a JSON object'):
            _parse_json_object('```json\n[1, 2]\n```')

    def test_cloudflare_headers_and_exponential_backoff(self):
        opener = FakeOpener([http_error(429), http_error(429), http_error(503, retry_after='3'),
                             http_error(429, retry_after='999'), http_error(400),
                             completion('<think>x</think>```json\n' + json.dumps(ANSWER) + '\n```')])
        provider = HTTPProvider(MODEL, BASE_URL, 'sk-or-test', provider='openai-compatible', retries=4,
                                retry_delay=2, settings={'reasoning_effort': 'low', 'max_tokens': 4096})
        with mock.patch.object(request, 'build_opener', return_value=opener), \
                mock.patch('agentrx.providers.time.sleep') as sleep:
            answer = provider.complete('judge', {'trajectory': {'steps': []}, 'checklist': False})
        self.assertEqual(answer, ANSWER)
        # 2s * 2**attempt, then numeric Retry-After, capped at 60; the 400 fallback retries without sleeping.
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 4, 3.0, 60.0])
        first = opener.requests[0]
        self.assertEqual(first.full_url, BASE_URL + '/chat/completions')
        self.assertEqual(first.get_header('User-agent'), 'AgentReroute/1.0')
        self.assertEqual(first.get_header('Http-referer'), 'https://github.com/AgentReroute')
        self.assertEqual(first.get_header('X-title'), 'AgentReroute')
        self.assertEqual(first.get_header('Authorization'), 'Bearer sk-or-test')
        bodies = [json.loads(r.data) for r in opener.requests]
        self.assertTrue(all('response_format' in b and b['reasoning_effort'] == 'low' for b in bodies[:5]))
        self.assertNotIn('response_format', bodies[5])
        self.assertNotIn('reasoning_effort', bodies[5])
        self.assertEqual(bodies[5]['max_tokens'], 4096)         # the token bound survives the fallback
        self.assertEqual((provider.last_telemetry['attempts'], provider.last_telemetry['response_format']),
                         (6, False))
        self.assertEqual(provider._backoff(10), 60)

    def test_env_key_registry_fallback_and_empty_override(self):
        reg = mock.MagicMock()
        reg.QueryValueEx.return_value = ('from-hkcu', 1)
        with mock.patch.dict('sys.modules', {'winreg': reg}), mock.patch('agentrx.providers.sys.platform', 'win32'):
            with mock.patch.dict('os.environ', {}, clear=True):
                self.assertEqual(env_key('SOME_KEY'), 'from-hkcu')
            with mock.patch.dict('os.environ', {'SOME_KEY': ''}, clear=True):
                self.assertEqual(env_key('SOME_KEY'), '')         # explicit blank wins: no registry read
            with mock.patch.dict('os.environ', {'SOME_KEY': 'proc'}, clear=True):
                self.assertEqual(env_key('SOME_KEY'), 'proc')
        self.assertEqual(reg.QueryValueEx.call_count, 1)

    def test_complete_messages_sends_caller_messages_and_returns_text_and_usage(self):
        msgs = [{'role': 'system', 'content': 'be an agent'}, {'role': 'user', 'content': 'cancel #W1'}]
        opener = FakeOpener([completion(json.dumps(ANSWER))])
        provider = HTTPProvider(MODEL, 'https://generativelanguage.googleapis.com/v1beta/openai', 'k',
                                provider='gemini', retries=0)
        with mock.patch.object(request, 'build_opener', return_value=opener):
            text, usage, info = provider.complete_messages(msgs, json_mode=True)
        self.assertEqual(json.loads(text), ANSWER)
        self.assertEqual(usage['total_tokens'], 9)
        self.assertIsNone(info)                                  # clean stop + string content -> not lenient
        body = json.loads(opener.requests[0].data)
        self.assertEqual(body['messages'], msgs)                 # caller messages, not agentrx stage prompts
        self.assertIn('response_format', body)
        # json_mode=False (the user simulator) returns plain text and sends no response_format.
        plain = FakeOpener([completion('just text')])
        with mock.patch.object(request, 'build_opener', return_value=plain):
            text2, _, info2 = provider.complete_messages(msgs, json_mode=False)
        self.assertEqual((text2, info2), ('just text', None))
        self.assertNotIn('response_format', json.loads(plain.requests[0].data))
        # A null content (Gemini occasionally returns one) becomes '' and is flagged lenient with a reason.
        nulled = FakeOpener([completion(None)])
        with mock.patch.object(request, 'build_opener', return_value=nulled):
            text3, _, info3 = provider.complete_messages(msgs, json_mode=True)
        self.assertEqual(text3, '')
        self.assertTrue(info3['content_null'])

    def test_complete_messages_flags_lenient_on_max_tokens_and_malformed(self):
        provider = HTTPProvider(MODEL, 'https://generativelanguage.googleapis.com/v1beta/openai', 'k',
                                provider='gemini', retries=0)
        def resp(finish, content, out_tokens=7):
            return io.BytesIO(json.dumps({'id': 'g', 'model': MODEL,
                'usage': {'completion_tokens': out_tokens, 'total_tokens': out_tokens + 1},
                'choices': [{'finish_reason': finish, 'message': {'content': content}}]}).encode())
        # MAX_TOKENS (length): partial text is returned, flagged lenient, with the reason captured.
        mt = FakeOpener([resp('length', '{"action": "repl', out_tokens=4096)])
        with mock.patch.object(request, 'build_opener', return_value=mt):
            text, _, info = provider.complete_messages([{'role': 'user', 'content': 'x'}], json_mode=True)
        self.assertEqual(text, '{"action": "repl')
        self.assertEqual((info['finish_reason'], info['content_null'], info['output_tokens']),
                         ('length', False, 4096))
        self.assertEqual(info['content_chars'], len('{"action": "repl'))     # partial, not null
        # MALFORMED_FUNCTION_CALL: non-stop finish_reason must not raise; '' + lenient + null content.
        mf = FakeOpener([resp('function_call_filter: MALFORMED_FUNCTION_CALL', None)])
        with mock.patch.object(request, 'build_opener', return_value=mf):
            text2, _, info2 = provider.complete_messages([{'role': 'user', 'content': 'x'}], json_mode=True)
        self.assertEqual(text2, '')
        self.assertEqual((info2['finish_reason'], info2['content_null']),
                         ('function_call_filter: MALFORMED_FUNCTION_CALL', True))

    def test_generate_content_native_skips_thought_parts_and_reads_usage_metadata(self):
        provider = HTTPProvider('gemini-3.5-flash-lite', 'https://generativelanguage.googleapis.com/v1beta/openai',
                                'k', provider='gemini', retries=0, settings={'temperature': 0})
        def native(body):
            return io.BytesIO(json.dumps(body).encode())
        # A thought part plus a text part: only the text part is returned; thoughts count as output tokens.
        good = native({'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
            {'text': 'internal reasoning', 'thought': True}, {'text': '{"action": "reply"}'}]}}],
            'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 6, 'thoughtsTokenCount': 40,
                              'totalTokenCount': 146}})
        opener = FakeOpener([good])
        with mock.patch.object(request, 'build_opener', return_value=opener):
            text, usage, info = provider.generate_content([{'role': 'system', 'content': 'sys'},
                                                           {'role': 'user', 'content': 'hi'}], json_mode=True)
        self.assertEqual(text, '{"action": "reply"}')              # thought part skipped
        self.assertEqual((usage['prompt_tokens'], usage['completion_tokens']), (100, 46))  # 6 + 40 thoughts
        self.assertIsNone(info)                                    # STOP + text -> not lenient
        self.assertIn(':generateContent', opener.requests[0].full_url)
        self.assertEqual(opener.requests[0].get_header('X-goog-api-key'), 'k')
        # Thought-only response (no text part): empty text, lenient, with the thought token count recorded.
        empty = native({'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
            {'text': 'all thinking', 'thought': True}]}}],
            'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 0, 'thoughtsTokenCount': 512,
                              'totalTokenCount': 612}})
        with mock.patch.object(request, 'build_opener', return_value=FakeOpener([empty])):
            text2, usage2, info2 = provider.generate_content([{'role': 'user', 'content': 'hi'}], json_mode=True)
        self.assertEqual(text2, '')
        self.assertEqual((info2['thoughts_tokens'], info2['has_thought_part'], info2['output_tokens']),
                         (512, True, 0))
        self.assertEqual(usage2['completion_tokens'], 512)         # thoughts billed as output, not reported as zero

    def test_400_fallback_is_openai_compatible_only_and_once(self):
        for kind, outcomes in (('openai', [http_error(400)]), ('openai-compatible', [http_error(400)] * 2)):
            opener = FakeOpener(outcomes)
            provider = HTTPProvider(MODEL, BASE_URL, 'k', provider=kind, retries=2, retry_delay=0)
            with mock.patch.object(request, 'build_opener', return_value=opener), \
                    self.assertRaisesRegex(ProviderError, 'Model HTTP 400'):
                provider.complete('judge', {'checklist': False})
            self.assertEqual(len(opener.requests), len(outcomes), kind)
