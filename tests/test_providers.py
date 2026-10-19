import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch
from agentrx.providers import HTTPProvider, OllamaProvider, ProviderError


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
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ProviderError, 'OPENAI_API_KEY'):
                HTTPProvider.from_config({'provider': 'openai', 'model': 'gpt-5'})

    def test_gemini_provider_uses_openai_compatible_endpoint_and_key(self):
        p = HTTPProvider.from_config({'provider': 'gemini', 'api_key': 'gem-key', 'base_url': self.url, 'retries': 0})
        self.assertEqual(p.complete('global', {'policy': '', 'tools': {}}), {'constraints': []})
        self.assertEqual(Handler.records[0]['path'], '/v1/chat/completions')
        self.assertEqual(Handler.records[0]['auth'], 'Bearer gem-key')

    def test_gemini_missing_credentials_fail(self):
        with patch.dict('os.environ', {}, clear=True):
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
