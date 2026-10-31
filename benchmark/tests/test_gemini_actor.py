"""Gemini actor bridge: benchmark complete(request) -> agentrx HTTPProvider.generate_content (native, fake opener)."""
import io
import json
import unittest
from unittest import mock
from urllib import request

from benchmark.env.llm import GeminiActor


def native(text, prompt=0, cand=0, thoughts=0, finish='STOP'):
    return io.BytesIO(json.dumps({
        'candidates': [{'finishReason': finish, 'content': {'parts': [{'text': text}]}}],
        'usageMetadata': {'promptTokenCount': prompt, 'candidatesTokenCount': cand,
                          'thoughtsTokenCount': thoughts, 'totalTokenCount': prompt + cand + thoughts}}).encode())


class _Opener:
    def __init__(self, items):
        self.items, self.requests = list(items), []

    def open(self, req, timeout=None):
        self.requests.append(req)
        return self.items.pop(0)


class GeminiActorTests(unittest.TestCase):
    def test_bridges_benchmark_request_to_the_native_endpoint(self):
        with mock.patch.dict('os.environ', {'GEMINI_API_KEY': 'gk'}):
            actor = GeminiActor('gemini-3.5-flash-lite')
        self.assertEqual(actor.measurement_kind, 'model')
        self.assertEqual(actor.model_digest('gemini-3.5-flash-lite'), 'gemini:gemini-3.5-flash-lite')
        opener = _Opener([native('{"action": "reply", "content": "hi"}', prompt=12, cand=5, thoughts=4)])
        req = {'model': 'gemini-3.5-flash-lite', 'messages': [{'role': 'system', 'content': 'be an agent'},
               {'role': 'user', 'content': 'x'}], 'temperature': 0.0, 'seed': 0, 'json': True,
               'meta': {'agent': 'planner'}}
        with mock.patch.object(request, 'build_opener', return_value=opener):
            comp = actor.complete(req)
        self.assertEqual(json.loads(comp.text)['content'], 'hi')
        self.assertEqual(comp.tokens_in, 12)
        self.assertEqual(comp.tokens_out, 9)                 # candidates(5) + thoughts(4), thoughts billed as output
        self.assertFalse(comp.lenient)
        sent = opener.requests[0]
        self.assertIn(':generateContent', sent.full_url)      # native endpoint, not /chat/completions
        body = json.loads(sent.data)
        self.assertEqual(body['contents'], [{'role': 'user', 'parts': [{'text': 'x'}]}])
        self.assertEqual(body['systemInstruction'], {'parts': [{'text': 'be an agent'}]})
        self.assertEqual(body['generationConfig']['responseMimeType'], 'application/json')

    def test_api_key_comes_from_env_not_config(self):
        with mock.patch.dict('os.environ', {}, clear=True), mock.patch('agentrx.providers.sys.platform', 'linux'):
            with self.assertRaisesRegex(Exception, 'GEMINI_API_KEY'):
                GeminiActor('gemini-3.5-flash-lite')


if __name__ == '__main__':
    unittest.main()
