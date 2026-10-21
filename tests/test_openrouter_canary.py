"""OpenRouter hardening of HTTPProvider, plus ONE live canary call (skipped without OPENROUTER_API_KEY).

The live test makes a single chat completion. Only if OpenRouter answers 404/400 for the slug does it make one
extra GET /models to report the active MiniMax slug; it never retries the completion with that slug
(set OPENROUTER_CANARY_MODEL and rerun instead).
"""
import io
import json
import os
import unittest
from email.message import Message
from unittest import mock
from urllib import error, request

from agentrx.predictions import validate_prediction
from agentrx.providers import HTTPProvider, ProviderError, _parse_json_object, env_key
from agentrx.taxonomy import CATEGORIES

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

    def test_400_fallback_is_openai_compatible_only_and_once(self):
        for kind, outcomes in (('openai', [http_error(400)]), ('openai-compatible', [http_error(400)] * 2)):
            opener = FakeOpener(outcomes)
            provider = HTTPProvider(MODEL, BASE_URL, 'k', provider=kind, retries=2, retry_delay=0)
            with mock.patch.object(request, 'build_opener', return_value=opener), \
                    self.assertRaisesRegex(ProviderError, 'Model HTTP 400'):
                provider.complete('judge', {'checklist': False})
            self.assertEqual(len(opener.requests), len(outcomes), kind)


def canary_trajectory():
    return {'trajectory_id': 'openrouter-canary', 'domain': 'retail',
            'instruction': 'Please cancel my pending order #W1000001.',
            'steps': [{'index': 1, 'substeps': [{'sub_index': 1, 'role': 'user',
                                                 'content': 'Please cancel my pending order #W1000001.'}]},
                      {'index': 2, 'substeps': [{'sub_index': 1, 'role': 'planner', 'content': '',
                                                 'tool_calls': [{'id': 'call-2', 'name': 'cancel_pending_order',
                                                                 'arguments': {'order_id': ''}}]}]}]}


def resolve_minimax_slug():
    """One GET /models: the first active minimax/*:free slug, else the first minimax/* slug."""
    req = request.Request(BASE_URL + '/models', headers={'User-Agent': 'AgentReroute/1.0'})
    with request.urlopen(req, timeout=30) as response:
        ids = [m['id'] for m in json.loads(response.read()).get('data', [])]
    minimax = [i for i in ids if i.startswith('minimax/')]
    return next((i for i in minimax if i.endswith(':free')), minimax[0] if minimax else None)


class LiveOpenRouterCanary(unittest.TestCase):
    def test_live_openrouter_single_call_canary(self):
        key = env_key('OPENROUTER_API_KEY')
        if not key:
            self.skipTest('OPENROUTER_API_KEY not set')
        model = os.environ.get('OPENROUTER_CANARY_MODEL', MODEL)
        provider = HTTPProvider(model, BASE_URL, key, provider='openai-compatible', settings={'temperature': 0},
                                timeout=120, retries=3, retry_delay=2)
        trajectory = canary_trajectory()
        payload = {'trajectory': trajectory, 'checklist': False,
                   'policy': 'Before cancelling, the order id must be a non-empty id from the user.',
                   'tools': {'cancel_pending_order': {
                       'name': 'cancel_pending_order', 'description': 'Cancel a pending order.',
                       'parameters': {'type': 'object', 'required': ['order_id', 'reason'], 'properties': {
                           'order_id': {'type': 'string', 'description': "The order id, such as '#W0000000'."},
                           'reason': {'type': 'string', 'enum': ['no longer needed', 'ordered by mistake']}}}}}}
        try:
            answer = provider.complete('judge', payload)        # the single live call
        except ProviderError as exc:
            if 'HTTP 404' in str(exc) or 'HTTP 400' in str(exc):
                self.fail(f'OpenRouter rejected slug {model!r} ({exc}); active slug from /models: '
                          f'{resolve_minimax_slug()!r}. Rerun with OPENROUTER_CANARY_MODEL=<slug>.')
            raise
        t = provider.last_telemetry
        pred = validate_prediction(answer, trajectory)
        print('\n--- OpenRouter canary ---------------------------------------------')
        print(f"model requested : {model}\nmodel served    : {t.get('model')}\nresponse_id     : {t.get('response_id')}")
        print(f"http_status     : {t.get('http_status')}   attempts: {t['attempts']}   "
              f"response_format kept: {t.get('response_format')}")
        usage = t.get('usage') or {}
        print(f"tokens          : prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} "
              f"total={usage.get('total_tokens')}\nlatency         : {t['seconds']:.2f}s")
        print('prediction      : ' + json.dumps(pred, ensure_ascii=False, indent=1))
        print('--------------------------------------------------------------------')
        self.assertIsInstance(answer, dict)
        self.assertIn(pred['step_index'], (1, 2))
        self.assertIn(pred['category'], CATEGORIES)
        self.assertGreaterEqual(t['attempts'], 1)
        self.assertGreater(t['seconds'], 0)


if __name__ == '__main__':
    unittest.main()
