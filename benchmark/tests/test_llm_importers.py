import io
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.env.llm import CachedLLM, LLMError, OllamaClient, ScriptedLLM, parse_json, request
from benchmark.env.tasks import fetch_tau_retail
from benchmark.importers import import_agentrx, import_whowhen
from benchmark.io import read_json, write_json
from benchmark.corpus import load_gold, load_public


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class OllamaClientTest(unittest.TestCase):
    def test_chat_payload_and_parsing(self):
        sent = []

        def opener(req, timeout):
            body = json.loads(req.data.decode()) if req.data else None
            sent.append((req.full_url, body))
            if req.full_url.endswith('/api/show'):
                return FakeResponse(json.dumps({'digest': 'abc123'}).encode())
            return FakeResponse(json.dumps({'message': {'content': '{"action": "reply", "content": "hi"}'},
                                            'prompt_eval_count': 11, 'eval_count': 5}).encode())

        client = OllamaClient('http://localhost:11434', opener=opener)
        out = client.complete(request('qwen2.5:7b', [{'role': 'user', 'content': 'x'}], 0.0, 7, agent='planner'))
        url, body = sent[-1]
        self.assertTrue(url.endswith('/api/chat'))
        self.assertEqual(body['format'], 'json')
        self.assertEqual(body['options']['seed'], 7)
        self.assertNotIn('meta', body)
        self.assertEqual((out.tokens_in, out.tokens_out), (11, 5))
        self.assertEqual(client.model_digest('qwen2.5:7b'), 'abc123')

    def test_retries_timeouts_but_not_client_errors(self):
        import urllib.error
        calls, waits = [], []

        def flaky(req, timeout):
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError('timed out')
            return FakeResponse(json.dumps({'message': {'content': '{}'}}).encode())

        client = OllamaClient(opener=flaky, retries=4, backoff=1, sleep=waits.append)
        client.complete(request('m', [{'role': 'user', 'content': 'x'}]))
        self.assertEqual((len(calls), waits), (3, [1, 3]))

        def bad(req, timeout):
            calls.append(1)
            raise urllib.error.HTTPError(req.full_url, 400, 'bad request', {}, None)

        calls.clear()
        with self.assertRaises(LLMError):
            OllamaClient(opener=bad, sleep=waits.append).complete(request('m', [{'role': 'user', 'content': 'x'}]))
        self.assertEqual(len(calls), 1)

    def test_cache_returns_identical_completions(self):
        calls = []
        inner = ScriptedLLM(lambda req: calls.append(1) or {'n': len(calls)})
        with tempfile.TemporaryDirectory() as tmp:
            llm = CachedLLM(inner, tmp)
            req = request('m', [{'role': 'user', 'content': 'same'}], seed=1)
            first, second = llm.complete(req), llm.complete(req)
            self.assertEqual(first.text, second.text)
            self.assertTrue(second.cached)
            self.assertEqual((llm.hits, llm.misses), (1, 1))
            llm.complete(request('m', [{'role': 'user', 'content': 'same'}], seed=2))
            self.assertEqual(llm.misses, 2)

    def test_parse_json_tolerates_fences_and_prose(self):
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {'a': 1})
        self.assertEqual(parse_json('Sure! {"a": 2} done'), {'a': 2})
        with self.assertRaises(ValueError):
            parse_json('[1, 2]')

    def test_fetch_tau_records_commit_and_hashes(self):
        tasks = 'TASKS = [Task(user_id="u", instruction="i", actions=[], outputs=[])]'

        def opener(url, timeout):
            if 'api.github.com' in url:
                return FakeResponse(json.dumps({'sha': 'deadbeef'}).encode())
            self.assertIn('/deadbeef/', url)
            return FakeResponse(tasks.encode() if url.endswith('.py') else b'{}')

        with tempfile.TemporaryDirectory() as tmp:
            manifest = fetch_tau_retail(tmp, opener=opener)
            self.assertEqual(manifest['commit'], 'deadbeef')
            self.assertEqual(manifest['n_tasks'], 1)
            self.assertEqual(len(manifest['files']), 5)


class ImporterTest(unittest.TestCase):
    def test_agentrx_and_whowhen(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            case = {'trajectory': {'trajectory_id': '7', 'domain': 'tau', 'instruction': 'q', 'steps': [
                {'index': 1, 'substeps': [{'sub_index': 1, 'role': 'user', 'content': 'hi'}]},
                {'index': 2, 'substeps': [{'sub_index': 1, 'role': 'assistant', 'content': 'there are 3'}]}]},
                'gold': {'trajectory_id': '7', 'critical': {'failure_id': '1', 'step_index': 2,
                                                            'category': 'MisinterpretationOfToolOutput',
                                                            'agent': 'assistant'},
                         'failures': [{'failure_id': '1', 'step_index': 2, 'category': 'MisinterpretationOfToolOutput',
                                       'agent': 'assistant'}]}}
            write_json(tmp / 'cases.json', [case])
            self.assertEqual(import_agentrx(tmp / 'cases.json', tmp / 'corpus'), 1)
            gold = load_gold(tmp / 'corpus', 'external_agentrx')
            g = next(iter(gold.values()))
            self.assertEqual(g['critical']['mode'], 'misinterpretation_of_tool_output')
            self.assertEqual(g['label_mapping']['fit'], 'approximate')
            (tmp / 'ww' / 'Hand-Crafted').mkdir(parents=True)
            write_json(tmp / 'ww' / 'Hand-Crafted' / '1.json', {
                'history': [{'role': 'user', 'content': 'task'}, {'name': 'WebSurfer', 'content': 'wrong page'}],
                'mistake_agent': 'WebSurfer', 'mistake_step': '1', 'question': 'task'})
            self.assertEqual(import_whowhen(tmp / 'ww', tmp / 'corpus'), 1)
            ww = next(iter(load_gold(tmp / 'corpus', 'external_whowhen').values()))
            self.assertIsNone(ww['critical']['mode'])
            self.assertEqual(len(load_public(tmp / 'corpus', 'external_whowhen')), 1)
            self.assertEqual(set(read_json(tmp / 'corpus' / 'manifest.json')['external']),
                             {'external_agentrx', 'external_whowhen'})


if __name__ == '__main__':
    unittest.main()
