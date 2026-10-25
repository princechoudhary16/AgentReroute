"""Actor decoding contract, the repair ladder, and the cache poisoning guard.

The central test here is ``test_contract_matches_the_workflow_consumer_exactly``. The decoding contract
in ``env.llm`` and the action validation in ``env.workflow`` are two implementations of one grammar, and
drift between them is expensive in both directions: stricter in the contract and CachedLLM refuses to
store replies the workflow runs happily, so every replay re-fetches and the reproducible clean prefix
that makes an injected run comparable to its clean pair is lost; looser and an unusable turn is cached
forever, which is how corpus-v1 rotted. That test fails if either side moves alone.
"""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib import request as urllib_request

from benchmark.env.llm import (ACTION_CONTRACT, ACTION_SCHEMA, GEMINI_ACTION_SCHEMA, TEXT_CONTRACT,
                               CachedLLM, Completion, GeminiActor, check_actor_reply, gemini_schema,
                               parse_json, request, with_note)
from benchmark.env.scripted import scripted_llm
from benchmark.env.tasks import load_fixture
from benchmark.env.workflow import Workflow, _validate_action


def native(text=None, finish='STOP', prompt=0, cand=0, thoughts=0, parts=None):
    """One native generateContent response envelope."""
    if parts is None:
        parts = [] if text is None else [{'text': text}]
    return io.BytesIO(json.dumps({
        'candidates': [{'finishReason': finish, 'content': {'parts': parts}}],
        'usageMetadata': {'promptTokenCount': prompt, 'candidatesTokenCount': cand,
                          'thoughtsTokenCount': thoughts, 'totalTokenCount': prompt + cand + thoughts}}).encode())


class _Opener:
    def __init__(self, items):
        self.items, self.requests = list(items), []

    def open(self, req, timeout=None):
        self.requests.append(req)
        return self.items.pop(0)


def actor(**kwargs):
    with mock.patch.dict('os.environ', {'GEMINI_API_KEY': 'gk'}):
        return GeminiActor('gemini-3.1-flash-lite', **kwargs)


def action_req(messages=None, **meta):
    return request('gemini-3.1-flash-lite', messages or [{'role': 'system', 'content': 'sys'},
                                                         {'role': 'user', 'content': 'go'}],
                   agent='planner', step_index=4, task_id='t1', **meta)


def run(a, req, replies):
    opener = _Opener(replies)
    with mock.patch.object(urllib_request, 'build_opener', return_value=opener):
        return a.complete(req), opener


# Every shape the actor can emit, and whether the DECODING contract may keep it. The expectation column
# is asserted against the workflow consumer too, never only against itself.
PAYLOADS = [
    '',
    '   ',
    'not json at all',
    '[1, 2]',
    '{}',
    '{"action": "nope"}',
    '{"action": "reply", "content": "hi"}',
    '{"action": "reply"}',                                     # consumer serialises a missing content
    '{"action": "reply", "content": 123}',                     # ... and a non-string one
    '{"action": "reply", "content": ""}',
    '{"action": "tool", "tool": "get_order_details", "arguments": {"order_id": "#W1"}}',
    '{"action": "tool", "tool": "x"}',                         # consumer defaults arguments to {}
    '{"action": "tool"}',                                      # no tool name: schema allows it, we do not
    '{"action": "tool", "tool": 123}',
    '{"action": "tool", "tool": "x", "arguments": "nope"}',
    '{"action": "handoff", "to": "order_agent", "message": "m", "context": {"order_id": "#W1"}}',
    '{"action": "handoff", "to": "order_agent"}',
    '{"action": "handoff"}',
    '{"action": "handoff", "to": "x", "context": "nope"}',      # consumer coerces a non-dict context
    '```json\n{"action": "reply", "content": "fenced"}\n```',
    '{"thought": "he said ("x") then", "action": "reply", "content": "y"}',   # measured: quotes unescaped
]


def consumer_accepts(text):
    """What ``workflow`` does with this text for the planner, which is allowed every action."""
    try:
        _validate_action('planner', parse_json(text))
        return True
    except ValueError:
        return False


class ContractTests(unittest.TestCase):
    def test_contract_matches_the_workflow_consumer_exactly(self):
        for text in PAYLOADS:
            with self.subTest(text=text[:60]):
                self.assertEqual(check_actor_reply(text, ACTION_CONTRACT)[0], consumer_accepts(text))

    def test_contract_rejects_the_two_measured_failure_classes(self):
        self.assertFalse(check_actor_reply('', ACTION_CONTRACT)[0])                      # MALFORMED, 0 tokens
        unescaped = '{"thought": "he said ("x") then", "action": "reply", "content": "y"}'
        self.assertFalse(check_actor_reply(unescaped, ACTION_CONTRACT)[0])               # STOP, not parseable

    def test_text_contract_requires_non_blank_and_no_contract_accepts_anything(self):
        self.assertTrue(check_actor_reply('yes please', TEXT_CONTRACT)[0])
        self.assertFalse(check_actor_reply('  \n ', TEXT_CONTRACT)[0])
        self.assertFalse(check_actor_reply(None, TEXT_CONTRACT)[0])
        for text in ('', 'anything', '{"verdict": 3}'):
            self.assertTrue(check_actor_reply(text, None)[0], text)

    def test_usable_action_is_canonicalised_and_rejected_text_is_returned_verbatim(self):
        usable, canonical = check_actor_reply('```json\n{"action": "reply", "content": "x"}\n```', ACTION_CONTRACT)
        self.assertTrue(usable)
        self.assertEqual(json.loads(canonical), {'action': 'reply', 'content': 'x'})
        self.assertFalse(canonical.startswith('```'))
        self.assertEqual(check_actor_reply('junk', ACTION_CONTRACT), (False, 'junk'))

    def test_with_note_does_not_mutate_the_callers_messages(self):
        messages = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'go'}]
        noted = with_note(messages, '!!')
        self.assertEqual(messages[-1]['content'], 'go')          # the recorded step input must not move
        self.assertEqual(noted[-1]['content'], 'go!!')
        self.assertIs(noted[0], messages[0])                     # untouched messages are shared, not copied


class LadderTests(unittest.TestCase):
    def test_first_draw_is_schema_constrained_and_keeps_the_json_mime_type(self):
        a = actor()
        comp, opener = run(a, action_req(contract=ACTION_CONTRACT),
                           [native('{"action": "reply", "content": "hi"}', cand=3, prompt=10)])
        self.assertFalse(comp.lenient)
        self.assertEqual(len(opener.requests), 1)                # a usable first draw never retries
        cfg = json.loads(opener.requests[0].data)['generationConfig']
        self.assertEqual(cfg['responseSchema'], GEMINI_ACTION_SCHEMA)
        self.assertEqual(cfg['responseMimeType'], 'application/json')   # Gemini needs both together
        self.assertEqual(a.stats()['repaired'], 0)

    def test_no_declared_contract_keeps_todays_single_draw_and_no_schema(self):
        a = actor()
        comp, opener = run(a, action_req(), [native('{"verdict": "anything"}', cand=2)])
        self.assertFalse(comp.lenient)
        self.assertEqual(len(opener.requests), 1)
        self.assertNotIn('responseSchema', json.loads(opener.requests[0].data)['generationConfig'])

    def test_malformed_first_draw_is_repaired_on_a_plain_draw(self):
        a = actor()
        messages = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'go'}]
        comp, opener = run(a, action_req(messages, contract=ACTION_CONTRACT), [
            native(finish='MALFORMED_RESPONSE', parts=[], prompt=10),
            native('{"action": "tool", "tool": "get_order_details", "arguments": {}}', prompt=11, cand=7)])
        self.assertFalse(comp.lenient)                           # repaired, so the turn is clean
        self.assertEqual(json.loads(comp.text)['tool'], 'get_order_details')
        self.assertEqual(a.stats()['repaired'], 1)
        self.assertEqual((comp.tokens_in, comp.tokens_out), (21, 7))   # every billed draw is counted
        first, second = (json.loads(r.data) for r in opener.requests)
        self.assertIn('responseSchema', first['generationConfig'])
        self.assertNotIn('responseMimeType', second.get('generationConfig', {}))
        self.assertEqual(first['contents'][-1]['parts'][0]['text'], 'go')
        self.assertTrue(second['contents'][-1]['parts'][0]['text'].startswith('go\n\nCRITICAL:'))
        self.assertEqual(messages[-1]['content'], 'go')           # caller's list still unmodified

    def test_clean_stop_that_is_not_a_usable_action_is_retried(self):
        """The class ``lenient`` cannot see: finishReason STOP, text present, not a usable action."""
        a = actor()
        comp, opener = run(a, action_req(contract=ACTION_CONTRACT), [
            native('{"a": "b" "c"}', cand=4),
            native('{"action": "reply", "content": "recovered"}', cand=4)])
        self.assertFalse(comp.lenient)
        self.assertEqual(len(opener.requests), 2)
        self.assertEqual(json.loads(comp.text)['content'], 'recovered')

    def test_all_draws_failing_marks_the_completion_lenient(self):
        a = actor()
        comp, opener = run(a, action_req(contract=ACTION_CONTRACT), [
            native(finish='MALFORMED_RESPONSE', parts=[], prompt=5),
            native('still not json', cand=2, prompt=5),
            native('', cand=0, prompt=5)])
        self.assertTrue(comp.lenient)
        self.assertEqual(len(opener.requests), 3)                # schema draw plus two plain draws
        self.assertEqual(a.stats()['unrecovered'], 1)
        self.assertEqual(comp.tokens_in, 15)

    def test_use_response_schema_false_falls_back_to_mime_type_only(self):
        a = actor(use_response_schema=False)
        _, opener = run(a, action_req(contract=ACTION_CONTRACT),
                        [native('{"action": "reply", "content": "hi"}', cand=1)])
        cfg = json.loads(opener.requests[0].data)['generationConfig']
        self.assertNotIn('responseSchema', cfg)
        self.assertEqual(cfg['responseMimeType'], 'application/json')

    def test_user_turn_retries_once_on_an_empty_reply(self):
        a = actor()
        req = request('gemini-3.1-flash-lite', [{'role': 'user', 'content': 'hi'}], json_mode=False,
                      agent='user', step_index=1, contract=TEXT_CONTRACT)
        comp, opener = run(a, req, [native('', cand=0), native('Yes, go ahead.', cand=4)])
        self.assertFalse(comp.lenient)
        self.assertEqual(comp.text, 'Yes, go ahead.')
        self.assertEqual(len(opener.requests), 2)

    def test_lenient_log_records_every_rejected_draw_and_the_repair(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'lenient.jsonl'
            a = actor(lenient_log=str(path))
            run(a, action_req(contract=ACTION_CONTRACT), [
                native(finish='MALFORMED_RESPONSE', parts=[]),
                native('{"action": "reply", "content": "ok"}', cand=2)])
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['outcome'] for r in rows], ['rejected', 'repaired'])
            self.assertEqual(rows[0]['reason'], 'lenient')
            self.assertEqual(rows[0]['agent'], 'planner')
            self.assertTrue((Path(d) / 'lenient_raw.jsonl').exists())

    def test_the_two_provider_schema_dialects_describe_one_grammar(self):
        """Ollama takes standard JSON Schema; Gemini wants the same thing with upper-case types.

        The measured Gemini config in benchmark/runs/threeway.log is reproduced verbatim here, so the
        derivation cannot silently drift away from the configuration that actually scored 14/14.
        """
        measured = {
            'type': 'OBJECT',
            'properties': {'thought': {'type': 'STRING'},
                           'action': {'type': 'STRING', 'enum': ['tool', 'handoff', 'reply']},
                           'tool': {'type': 'STRING'}, 'arguments': {'type': 'OBJECT'},
                           'to': {'type': 'STRING'}, 'message': {'type': 'STRING'},
                           'context': {'type': 'OBJECT'}, 'content': {'type': 'STRING'}},
            'required': ['action'],
        }
        self.assertEqual(GEMINI_ACTION_SCHEMA, measured)
        self.assertEqual(gemini_schema(ACTION_SCHEMA), measured)
        self.assertEqual(ACTION_SCHEMA['type'], 'object')          # Ollama rejects 'OBJECT'
        self.assertEqual(sorted(ACTION_SCHEMA['properties']), sorted(measured['properties']))
        for name, spec in ACTION_SCHEMA['properties'].items():
            self.assertEqual(spec['type'], measured['properties'][name]['type'].lower(), name)

    def test_response_schema_without_json_mode_is_a_caller_error(self):
        provider = actor().provider
        with self.assertRaisesRegex(Exception, 'responseSchema requires json_mode'):
            provider.generate_content([{'role': 'user', 'content': 'x'}], json_mode=False,
                                      response_schema=ACTION_SCHEMA)


class _Fake:
    measurement_kind = 'model'

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def complete(self, req):
        self.calls += 1
        return self.replies.pop(0)


def completion(text, lenient=False):
    return Completion(text, 1, 1, 'm', lenient=lenient)


class CacheGuardTests(unittest.TestCase):
    def test_an_unusable_action_reply_is_never_written(self):
        with tempfile.TemporaryDirectory() as d:
            inner = _Fake([completion('{"action": "tool"}'), completion('{"action": "reply", "content": "ok"}')])
            cache = CachedLLM(inner, d)
            req = action_req(contract=ACTION_CONTRACT)
            self.assertEqual(cache.complete(req).text, '{"action": "tool"}')   # returned, just not stored
            self.assertEqual(cache.stats()['refused_writes'], 1)
            self.assertEqual(cache.complete(req).text, '{"action": "reply", "content": "ok"}')
            self.assertEqual(inner.calls, 2)                                   # no hit on the poisoned turn
            self.assertEqual(cache.stats()['hits'], 0)
            self.assertEqual(cache.complete(req).cached, True)                 # the usable one did persist

    def test_a_poisoned_entry_written_before_the_guard_is_refetched_and_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            inner = _Fake([completion('{"action": "tool"}'), completion('{"action": "reply", "content": "ok"}')])
            cache = CachedLLM(inner, d)
            no_contract = action_req()                      # same cache key: meta is not part of it
            cache.complete(no_contract)                     # stored, because nothing declared a contract
            self.assertEqual(cache.stats()['refused_writes'], 0)
            fixed = cache.complete(action_req(contract=ACTION_CONTRACT))
            self.assertEqual(cache.stats()['stale_poison'], 1)
            self.assertEqual(json.loads(fixed.text)['content'], 'ok')
            self.assertTrue(cache.complete(action_req(contract=ACTION_CONTRACT)).cached)   # healed on disk

    def test_read_only_serves_a_poisoned_entry_but_counts_it(self):
        with tempfile.TemporaryDirectory() as d:
            inner = _Fake([completion('{"action": "tool"}')])
            CachedLLM(inner, d).complete(action_req())
            frozen = CachedLLM(_Fake([]), d, read_only=True)
            served = frozen.complete(action_req(contract=ACTION_CONTRACT))
            self.assertEqual(served.text, '{"action": "tool"}')
            self.assertEqual(frozen.stats()['poisoned_hits'], 1)

    def test_a_caller_that_declares_no_contract_is_cached_normally(self):
        """Regression guard: the LLM judge baselines send json_mode=True with a verdict, not an action.

        Gating the guard on ``req['json']`` would have refused every one of those replies, re-called the
        model on each replay, and hard-failed a read-only run.
        """
        with tempfile.TemporaryDirectory() as d:
            inner = _Fake([completion('{"step_index": 7, "agent": "planner"}')])
            cache = CachedLLM(inner, d)
            judge = request('m', [{'role': 'user', 'content': 'judge this'}], agent='judge')
            self.assertFalse(cache.complete(judge).cached)
            self.assertTrue(cache.complete(judge).cached)
            self.assertEqual((inner.calls, cache.stats()['refused_writes']), (1, 0))

    def test_a_lenient_reply_is_never_written_whatever_the_contract(self):
        with tempfile.TemporaryDirectory() as d:
            inner = _Fake([completion('{"verdict": 1}', lenient=True), completion('{"verdict": 1}')])
            cache = CachedLLM(inner, d)
            judge = request('m', [{'role': 'user', 'content': 'x'}], agent='judge')
            cache.complete(judge)
            self.assertEqual(cache.stats()['refused_writes'], 1)
            cache.complete(judge)
            self.assertEqual(inner.calls, 2)


class _Recorder:
    measurement_kind = 'offline_fixture'

    def __init__(self, inner):
        self.inner, self.requests = inner, []

    def complete(self, req):
        self.requests.append(req)
        return self.inner.complete(req)


class WorkflowDeclarationTests(unittest.TestCase):
    def test_every_workflow_completion_declares_its_contract(self):
        """A new actor call site that forgets ``contract`` silently loses the guard, so pin it here."""
        domain = load_fixture()
        task = domain.tasks[0]
        llm = _Recorder(scripted_llm(domain, task))
        Workflow(domain, task, llm, 'scripted').run()
        self.assertTrue(llm.requests)
        kinds = set()
        for req in llm.requests:
            meta = req['meta']
            expected = TEXT_CONTRACT if meta.get('agent') == 'user' else ACTION_CONTRACT
            self.assertEqual(meta.get('contract'), expected, meta)
            kinds.add(expected)
        self.assertEqual(kinds, {TEXT_CONTRACT, ACTION_CONTRACT})   # both paths actually exercised


if __name__ == '__main__':
    unittest.main()


class _FakeOllama:
    """Stands in for urlopen: serves queued /api/chat bodies and records what was sent."""

    def __init__(self, bodies):
        self.bodies, self.payloads = list(bodies), []

    def __call__(self, req, timeout=None):
        self.payloads.append(json.loads(req.data))
        return io.BytesIO(json.dumps(self.bodies.pop(0)).encode())


def ollama_body(text, prompt=10, output=5):
    return {'message': {'content': text}, 'prompt_eval_count': prompt, 'eval_count': output}


class OllamaActorTests(unittest.TestCase):
    """The native-endpoint actor: states its context window and refuses to be truncated into."""

    def client(self, bodies, **kwargs):
        from benchmark.env.llm import OllamaClient
        opener = _FakeOllama(bodies)
        return OllamaClient('http://gpu-box:11434', opener=opener, **kwargs), opener

    def test_states_the_window_and_forbids_truncation(self):
        client, opener = self.client([ollama_body('{"action": "reply", "content": "hi"}')], num_ctx=16384)
        client.complete(action_req(contract=ACTION_CONTRACT))
        sent = opener.payloads[0]
        self.assertEqual(sent['options']['num_ctx'], 16384)
        self.assertEqual((sent['truncate'], sent['shift']), (False, False))
        self.assertEqual(sent['format'], ACTION_SCHEMA)        # schema, not bare json mode

    def test_context_overflow_raises_instead_of_returning_a_truncated_turn(self):
        """Ollama truncates silently by default, and a truncated prompt reads as an agent error."""
        from benchmark.env.llm import LLMError
        client, _ = self.client([ollama_body('{"action": "reply", "content": "x"}', prompt=8100, output=100)],
                                num_ctx=8192)
        with self.assertRaisesRegex(LLMError, 'context window'):
            client.complete(action_req(contract=ACTION_CONTRACT))

    def test_a_contract_failure_gets_one_perturbed_draw_then_is_marked_lenient(self):
        client, opener = self.client([ollama_body('not json at all'), ollama_body('still not json')])
        completion = client.complete(action_req(contract=ACTION_CONTRACT))
        self.assertTrue(completion.lenient)                    # CachedLLM will refuse to store it
        self.assertEqual(len(opener.payloads), 2)
        self.assertEqual(opener.payloads[0]['format'], ACTION_SCHEMA)
        self.assertEqual(opener.payloads[1]['format'], 'json')  # schema dropped, asked in words instead
        self.assertTrue(opener.payloads[1]['messages'][-1]['content'].endswith('action keys.')
                        or 'CRITICAL' in opener.payloads[1]['messages'][-1]['content'])
        self.assertEqual((completion.tokens_in, completion.tokens_out), (20, 10))   # both draws billed

    def test_a_repaired_draw_is_clean_and_counted(self):
        client, opener = self.client([ollama_body('garbage'),
                                      ollama_body('{"action": "reply", "content": "recovered"}')])
        completion = client.complete(action_req(contract=ACTION_CONTRACT))
        self.assertFalse(completion.lenient)
        self.assertEqual(json.loads(completion.text)['content'], 'recovered')
        self.assertEqual(client.stats()['repaired'], 1)

    def test_a_caller_without_a_contract_gets_one_draw_in_plain_json_mode(self):
        """The judge baselines go through this client and must not acquire an action schema."""
        client, opener = self.client([ollama_body('{"step_index": 7}')])
        judge = request('qwen2.5:32b', [{'role': 'user', 'content': 'judge'}], agent='judge')
        completion = client.complete(judge)
        self.assertFalse(completion.lenient)
        self.assertEqual(len(opener.payloads), 1)
        self.assertEqual(opener.payloads[0]['format'], 'json')
