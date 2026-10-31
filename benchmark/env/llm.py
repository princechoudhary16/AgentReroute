"""LLM clients: local Ollama, a deterministic response cache, and scripted test doubles.

Every completion goes through ``LLMClient.complete(request)``. A request is a
plain dict: ``{model, messages, temperature, seed, json, meta}``. ``meta`` is
never sent to the model and never part of the cache key; it only helps
scripted fixtures (and logs) know which agent is speaking.
"""
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..io import digest


def _truncate_strings(obj, limit=4000):
    """Recursively truncate long string values (keeping all keys) so a raw body log stays bounded."""
    if isinstance(obj, str):
        return obj if len(obj) <= limit else obj[:limit] + f'...<+{len(obj) - limit} chars>'
    if isinstance(obj, dict):
        return {k: _truncate_strings(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_strings(v, limit) for v in obj]
    return obj

DEFAULT_MODEL = 'qwen2.5:7b'
DEFAULT_HOST = 'http://localhost:11434'
# Ollama's own default is 4096, which silently truncates a trace of any size. 16384 matches
# agentrx.providers.OllamaProvider; size it against the longest prompt a run will actually send.
DEFAULT_NUM_CTX = 16384


@dataclass
class Completion:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ''
    cached: bool = False
    lenient: bool = False        # the actor's lenient path fired (odd finish_reason, refusal, null content)

    @property
    def tokens(self):
        return self.tokens_in + self.tokens_out


class LLMError(RuntimeError):
    pass


def request(model, messages, temperature=0.0, seed=0, json_mode=True, **meta):
    return {'model': model, 'messages': messages, 'temperature': float(temperature), 'seed': int(seed),
            'json': bool(json_mode), 'meta': meta}


def parse_json(text):
    """Parse a model reply as a JSON object, tolerating code fences and surrounding prose."""
    text = (text or '').strip()
    if text.startswith('```'):
        text = text.strip('`')
        text = text[text.find('\n') + 1:] if '\n' in text else text
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find('{'), text.rfind('}')
        if start < 0 or end <= start:
            raise ValueError('reply is not a JSON object') from None
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError('reply is not a JSON object')
    return value


# ------------------------------------------------------------------ actor decoding contract
ACTION_CONTRACT = 'action'    # an agent turn: one JSON action object
TEXT_CONTRACT = 'text'        # a simulated user turn: non-empty plain text

# Decoding grammar for an agent turn, as standard JSON Schema. Reproduced from the shape measured in
# benchmark/runs/threeway.log config C, which drew 14/14 usable action turns against 19/23 for the JSON
# mime type alone. Every key is optional except ``action``, exactly as measured, so a schema-constrained
# draw can still omit a tool name; check_actor_reply is what rejects that. The schema deliberately says
# nothing about WHICH tools or handoff targets are allowed: an unknown tool, or a specialist attempting
# a handoff, is a fault this benchmark exists to record, not a decoding failure to retry away.
#
# This is the Ollama dialect, which its ``format`` field takes directly. Gemini's native responseSchema
# wants the same grammar with upper-case type names, so GEMINI_ACTION_SCHEMA is derived below rather
# than written twice.
ACTION_SCHEMA = {
    'type': 'object',
    'properties': {
        'thought': {'type': 'string'},
        'action': {'type': 'string', 'enum': ['tool', 'handoff', 'reply']},
        'tool': {'type': 'string'},
        'arguments': {'type': 'object'},
        'to': {'type': 'string'},
        'message': {'type': 'string'},
        'context': {'type': 'object'},
        'content': {'type': 'string'},
    },
    'required': ['action'],
}


def gemini_schema(schema):
    """The same grammar in Gemini's responseSchema dialect, whose ``type`` values are upper case.

    Gemini's native generateContent takes an OpenAPI-flavoured schema (OBJECT, STRING) while Ollama's
    ``format`` takes standard JSON Schema and rejects the Gemini spelling outright:
    ``JSON schema error at #: unrecognized type OBJECT``. Sharing one constant between the two providers
    meant whichever dialect it was written in, the other backend's actor could never emit a schema-
    constrained draw. Deriving one from the other keeps a single source of truth for the grammar.
    """
    if not isinstance(schema, dict):
        return schema
    converted = {}
    for key, value in schema.items():
        if key == 'type' and isinstance(value, str):
            converted[key] = value.upper()
        elif key == 'properties' and isinstance(value, dict):
            converted[key] = {name: gemini_schema(item) for name, item in value.items()}
        elif key == 'items':
            converted[key] = gemini_schema(value)
        else:
            converted[key] = value
    return converted


GEMINI_ACTION_SCHEMA = gemini_schema(ACTION_SCHEMA)

JSON_RETRY_NOTE = ('\n\nCRITICAL: reply with a single valid JSON object only. No markdown fences, no prose '
                   'outside the JSON, and escape every quote that appears inside a string value.')
TEXT_RETRY_NOTE = '\n\nReply with your next message as the user, in plain text.'


def check_actor_reply(text, contract):
    """``(usable, canonical_text)`` for one actor draw under ``contract``.

    This is the DECODING contract and nothing more. It accepts exactly what
    ``workflow._validate_action`` accepts, minus that function's one policy clause (whether the acting
    agent may hand off at all). Keeping the two in step matters in both directions:

    * stricter here than the consumer, and CachedLLM refuses to store replies the workflow happily runs,
      so every replay of that prefix re-fetches - paid for twice, and the reproducible clean prefix that
      makes an injected run comparable to its clean pair is gone;
    * looser here, and a reply that decodes into nothing usable is cached forever.

    ``benchmark/tests/test_actor_contract.py`` pins that equality, so tightening either side alone fails.
    """
    if contract == TEXT_CONTRACT:
        return bool((text or '').strip()), text
    if contract != ACTION_CONTRACT:
        return True, text
    try:
        action = parse_json(text)
    except ValueError:
        return False, text
    kind = action.get('action')
    if kind == 'tool':
        usable = isinstance(action.get('tool'), str) and isinstance(action.get('arguments', {}), dict)
    elif kind == 'handoff':
        usable = isinstance(action.get('to'), str)
    elif kind == 'reply':
        usable = True        # the consumer serialises a non-string content rather than rejecting it
    else:
        usable = False
    return usable, (json.dumps(action, ensure_ascii=False) if usable else text)


def with_note(messages, note):
    """``messages`` with ``note`` appended to a shallow copy of the last message.

    Copied, never mutated. The caller keeps the original list as the step's recorded input, and that
    step's input_digest is what a fault operator matches its injection site on (faults.operators). A
    repaired draw must not move the digest, or the same fault would land on a different site in a replay.
    """
    out = list(messages)
    last = dict(out[-1])
    last['content'] = str(last.get('content', '')) + note
    out[-1] = last
    return out


class OllamaClient:
    """Chat completions from a local Ollama server (``/api/chat``)."""
    measurement_kind = 'model'

    def __init__(self, host=DEFAULT_HOST, timeout=300, num_ctx=DEFAULT_NUM_CTX,
                 opener=urllib.request.urlopen, retries=4, backoff=10.0, sleep=time.sleep,
                 use_response_schema=True):
        # The OpenAI-compatible /v1 endpoint cannot set num_ctx and lets Ollama truncate anything past
        # its 4096 default, so this speaks the native /api/chat endpoint and states the window.
        self.host, self.timeout, self.num_ctx, self.opener = host.rstrip('/'), timeout, num_ctx, opener
        self.retries, self.backoff, self.sleep = retries, backoff, sleep
        self.use_response_schema = bool(use_response_schema)
        self.draws = self.repaired = self.unrecovered = 0
        self._digests = {}

    def _post(self, path, payload):
        """POST with retries: timeouts, connection errors, 429 and 5xx back off and retry; other 4xx fail."""
        body = json.dumps(payload).encode('utf-8')
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.host + path, data=body, headers={'Content-Type': 'application/json'})
            try:
                with self.opener(req, timeout=self.timeout) as response:
                    return json.loads(response.read().decode('utf-8'))
            except urllib.error.HTTPError as exc:
                # Read the body BEFORE closing: Ollama puts the actual reason there, and a bare
                # "HTTP Error 400: Bad Request" is undiagnosable.
                detail = ''
                try:
                    detail = exc.read(4096).decode('utf-8', errors='replace')
                except (OSError, ValueError):
                    pass
                code = exc.code
                exc.close()
                if any(term in detail.lower() for term in
                       ('exceed_context_size', 'context size', 'context length', 'context window')):
                    raise LLMError(f'Ollama context limit exceeded (HTTP {code}); raise num_ctx. '
                                   f'Nothing was accepted: {detail[:300]}') from None
                if code != 429 and code < 500:
                    raise LLMError(f'Ollama request to {self.host}{path} failed: HTTP {code}: '
                                   f'{detail[:600] or "(empty body)"}') from None
                error = exc
            except (urllib.error.URLError, OSError) as exc:
                error = exc
            if attempt < self.retries:
                self.sleep(self.backoff * 3 ** attempt)
        raise LLMError(f'Ollama request to {self.host}{path} failed after {self.retries + 1} attempts: '
                       f'{error}') from error

    def model_digest(self, model):
        if model not in self._digests:
            try:
                info = self._post('/api/show', {'model': model})
                self._digests[model] = info.get('digest') or digest(info.get('modelfile', model))
            except LLMError:
                self._digests[model] = 'unknown'
        return self._digests[model]

    def available_models(self):
        req = urllib.request.Request(self.host + '/api/tags')
        try:
            with self.opener(req, timeout=10) as response:
                return [m['name'] for m in json.loads(response.read().decode('utf-8')).get('models', [])]
        except (urllib.error.URLError, OSError) as exc:
            raise LLMError(f'Ollama is not reachable at {self.host}: {exc}') from exc

    def stats(self):
        return {'draws': self.draws, 'repaired': self.repaired, 'unrecovered': self.unrecovered,
                'num_ctx': self.num_ctx, 'response_schema': self.use_response_schema}

    def complete(self, req):
        """One agent or user turn, with the context window stated and overflow refused.

        Two things this guards that the previous version did not:

        * ``truncate: False, shift: False`` plus an explicit token check. Ollama silently truncates a
          prompt longer than num_ctx, and a truncated prompt yields a degraded action indistinguishable
          from a real agent error, which is exactly how corpus-v1 rotted. Overflow raises instead.
        * the declared action contract. A schema-constrained draw is tried first (measured 0 unusable of
          14 against 4 of 23 for JSON-mode alone), and because this server is deterministic an identical
          retry would return identical bytes, so the one fallback draw drops the schema and asks in
          words. If both fail the completion is marked lenient, which stops CachedLLM storing it.
        """
        contract = (req.get('meta') or {}).get('contract')
        schema = ACTION_SCHEMA if (self.use_response_schema and req['json']
                                   and contract == ACTION_CONTRACT) else None
        draws = [(schema, None)]
        if contract is not None:
            draws.append((None, JSON_RETRY_NOTE if contract == ACTION_CONTRACT else TEXT_RETRY_NOTE))
        text, tokens_in, tokens_out = '', 0, 0
        for attempt, (draw_schema, note) in enumerate(draws):
            messages = req['messages'] if note is None else with_note(req['messages'], note)
            payload = {'model': req['model'], 'messages': messages, 'stream': False,
                       'truncate': False, 'shift': False,
                       'options': {'temperature': req['temperature'], 'seed': req['seed'],
                                   'num_ctx': self.num_ctx}}
            if req['json']:
                payload['format'] = draw_schema if draw_schema is not None else 'json'
            data = self._post('/api/chat', payload)
            self.draws += 1
            if 'message' not in data:
                raise LLMError(f'Ollama returned no message: {data.get("error", data)}')
            prompt_tokens = data.get('prompt_eval_count', 0) or 0
            output_tokens = data.get('eval_count', 0) or 0
            tokens_in, tokens_out = tokens_in + prompt_tokens, tokens_out + output_tokens
            if prompt_tokens + output_tokens >= self.num_ctx:
                raise LLMError(f'Ollama context window ({self.num_ctx}) reached at {prompt_tokens}+'
                               f'{output_tokens} tokens; raise num_ctx. Nothing here was trusted: a '
                               'truncated prompt produces a degraded action that reads as an agent error.')
            text = data['message'].get('content', '')
            usable, canonical = check_actor_reply(text, contract)
            if usable:
                self.repaired += int(bool(attempt))
                return Completion(canonical, tokens_in, tokens_out, req['model'])
        self.unrecovered += 1
        return Completion(text, tokens_in, tokens_out, req['model'], lenient=True)


class CachedLLM:
    """Disk cache in front of a client. Identical requests return identical completions,
    which makes prefix replay deterministic. Keyed by model digest, messages and options."""

    def __init__(self, inner, cache_dir, read_only=False, fallbacks=()):
        self.inner, self.read_only = inner, read_only
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Read-only caches consulted on a miss, never written to. A corpus merged from several
        # generation runs has one cache per run, and every actor response in them was already paid
        # for; without this, training re-buys all of them. Entries are content-addressed, so a hit
        # from any of them is byte-identical to what this cache would have stored. They stay
        # read-only so training cannot mutate a generation artifact.
        self.fallbacks = [Path(d) for d in fallbacks if Path(d).exists()]
        self.measurement_kind = getattr(inner, 'measurement_kind', 'model')
        self.hits = self.misses = self.fallback_hits = 0
        self.refused_writes = self.stale_poison = self.poisoned_hits = 0

    def _read(self, directory, key):
        path = Path(directory) / key[:2] / f'{key}.json'
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding='utf-8'))
        return Completion(data['text'], data['tokens_in'], data['tokens_out'], data['model'], cached=True,
                          lenient=data.get('lenient', False))

    def key(self, req):
        model_digest = self.inner.model_digest(req['model']) if hasattr(self.inner, 'model_digest') else ''
        return digest({'model': req['model'], 'model_digest': model_digest, 'messages': req['messages'],
                       'temperature': req['temperature'], 'seed': req['seed'], 'json': req['json']})

    def usable(self, completion, req):
        """Whether a completion may be served from, or written to, the cache.

        ``lenient`` alone is not enough: it is derived from finishReason and emptiness, so a reply that
        stops cleanly and is not a usable action passes it, and that is the class that rotted corpus-v1.
        The request's declared contract closes the gap. A request that declares none (an LLM judge
        baseline, a health check) is checked for leniency only, so this guard cannot start rejecting a
        caller that knows nothing about it and send it back to the API on every replay.
        """
        if completion.lenient:
            return False
        return check_actor_reply(completion.text, (req.get('meta') or {}).get('contract'))[0]

    def complete(self, req):
        key = self.key(req)
        path = self.cache_dir / key[:2] / f'{key}.json'
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            cached = Completion(data['text'], data['tokens_in'], data['tokens_out'], data['model'], cached=True,
                                lenient=data.get('lenient', False))
            if self.usable(cached, req):
                self.hits += 1
                return cached
            if self.read_only:
                # A frozen run cannot re-fetch, and serving poison silently is worse than counting it.
                self.hits += 1
                self.poisoned_hits += 1
                return cached
            self.stale_poison += 1          # written before this guard existed; re-fetch and overwrite it
        else:
            for directory in self.fallbacks:
                spare = self._read(directory, key)
                if spare is not None and self.usable(spare, req):
                    self.hits += 1
                    self.fallback_hits += 1
                    if not self.read_only:
                        self._store(path, spare)      # promote, so the next lookup is a direct hit
                    return spare
            if self.read_only:
                raise LLMError(f'cache miss in read-only mode for {req["meta"]}')
        self.misses += 1
        result = self.inner.complete(req)
        if not self.usable(result, req):
            self.refused_writes += 1        # never persist an unusable turn: that is how corpus-v1 rotted
            return result
        self._store(path, result)
        return result

    @staticmethod
    def _store(path, completion):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'text': completion.text, 'tokens_in': completion.tokens_in,
                                    'tokens_out': completion.tokens_out, 'model': completion.model,
                                    'lenient': completion.lenient}), encoding='utf-8')

    def stats(self):
        """Cache counters for the generation summary."""
        return {'hits': self.hits, 'misses': self.misses, 'fallback_hits': self.fallback_hits,
                'refused_writes': self.refused_writes, 'stale_poison': self.stale_poison,
                'poisoned_hits': self.poisoned_hits}


class ScriptedLLM:
    """Test double: ``responder(request) -> str or dict``. Marked as a fixture in reports."""
    measurement_kind = 'offline_fixture'

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def complete(self, req):
        self.calls.append(req)
        out = self.responder(req)
        text = out if isinstance(out, str) else json.dumps(out)
        return Completion(text, sum(len(m['content']) for m in req['messages']) // 4, len(text) // 4, 'scripted')


class GeminiActor:
    """Benchmark actor backed by agentrx's hardened HTTPProvider (Gemini via native generateContent).

    Reuses that transport and its retry/backoff; adds no new HTTP client. The API key is read from the
    environment by the provider (never from a config file). CachedLLM wraps this unchanged; model_digest
    pins the cache to the model id, since a hosted model has no weights digest.

    Decoding repair ladder. The transport retries 429/5xx, which is a different problem from a reply that
    decodes into nothing usable: under ``responseMimeType: application/json`` Gemini Flash-Lite aborts
    some action turns with finishReason MALFORMED_RESPONSE and no output tokens, and emits others that
    stop cleanly but are not parseable (one measured case was well-formed JSON spoiled by unescaped
    quotes inside a string value). A request that declares ``meta['contract']`` therefore gets up to
    three draws, strongest first: schema-constrained, then twice without the JSON mime type and with a
    strict formatting note on a shallow copy of the last message. Unusable rates measured on one tau task
    (benchmark/runs/threeway.log): schema 0/14, no mime type 2/20, mime type alone 4/23.

    A draw is kept only when the transport did not flag it lenient AND it satisfies the contract. That
    second condition is the one that matters: ``lenient`` is computed from finishReason and emptiness, so
    it cannot see a STOP reply whose text is not a usable action, which was 2 of the 4 failures in the
    measured mime-type run. ``lenient`` on the returned Completion therefore means every draw failed,
    which is exactly what CachedLLM refuses to write and what generation counts per episode.

    ``use_response_schema=False`` drops back to the mime-type-only first draw, for the pilot gate: a
    schema constrains the token stream, so it has to be shown not to push the actor into degenerate
    turns before a whole corpus is generated behind it.
    """
    measurement_kind = 'model'

    def __init__(self, model, base_url=None, api_key_env='GEMINI_API_KEY', settings=None, timeout=180, retries=3,
                 lenient_log=None, use_response_schema=True):
        from agentrx.providers import HTTPProvider   # lazy: keeps benchmark import-independent of agentrx
        config = {'provider': 'gemini', 'model': model, 'api_key_env': api_key_env,
                  'settings': settings or {}, 'timeout': timeout, 'retries': retries}
        if base_url:
            config['base_url'] = base_url
        self.provider = HTTPProvider.from_config(config)
        self.model = model
        self.lenient_log = lenient_log      # path: append one JSON line per draw outcome, with its cause
        self.use_response_schema = bool(use_response_schema)
        self.draws = self.repaired = self.unrecovered = 0

    def model_digest(self, model):
        return f'gemini:{model}'

    def stats(self):
        """Decoding counters for the generation summary (``repaired`` cost extra draws but kept the turn)."""
        return {'draws': self.draws, 'repaired': self.repaired, 'unrecovered': self.unrecovered,
                'http_requests': self.provider.http_requests, 'response_schema': self.use_response_schema}

    def _draw_plan(self, req):
        """``(contract, [(json_mode, response_schema, note), ...])``, strongest draw first.

        A request that declares no contract gets exactly one draw, which is the behaviour every other
        caller of this actor already relies on.
        """
        contract = (req.get('meta') or {}).get('contract')
        json_mode = bool(req['json'])
        if contract == ACTION_CONTRACT:
            schema = GEMINI_ACTION_SCHEMA if (self.use_response_schema and json_mode) else None
            return contract, [(json_mode, schema, None), (False, None, JSON_RETRY_NOTE),
                              (False, None, JSON_RETRY_NOTE)]
        if contract == TEXT_CONTRACT:
            return contract, [(json_mode, None, None), (json_mode, None, TEXT_RETRY_NOTE)]
        return contract, [(json_mode, None, None)]

    def _append(self, path, record):
        if not path:
            return
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except OSError:
            pass

    def _raw_log(self):
        """Sibling of the lenient log, or None when the actor was built without one (callers no-op)."""
        if not self.lenient_log:
            return None
        return os.path.join(os.path.dirname(self.lenient_log) or '.', 'lenient_raw.jsonl')

    def complete(self, req):
        meta = req.get('meta') or {}
        tag = {'agent': meta.get('agent'), 'step_index': meta.get('step_index'), 'task_id': meta.get('task_id')}
        contract, plan = self._draw_plan(req)
        tokens_in = tokens_out = 0
        text = ''
        for attempt, (json_mode, schema, note) in enumerate(plan):
            messages = req['messages'] if note is None else with_note(req['messages'], note)
            text, usage, info = self.provider.generate_content(messages, json_mode=json_mode,
                                                               response_schema=schema)
            self.draws += 1
            # Every draw was billed, so the episode carries the full cost, not just the kept draw's.
            tokens_in += usage.get('prompt_tokens', 0) or 0
            tokens_out += usage.get('completion_tokens', 0) or 0
            usable, canonical = check_actor_reply(text, contract)
            if info is None and usable:
                if attempt:
                    self.repaired += 1
                    self._append(self.lenient_log, {**tag, 'outcome': 'repaired', 'attempt': attempt,
                                                    'json_mode': json_mode, 'contract': contract})
                return Completion(canonical, tokens_in, tokens_out, self.model, lenient=False)
            self._append(self.lenient_log, {**(info or {}), **tag, 'outcome': 'rejected', 'attempt': attempt,
                                            'reason': 'lenient' if info is not None else 'contract',
                                            'json_mode': json_mode, 'response_schema': schema is not None,
                                            'contract': contract})
            # The complete decoded response body, pre-parse, for diagnosis (the parser is under suspicion).
            raw = getattr(self.provider, 'last_raw', None)
            if raw is not None:
                self._append(self._raw_log(), {**tag, 'attempt': attempt, 'raw': _truncate_strings(raw, 4000)})
        self.unrecovered += 1
        self._append(self.lenient_log, {**tag, 'outcome': 'unrecovered', 'draws': len(plan), 'contract': contract})
        return Completion(text, tokens_in, tokens_out, self.model, lenient=True)
