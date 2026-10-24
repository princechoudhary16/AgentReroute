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


class OllamaClient:
    """Chat completions from a local Ollama server (``/api/chat``)."""
    measurement_kind = 'model'

    def __init__(self, host=DEFAULT_HOST, timeout=300, num_ctx=8192, opener=urllib.request.urlopen,
                 retries=4, backoff=10.0, sleep=time.sleep):
        self.host, self.timeout, self.num_ctx, self.opener = host.rstrip('/'), timeout, num_ctx, opener
        self.retries, self.backoff, self.sleep = retries, backoff, sleep
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
                exc.close()
                if exc.code != 429 and exc.code < 500:
                    raise LLMError(f'Ollama request to {self.host}{path} failed: {exc}') from exc
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

    def complete(self, req):
        payload = {'model': req['model'], 'messages': req['messages'], 'stream': False,
                   'options': {'temperature': req['temperature'], 'seed': req['seed'], 'num_ctx': self.num_ctx}}
        if req['json']:
            payload['format'] = 'json'
        data = self._post('/api/chat', payload)
        if 'message' not in data:
            raise LLMError(f'Ollama returned no message: {data.get("error", data)}')
        return Completion(data['message'].get('content', ''), data.get('prompt_eval_count', 0) or 0,
                          data.get('eval_count', 0) or 0, req['model'])


class CachedLLM:
    """Disk cache in front of a client. Identical requests return identical completions,
    which makes prefix replay deterministic. Keyed by model digest, messages and options."""

    def __init__(self, inner, cache_dir, read_only=False):
        self.inner, self.read_only = inner, read_only
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.measurement_kind = getattr(inner, 'measurement_kind', 'model')
        self.hits = self.misses = 0

    def key(self, req):
        model_digest = self.inner.model_digest(req['model']) if hasattr(self.inner, 'model_digest') else ''
        return digest({'model': req['model'], 'model_digest': model_digest, 'messages': req['messages'],
                       'temperature': req['temperature'], 'seed': req['seed'], 'json': req['json']})

    def complete(self, req):
        key = self.key(req)
        path = self.cache_dir / key[:2] / f'{key}.json'
        if path.exists():
            self.hits += 1
            data = json.loads(path.read_text(encoding='utf-8'))
            return Completion(data['text'], data['tokens_in'], data['tokens_out'], data['model'], cached=True,
                              lenient=data.get('lenient', False))
        if self.read_only:
            raise LLMError(f'cache miss in read-only mode for {req["meta"]}')
        self.misses += 1
        result = self.inner.complete(req)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'text': result.text, 'tokens_in': result.tokens_in,
                                    'tokens_out': result.tokens_out, 'model': result.model,
                                    'lenient': result.lenient}), encoding='utf-8')
        return result


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
    """Benchmark actor backed by agentrx's hardened HTTPProvider (Gemini via its OpenAI-compatible endpoint).

    Reuses that transport and its retry/backoff; adds no new HTTP client. The API key is read from the
    environment by the provider (never from a config file). CachedLLM wraps this unchanged; model_digest
    pins the cache to the model id, since a hosted model has no weights digest.
    """
    measurement_kind = 'model'

    def __init__(self, model, base_url=None, api_key_env='GEMINI_API_KEY', settings=None, timeout=180, retries=3,
                 lenient_log=None):
        from agentrx.providers import HTTPProvider   # lazy: keeps benchmark import-independent of agentrx
        config = {'provider': 'gemini', 'model': model, 'api_key_env': api_key_env,
                  'settings': settings or {}, 'timeout': timeout, 'retries': retries}
        if base_url:
            config['base_url'] = base_url
        self.provider = HTTPProvider.from_config(config)
        self.model = model
        self.lenient_log = lenient_log      # path: append one JSON line per lenient response, with its cause

    def model_digest(self, model):
        return f'gemini:{model}'

    def complete(self, req):
        text, usage, info = self.provider.generate_content(req['messages'], json_mode=req['json'])
        if info is not None and self.lenient_log:
            meta = req.get('meta', {})
            tag = {'agent': meta.get('agent'), 'step_index': meta.get('step_index'), 'task_id': meta.get('task_id')}
            try:
                with open(self.lenient_log, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({**info, **tag}) + '\n')
            except OSError:
                pass
            # The complete decoded response body, pre-parse, for diagnosis (the parser itself is under suspicion).
            raw = getattr(self.provider, 'last_raw', None)
            if raw is not None:
                raw_path = os.path.join(os.path.dirname(self.lenient_log) or '.', 'lenient_raw.jsonl')
                try:
                    with open(raw_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({**tag, 'raw': _truncate_strings(raw, 4000)}) + '\n')
                except OSError:
                    pass
        return Completion(text, usage.get('prompt_tokens', 0) or 0, usage.get('completion_tokens', 0) or 0,
                          self.model, lenient=info is not None)


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
