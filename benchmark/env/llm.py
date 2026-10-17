"""LLM clients: local Ollama, a deterministic response cache, and scripted test doubles.

Every completion goes through ``LLMClient.complete(request)``. A request is a
plain dict: ``{model, messages, temperature, seed, json, meta}``. ``meta`` is
never sent to the model and never part of the cache key; it only helps
scripted fixtures (and logs) know which agent is speaking.
"""
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..io import digest

DEFAULT_MODEL = 'qwen2.5:7b'
DEFAULT_HOST = 'http://localhost:11434'


@dataclass
class Completion:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ''
    cached: bool = False

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

    def __init__(self, host=DEFAULT_HOST, timeout=300, num_ctx=8192, opener=urllib.request.urlopen):
        self.host, self.timeout, self.num_ctx, self.opener = host.rstrip('/'), timeout, num_ctx, opener
        self._digests = {}

    def _post(self, path, payload):
        body = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(self.host + path, data=body, headers={'Content-Type': 'application/json'})
        try:
            with self.opener(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except (urllib.error.URLError, OSError) as exc:
            raise LLMError(f'Ollama request to {self.host}{path} failed: {exc}') from exc

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
            return Completion(data['text'], data['tokens_in'], data['tokens_out'], data['model'], cached=True)
        if self.read_only:
            raise LLMError(f'cache miss in read-only mode for {req["meta"]}')
        self.misses += 1
        result = self.inner.complete(req)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'text': result.text, 'tokens_in': result.tokens_in,
                                    'tokens_out': result.tokens_out, 'model': result.model}), encoding='utf-8')
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
