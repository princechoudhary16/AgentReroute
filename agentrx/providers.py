"""JSON model clients and persistent request-level provenance without API secrets."""
import copy
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, parse, request
from .io import digest, read_json, write_json
from .prompts import messages, prompt_identity, ranked_response_schema


class ProviderError(RuntimeError):
    pass


def env_key(name):
    """os.environ, else (Windows) the user-level HKCU\\Environment value a process started earlier cannot see."""
    key = os.environ.get(name)
    if not key and sys.platform == 'win32':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as reg:
                key, _ = winreg.QueryValueEx(reg, name)
        except OSError:
            pass
    return key


def _parse_json_object(text):
    """The JSON object in a model reply, tolerating <think> blocks, markdown fences and surrounding prose."""
    if not isinstance(text, str):
        raise ProviderError('Malformed model JSON response (TypeError)')
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned, flags=re.I).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        first, last = cleaned.find('{'), cleaned.rfind('}')
        try:
            value = json.loads(cleaned[first:last + 1]) if 0 <= first < last else None
        except json.JSONDecodeError:
            value = None
        if value is None:
            raise ProviderError('Malformed model JSON response (JSONDecodeError)') from None
    if not isinstance(value, dict):
        raise ProviderError('Model must return a JSON object')
    return value


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError('Model endpoint redirect rejected; configure its final URL')


class HTTPProvider:
    measurement_kind = 'model_measurement'

    def __init__(self, model, base_url, api_key, provider='openai', deployment=None,
                 api_version='2024-10-21', settings=None, timeout=180, retries=3,
                 retry_delay=1, max_input_chars=None):
        url = parse.urlparse(base_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ProviderError('Invalid model base URL (no credentials/query/fragment allowed)')
        if url.scheme == 'http' and url.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ProviderError('Remote model endpoints must use HTTPS')
        if not api_key:
            raise ProviderError('Missing model API credential')
        self.model, self.api_key = model, api_key
        self.provider, self.settings = provider, dict(settings or {})
        allowed = {'temperature', 'top_p', 'seed', 'max_completion_tokens', 'max_tokens', 'reasoning_effort'}
        if set(self.settings) - allowed:
            raise ProviderError('Unsupported sampling setting')
        self.timeout, self.retries, self.retry_delay = timeout, retries, retry_delay
        self.max_input_chars = max_input_chars
        if provider == 'azure':
            if not deployment:
                raise ProviderError('Azure needs AZURE_OPENAI_DEPLOYMENT')
            self.url = (base_url.rstrip('/') + '/openai/deployments/' + parse.quote(deployment, safe='') +
                        '/chat/completions?api-version=' + parse.quote(api_version, safe=''))
        else:
            self.url = base_url.rstrip('/') + '/chat/completions'
        self.identity = {'provider': provider, 'model': model, 'base_url': base_url,
                         'deployment': deployment, 'api_version': api_version if provider == 'azure' else 'v1',
                         'settings': self.settings, 'client': 'python-urllib', 'python': platform.python_version()}
        self.last_telemetry = {}
        self.last_raw = None
        self.http_requests = 0          # every HTTP request sent by this instance, retries included

    @classmethod
    def from_config(cls, config):
        kind = config.get('provider', 'openai')
        if kind not in ('openai', 'openai-compatible', 'azure', 'ollama', 'replay', 'gemini'):
            raise ProviderError('Unsupported model provider')
        if kind == 'replay':
            return ReplayProvider(config['replay_file'])
        if kind == 'ollama':
            return OllamaProvider.from_config(config)
        key_env = config.get('api_key_env',
                             'AZURE_OPENAI_API_KEY' if kind == 'azure'
                             else ('GEMINI_API_KEY' if kind == 'gemini' else 'OPENAI_API_KEY'))
        key = config.get('api_key') or env_key(key_env)
        if not key:
            raise ProviderError(f'Set {key_env} in your local environment or config before model-backed runs')
        if config.get('base_url_env') and not os.environ.get(config['base_url_env']):
            raise ProviderError(f"Set {config['base_url_env']} to your versioned model endpoint")
        default_base = ('https://generativelanguage.googleapis.com/v1beta/openai' if kind == 'gemini'
                        else ('' if kind == 'azure' else 'https://api.openai.com/v1'))
        base = (os.environ.get(config.get('base_url_env', ''))
                or config.get('base_url')
                or os.environ.get('AZURE_OPENAI_ENDPOINT' if kind == 'azure'
                                  else ('GEMINI_BASE_URL' if kind == 'gemini' else 'OPENAI_BASE_URL'),
                                  default_base))
        default_model = 'gemini-3.5-flash' if kind == 'gemini' else 'gpt-5'
        model = os.environ.get(config.get('model_env', '')) or config.get('model') or default_model
        return cls(model, base, key, provider=kind,
                   deployment=config.get('deployment') or os.environ.get('AZURE_OPENAI_DEPLOYMENT'),
                   api_version=config.get('api_version', os.environ.get('AZURE_OPENAI_API_VERSION', '2024-10-21')),
                   settings=config.get('settings'), timeout=config.get('timeout', 180),
                   retries=config.get('retries', 3), max_input_chars=config.get('max_input_chars'))

    def _backoff(self, attempt, retry_after=None):
        """Seconds before the next attempt: a numeric Retry-After header, else exponential; both capped at 60."""
        try:
            return min(60.0, max(0.0, float(retry_after)))
        except (TypeError, ValueError):
            return min(60, self.retry_delay * (2 ** attempt))

    def complete(self, stage, payload):
        body = {'model': self.model, 'messages': messages(stage, payload),
                'response_format': {'type': 'json_object'}, **self.settings}
        encoded = json.dumps(body, ensure_ascii=False).encode()
        if self.max_input_chars and sum(len(m['content']) for m in body['messages']) > self.max_input_chars:
            raise ProviderError('Configured input limit exceeded; no content was truncated')
        # Cloudflare in front of OpenRouter answers 403 to the default Python-urllib User-Agent.
        headers = {'Content-Type': 'application/json', 'User-Agent': 'AgentReroute/1.0',
                   'HTTP-Referer': 'https://github.com/AgentReroute', 'X-Title': 'AgentReroute'}
        headers['api-key' if self.provider == 'azure' else 'Authorization'] = (
            self.api_key if self.provider == 'azure' else 'Bearer ' + self.api_key)
        opener = request.build_opener(NoRedirect())
        start = time.monotonic()
        self.last_raw = None
        attempt = sent = 0
        while True:
            try:
                sent += 1
                self.http_requests += 1
                req = request.Request(self.url, data=encoded, headers=headers, method='POST')
                with opener.open(req, timeout=self.timeout) as response:
                    content = response.read(20_000_001)
                    if len(content) > 20_000_000:
                        raise ProviderError('Model response exceeds size limit')
                    raw = json.loads(content)
                    status = getattr(response, 'status', None)
                self.last_raw = raw
                self.last_telemetry = {'model': raw.get('model'), 'response_id': raw.get('id'),
                    'usage': raw.get('usage', {}), 'seconds': time.monotonic() - start,
                    'attempts': sent, 'http_status': status, 'response_format': 'response_format' in body,
                    'timestamp': datetime.now(timezone.utc).isoformat()}
                choice = raw['choices'][0]
                if choice.get('finish_reason') not in ('stop', None):
                    raise ProviderError(f"Incomplete model response: {choice.get('finish_reason')}")
                message = choice['message']
                if message.get('refusal'):
                    raise ProviderError('Model refused diagnostic request')
                return _parse_json_object(message['content'])
            except error.HTTPError as exc:
                retry_after = exc.headers.get('Retry-After') if exc.headers else None
                exc.close()
                if exc.code == 400 and self.provider == 'openai-compatible' and 'response_format' in body:
                    # Some OpenRouter :free upstreams reject response_format; the prompt still demands JSON.
                    del body['response_format']
                    encoded = json.dumps(body, ensure_ascii=False).encode()
                    continue
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise ProviderError(f'Model HTTP {exc.code}; response body omitted to protect credentials') from None
                delay = self._backoff(attempt, retry_after)
            except (error.URLError, TimeoutError, OSError):
                if attempt == self.retries:
                    raise ProviderError('Model transport failed after retries') from None
                delay = self._backoff(attempt)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise ProviderError(f'Malformed model JSON response ({type(exc).__name__})') from None
            time.sleep(delay)
            attempt += 1


class OllamaProvider:
    """Local Ollama via the native /api/chat endpoint (no credential, explicit context window).

    The OpenAI-compatible endpoint cannot set num_ctx and Ollama silently truncates prompts
    that exceed it, so the native API is used and any context overflow is raised as an error.
    """
    measurement_kind = 'model_measurement'
    supports_response_schema = True
    OPTIONS = {'temperature', 'top_p', 'top_k', 'seed', 'num_predict', 'repeat_penalty'}

    def __init__(self, model, base_url='http://127.0.0.1:11434', num_ctx=16384, think=None,
                 settings=None, timeout=900, retries=1, retry_delay=1, keep_alive='10m'):
        url = parse.urlparse(base_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ProviderError('Invalid Ollama base URL (no credentials/query/fragment allowed)')
        if url.scheme == 'http' and url.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ProviderError('Remote Ollama endpoints must use HTTPS')
        if not isinstance(num_ctx, int) or num_ctx < 2048:
            raise ProviderError('num_ctx must be an integer >= 2048')
        self.settings = dict(settings or {})
        if set(self.settings) - self.OPTIONS:
            raise ProviderError('Unsupported Ollama option')
        self.model, self.num_ctx, self.think, self.keep_alive = model, num_ctx, think, keep_alive
        self.timeout, self.retries, self.retry_delay = timeout, retries, retry_delay
        self.url = base_url.rstrip('/') + '/api/chat'
        self.base_url = base_url.rstrip('/')
        self._prepared = False
        self.identity = {'provider': 'ollama', 'model': model, 'base_url': base_url, 'num_ctx': num_ctx,
                         'think': think, 'settings': self.settings, 'client': 'python-urllib',
                         'python': platform.python_version()}
        self.last_telemetry = {}
        self.last_raw = None

    @classmethod
    def from_config(cls, config):
        if config.get('base_url_env') and not os.environ.get(config['base_url_env']):
            raise ProviderError(f"Set {config['base_url_env']} to your Ollama endpoint")
        base = (os.environ.get(config.get('base_url_env', '')) or config.get('base_url')
                or os.environ.get('OLLAMA_HOST_URL', 'http://127.0.0.1:11434'))
        model = os.environ.get(config.get('model_env', '')) or config.get('model', 'qwen2.5-coder:7b')
        return cls(model, base, num_ctx=config.get('num_ctx', 16384), think=config.get('think'),
                   settings=config.get('settings'), timeout=config.get('timeout', 900),
                   retries=config.get('retries', 1), keep_alive=config.get('keep_alive', '10m'))

    def prepare_for_run(self):
        """Resolve weights before experiment/cache identities; constructors stay offline."""
        if self._prepared:
            return
        metadata = {}
        opener = request.build_opener(NoRedirect())
        for path in ('/api/tags', '/api/version'):
            try:
                with opener.open(self.base_url + path, timeout=min(self.timeout, 15)) as response:
                    content = response.read(20_000_001)
                if len(content) > 20_000_000:
                    raise ProviderError('Ollama metadata exceeds size limit')
                value = json.loads(content)
                if not isinstance(value, dict):
                    raise ProviderError('Malformed Ollama metadata')
                metadata[path] = value
            except (error.URLError, TimeoutError, OSError, json.JSONDecodeError):
                raise ProviderError('Could not resolve local Ollama model/version; check `ollama serve`') from None
        tag = self.model if ':' in self.model.rsplit('/', 1)[-1] else self.model + ':latest'
        model = next((m for m in metadata['/api/tags'].get('models', [])
                      if m.get('name') in (self.model, tag) or m.get('model') in (self.model, tag)), None)
        if not model or not isinstance(model.get('digest'), str) or not model['digest']:
            raise ProviderError(f'Ollama model {self.model!r} not found; run: ollama pull {self.model}')
        self.identity.update(model_digest=model['digest'], model_details=model.get('details', {}),
                             ollama_version=metadata['/api/version'].get('version'))
        self._prepared = True

    def _post(self, body):
        opener = request.build_opener(NoRedirect())
        encoded = json.dumps(body, ensure_ascii=False).encode()
        for attempt in range(self.retries + 1):
            try:
                req = request.Request(self.url, data=encoded, headers={'Content-Type': 'application/json'},
                                      method='POST')
                with opener.open(req, timeout=self.timeout) as response:
                    content = response.read(20_000_001)
                if len(content) > 20_000_000:
                    raise ProviderError('Model response exceeds size limit')
                return json.loads(content)
            except json.JSONDecodeError:
                raise ProviderError('Malformed Ollama HTTP response: invalid JSON') from None
            except error.HTTPError as exc:
                detail = exc.read(4096).decode('utf-8', errors='replace').lower()
                exc.close()
                if any(term in detail for term in ('exceed_context_size', 'context size', 'context length', 'context window')):
                    raise ProviderError(f'Ollama context limit exceeded (HTTP {exc.code}); increase num_ctx; no response accepted') from None
                if exc.code == 404:
                    raise ProviderError(f'Ollama model {self.model!r} not found; run: ollama pull {self.model}') from None
                if exc.code not in (500, 502, 503, 504) or attempt == self.retries:
                    raise ProviderError(f'Ollama HTTP {exc.code}') from None
            except (error.URLError, TimeoutError, OSError):
                if attempt == self.retries:
                    raise ProviderError('Ollama unreachable; is `ollama serve` running?') from None
            time.sleep(min(10, self.retry_delay * 2**attempt))
        raise ProviderError('Ollama retries exhausted')

    def complete(self, stage, payload):
        return self.complete_messages(messages(stage, payload),
            response_schema=ranked_response_schema(payload) if stage == 'ranked_judge' else None)

    def complete_messages(self, chat_messages, *, response_schema=None):
        """Native JSON chat with caller-supplied messages and the same transport checks."""
        body = {'model': self.model, 'messages': chat_messages, 'format': response_schema if response_schema is not None else 'json',
                'stream': False, 'keep_alive': self.keep_alive,
                'truncate': False, 'shift': False,
                'options': {'num_ctx': self.num_ctx, **self.settings}}
        if self.think is not None:
            body['think'] = self.think
        self.last_raw = None
        start = time.monotonic()
        raw = self._post(body)
        self.last_raw = raw
        if not isinstance(raw, dict):
            raise ProviderError('Malformed Ollama response: expected an object')
        if raw.get('done') is not True:
            raise ProviderError('Incomplete Ollama response: done must be true')
        prompt_tokens, output_tokens = raw.get('prompt_eval_count', 0), raw.get('eval_count', 0)
        for key in ('prompt_eval_count', 'eval_count', 'load_duration', 'prompt_eval_duration', 'eval_duration'):
            value = raw.get(key, 0)
            if type(value) is not int or value < 0:
                raise ProviderError(f'Malformed Ollama response: invalid {key}')
        self.last_telemetry = {
            'model': raw.get('model'), 'seconds': time.monotonic() - start,
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': output_tokens,
                      'total_tokens': prompt_tokens + output_tokens},
            'load_seconds': raw.get('load_duration', 0) / 1e9,
            'prefill_seconds': raw.get('prompt_eval_duration', 0) / 1e9,
            'decode_seconds': raw.get('eval_duration', 0) / 1e9,
            'timestamp': datetime.now(timezone.utc).isoformat()}
        if prompt_tokens + output_tokens >= self.num_ctx:
            raise ProviderError(f'Context window ({self.num_ctx}) reached; raise num_ctx, nothing was trusted')
        if raw.get('done_reason') not in ('stop', None):
            raise ProviderError(f"Incomplete model response: {raw.get('done_reason')}")
        try:
            answer = json.loads(raw['message']['content'])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderError(f'Malformed model JSON response ({type(exc).__name__})') from None
        if not isinstance(answer, dict):
            raise ProviderError('Model must return a JSON object')
        return answer


class ReplayProvider:
    """Explicit fixture replay for offline smoke tests, never benchmark evidence."""
    measurement_kind = 'offline_fixture'
    def __init__(self, path):
        self.responses = read_json(path)
        self.identity = {'provider': 'replay', 'fixture_hash': digest(self.responses)}
        self.last_telemetry, self.last_raw = {}, None

    def complete(self, stage, payload):
        if stage not in self.responses:
            raise ProviderError(f'No fixture response for stage {stage}')
        answer = copy.deepcopy(self.responses[stage])
        self.last_raw = answer
        self.last_telemetry = {'fixture': True, 'usage': {}, 'seconds': 0}
        return answer


class CachedProvider:
    def __init__(self, provider, directory, namespace):
        prepare = getattr(provider, 'prepare_for_run', None)
        if prepare is not None:
            prepare()
        self.provider, self.directory, self.namespace = provider, Path(directory), namespace
        self.identity = provider.identity
        self.measurement_kind = provider.measurement_kind

    def complete(self, stage, payload):
        prompt = messages(stage, payload)
        identity = {'namespace': self.namespace, 'provider': self.identity,
                    'prompt_identity': prompt_identity(), 'stage': stage, 'messages': prompt}
        key = digest(identity)
        path = self.directory/f'{key}.json'
        if path.exists():
            record = read_json(path)
            if record['request_hash'] != key:
                raise ProviderError('Cached request hash mismatch')
            if record['status'] == 'ok':
                return record['response']
        try:
            answer = self.provider.complete(stage, payload)
            if not isinstance(answer, dict):
                raise ProviderError('Provider must return an object')
            write_json(path, {'request_hash': key, 'request': identity, 'status': 'ok',
                             'response': answer, 'raw_response': getattr(self.provider, 'last_raw', answer),
                             'telemetry': getattr(self.provider, 'last_telemetry', {}),
                             'measurement_kind': self.measurement_kind})
            return answer
        except Exception as exc:
            write_json(path, {'request_hash': key, 'request': identity, 'status': 'error',
                             'error': str(exc), 'raw_response': getattr(self.provider, 'last_raw', None),
                             'measurement_kind': self.measurement_kind})
            raise
