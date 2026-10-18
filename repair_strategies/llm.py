"""LLM access for the two strategies that need a second model reading.

The project runs models locally through Ollama; ``OllamaClient`` uses its
native chat API. No API keys or cloud endpoints are involved.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol
from urllib import error, request


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMReply:
    text: str
    tokens: int = 0


class LLMClient(Protocol):
    def complete(self, messages: list[dict], *, purpose: str) -> LLMReply: ...


@dataclass
class ScriptedLLM:
    """Deterministic stand-in: replies cycle through a list or come from a function of the request."""
    replies: list[str] | Callable[[list[dict], str], str] = field(default_factory=list)
    tokens_per_call: int = 0
    requests: list[dict] = field(default_factory=list)

    def complete(self, messages: list[dict], *, purpose: str) -> LLMReply:
        self.requests.append({'purpose': purpose, 'messages': messages})
        if callable(self.replies):
            text = self.replies(messages, purpose)
        elif self.replies:
            text = self.replies[(len(self.requests) - 1) % len(self.replies)]
        else:
            raise LLMError('ScriptedLLM has no replies')
        return LLMReply(text, self.tokens_per_call)


DEFAULT_OLLAMA_URL = 'http://127.0.0.1:11434'


def ollama_url(value: str | None = None) -> str:
    """Server URL from an argument, ``OLLAMA_HOST`` (as Ollama itself reads it), or the local default."""
    url = (value or os.environ.get('OLLAMA_HOST') or DEFAULT_OLLAMA_URL).strip().rstrip('/')
    if '://' not in url:
        url = 'http://' + url
    if not url.startswith(('http://', 'https://')):
        raise ValueError(f'Ollama URL must be http(s): {url!r}')
    return url


class OllamaClient:
    """Native Ollama chat API (``/api/chat``), non-streaming.

    Requests JSON output (``format: "json"``), disables silent prompt truncation
    and context shifting, and treats a reply cut at the token limit as an error.
    """

    def __init__(self, model: str, base_url: str | None = None, num_ctx: int = 16384, temperature: float = 0.0,
                 timeout: float = 300.0, keep_alive: str = '5m', json_mode: bool = True):
        if not model:
            raise ValueError('an Ollama model name is required, e.g. qwen2.5:7b')
        self.model, self.base_url = model, ollama_url(base_url)
        self.url = self.base_url + '/api/chat'
        self.num_ctx, self.temperature, self.timeout = num_ctx, temperature, timeout
        self.keep_alive, self.json_mode = keep_alive, json_mode

    def complete(self, messages: list[dict], *, purpose: str) -> LLMReply:
        body = {'model': self.model, 'messages': messages, 'stream': False, 'keep_alive': self.keep_alive,
                'options': {'num_ctx': self.num_ctx, 'temperature': self.temperature},
                'truncate': False, 'shift': False}
        if self.json_mode:
            body['format'] = 'json'
        req = request.Request(self.url, json.dumps(body).encode(), {'Content-Type': 'application/json'},
                              method='POST')
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except error.HTTPError as exc:
            with exc:
                detail = exc.read().decode('utf-8', 'replace')[:300]
            hint = f' (is the model pulled? run: ollama pull {self.model})' if exc.code == 404 else ''
            raise LLMError(f'{purpose}: Ollama HTTP {exc.code}: {detail}{hint}') from None
        except (error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f'{purpose}: cannot reach Ollama at {self.base_url} ({exc}); is "ollama serve" '
                           'running?') from None
        except json.JSONDecodeError:
            raise LLMError(f'{purpose}: Ollama response is not JSON') from None
        if payload.get('error'):
            raise LLMError(f'{purpose}: Ollama error: {payload["error"]}')
        message = payload.get('message')
        if not isinstance(message, dict) or not isinstance(message.get('content'), str):
            raise LLMError(f'{purpose}: unexpected Ollama response shape')
        if payload.get('done') is False:
            raise LLMError(f'{purpose}: Ollama response is incomplete')
        if payload.get('done_reason') == 'length':
            raise LLMError(f'{purpose}: reply truncated at the token limit; raise num_ctx')
        tokens = int(payload.get('prompt_eval_count') or 0) + int(payload.get('eval_count') or 0)
        return LLMReply(message['content'], tokens)


def parse_json_object(text: str) -> dict:
    """The first JSON object in a reply (models sometimes wrap it in prose or a code fence)."""
    match = re.search(r'\{.*\}', text or '', flags=re.S)
    if not match:
        raise LLMError('reply contains no JSON object')
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(f'reply JSON is invalid: {exc}') from None
    if not isinstance(value, dict):
        raise LLMError('reply JSON is not an object')
    return value
