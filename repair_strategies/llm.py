"""LLM access used by the few strategies that need a second model reading.

``OpenAICompatibleClient`` talks to any ``/chat/completions`` endpoint, which
covers OpenAI, Azure-compatible gateways, vLLM, and Ollama's ``/v1`` API.
Keys come from environment variables and are never stored.
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


class OpenAICompatibleClient:
    def __init__(self, base_url: str, model: str, api_key_env: str | None = None, timeout: float = 60.0,
                 temperature: float | None = 0.0, max_tokens: int | None = None, json_mode: bool = False):
        if not base_url.startswith(('http://', 'https://')):
            raise ValueError('base_url must be an http(s) URL')
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.model, self.timeout = model, timeout
        self.temperature, self.max_tokens, self.json_mode = temperature, max_tokens, json_mode
        self.api_key = None
        if api_key_env:
            self.api_key = os.environ.get(api_key_env)
            if not self.api_key:
                raise LLMError(f'Environment variable {api_key_env} is not set')

    def complete(self, messages: list[dict], *, purpose: str) -> LLMReply:
        body = {'model': self.model, 'messages': messages}
        if self.temperature is not None:
            body['temperature'] = self.temperature
        if self.max_tokens is not None:
            body['max_tokens'] = self.max_tokens
        if self.json_mode:
            body['response_format'] = {'type': 'json_object'}
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        req = request.Request(self.url, json.dumps(body).encode(), headers, method='POST')
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except error.HTTPError as exc:
            with exc:
                detail = exc.read().decode('utf-8', 'replace')[:300]
            raise LLMError(f'{purpose}: HTTP {exc.code} from {self.url}: {detail}') from None
        except (error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f'{purpose}: cannot reach {self.url}: {exc}') from None
        except json.JSONDecodeError:
            raise LLMError(f'{purpose}: response is not JSON') from None
        try:
            choice = payload['choices'][0]
            text = choice['message']['content'] or ''
        except (KeyError, IndexError, TypeError):
            raise LLMError(f'{purpose}: unexpected response shape') from None
        if choice.get('finish_reason') == 'length':
            raise LLMError(f'{purpose}: response truncated at max_tokens')
        usage = payload.get('usage') or {}
        tokens = usage.get('total_tokens') or (usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0))
        return LLMReply(text, int(tokens or 0))


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
