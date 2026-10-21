"""Minimal LLM client interface. Live clients are out of scope for the offline core."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass(frozen=True)
class LLMReply:
    text: str
    tokens: int = 0


class LLMClient(Protocol):
    def complete(self, messages: list[dict], *, purpose: str) -> LLMReply: ...


@dataclass
class ScriptedLLM:
    """Deterministic stand-in: replies from a list (cycled) or a function of the request."""
    replies: list[str] | Callable[[list[dict], str], str] = field(default_factory=list)
    tokens_per_call: int = 500
    requests: list[dict] = field(default_factory=list)

    def complete(self, messages: list[dict], *, purpose: str) -> LLMReply:
        self.requests.append({'purpose': purpose, 'messages': messages})
        if callable(self.replies):
            text = self.replies(messages, purpose)
        else:
            text = self.replies[(len(self.requests) - 1) % len(self.replies)]
        return LLMReply(text, self.tokens_per_call)
