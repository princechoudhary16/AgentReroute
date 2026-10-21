"""Fresh-context verification gate.

The verifier sees only the task view and the final transcript of a rollout.
It never sees the repair directive, the RCA rationale, the chosen strategy,
the original failed trace, gold labels, or rewards. A repair counts as
recovered only if the verifier passes it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from .llm import LLMClient
from .replay import Rollout, TaskView
from .trace import Step

SYSTEM_PROMPT = (
    'You are an independent auditor of a customer-service / agent transcript. You did not '
    'take part in it. Decide whether the agent fully and correctly accomplished the user\'s '
    'task while following the policy. Judge only from the transcript: tool outputs are the '
    'ground truth for facts; the agent\'s claims are not. Reply with JSON only: '
    '{"verdict": "PASS" | "FAIL", "score": <0..1>, "gaps": [<short strings>]}.')


@dataclass(frozen=True)
class VerifierResult:
    passed: bool
    score: float
    verdict: str
    gaps: tuple[str, ...] = ()
    tokens: int = 0
    raw: str = field(default='', compare=False)


class Verifier(Protocol):
    def verify(self, task: TaskView, rollout: Rollout) -> VerifierResult: ...


def render_transcript(steps: list[Step], max_chars: int = 1500) -> str:
    """Plain transcript without injected guidance and without the system prompt."""
    lines = []
    for step in steps:
        if step.injected or step.actor == 'system':
            continue
        content = step.content if len(step.content) <= max_chars else step.content[:max_chars] + ' …[truncated]'
        if step.tool_calls:
            calls = '; '.join(f'{c.name}({json.dumps(c.arguments, ensure_ascii=False)})' for c in step.tool_calls)
            lines.append(f'[{step.role}] calls {calls}' + (f' — {content}' if content else ''))
        elif step.actor == 'tool':
            lines.append(f'[tool {step.name or ""}] {content}')
        else:
            lines.append(f'[{step.role}] {content}')
    return '\n'.join(lines)


class FreshContextVerifier:
    def __init__(self, llm: LLMClient, threshold: float = 0.5, include_policy: bool = True):
        self.llm = llm
        self.threshold = threshold
        self.include_policy = include_policy

    def build_request(self, task: TaskView, rollout: Rollout) -> list[dict]:
        parts = [f'## Task\n{task.instruction or "(not given; infer it from the user turns)"}']
        if self.include_policy and task.policy:
            parts.append(f'## Policy\n{task.policy}')
        parts.append(f'## Transcript\n{render_transcript(rollout.steps)}')
        return [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': '\n\n'.join(parts)}]

    def verify(self, task: TaskView, rollout: Rollout) -> VerifierResult:
        reply = self.llm.complete(self.build_request(task, rollout), purpose='verify')
        verdict, score, gaps = parse_verdict(reply.text)
        return VerifierResult(verdict == 'PASS' and score >= self.threshold, score, verdict,
                              gaps, reply.tokens, reply.text)


def parse_verdict(text: str) -> tuple[str, float, tuple[str, ...]]:
    """Lenient JSON parse; anything unparseable is a FAIL with score 0."""
    match = re.search(r'\{.*\}', text or '', flags=re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    verdict = str(data.get('verdict', 'FAIL')).upper()
    if verdict not in ('PASS', 'FAIL'):
        verdict = 'FAIL'
    try:
        score = min(1.0, max(0.0, float(data.get('score', 1.0 if verdict == 'PASS' else 0.0))))
    except (TypeError, ValueError):
        score = 0.0
    gaps = data.get('gaps', [])
    return verdict, score, tuple(str(g) for g in gaps) if isinstance(gaps, list) else ()
