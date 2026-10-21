"""Plan-state tracking: which required actions are done, failed, or still pending."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Mapping

from ..context import TaskSpec
from ..evidence.tool_output import parse_output
from ..trace import USER, ToolCall, Trace
from .rules import is_affirmative

DONE, FAILED, PENDING = 'done', 'failed', 'pending'


@dataclass(frozen=True)
class Obligation:
    tool: str
    arguments: Mapping = field(default_factory=dict)    # subset that a matching call must carry
    description: str = ''
    source: str = 'explicit'                             # explicit | confirmed | contract

    def matches(self, call: ToolCall) -> bool:
        if call.name != self.tool:
            return False
        for name, expected in self.arguments.items():
            actual = call.arguments.get(name)
            if isinstance(expected, list) and isinstance(actual, list):
                if sorted(map(str, expected)) != sorted(map(str, actual)):
                    return False
            elif actual != expected:
                return False
        return True

    def label(self) -> str:
        if self.description:
            return self.description
        args = ', '.join(f'{k}={v}' for k, v in self.arguments.items())
        return f'{self.tool}({args})'

    def to_dict(self) -> dict:
        return {'tool': self.tool, 'arguments': dict(self.arguments), 'description': self.description,
                'source': self.source}

    @classmethod
    def from_dict(cls, data: Mapping) -> 'Obligation':
        return cls(data['tool'], dict(data.get('arguments') or {}), data.get('description', ''),
                   data.get('source', 'explicit'))


@dataclass(frozen=True)
class Progress:
    obligation: Obligation
    status: str
    step_index: int | None = None
    error: str = ''


@dataclass(frozen=True)
class PlanState:
    items: tuple[Progress, ...]

    def _with(self, status):
        return tuple(p for p in self.items if p.status == status)

    @property
    def done(self) -> tuple[Progress, ...]:
        return self._with(DONE)

    @property
    def failed(self) -> tuple[Progress, ...]:
        return self._with(FAILED)

    @property
    def pending(self) -> tuple[Progress, ...]:
        return self._with(PENDING)

    @property
    def complete(self) -> bool:
        return all(p.status == DONE for p in self.items)

    def render(self) -> str:
        lines = ['Plan state (from the verified record of tool results):']
        for p in self.items:
            mark = {DONE: 'DONE', FAILED: 'FAILED', PENDING: 'PENDING'}[p.status]
            where = f' at step {p.step_index}' if p.step_index is not None else ''
            error = f' - {p.error}' if p.error else ''
            lines.append(f'- [{mark}] {p.obligation.label()}{where}{error}')
        if self.pending or self.failed:
            lines.append('Complete every PENDING or FAILED action before replying that the task is done.')
        return '\n'.join(lines)


def track(trace: Trace, obligations, from_step: int = 1) -> PlanState:
    """Status of each obligation from calls made at or after ``from_step``."""
    exchanges = [x for x in trace.tool_exchanges() if x.step_index >= from_step]
    items = []
    for obligation in obligations:
        status, step, error = PENDING, None, ''
        for x in exchanges:
            if not obligation.matches(x.call):
                continue
            parsed = parse_output(x.result.content) if x.result is not None else None
            if parsed is not None and parsed.ok:
                status, step, error = DONE, x.step_index, ''
                break
            status, step = FAILED, x.step_index
            error = (parsed.error if parsed is not None else 'no result returned') or ''
        items.append(Progress(obligation, status, step, error[:160]))
    return PlanState(tuple(items))


def _proposed_calls(text: str, task: TaskSpec) -> list[ToolCall]:
    """Write actions an agent message proposes, with any JSON arguments that follow the tool name."""
    calls = []
    for name in sorted(task.write_tools):
        for m in re.finditer(re.escape(name), text):
            arguments = {}
            brace = text.find('{', m.end())
            if brace != -1 and not text[m.end():brace].strip():
                try:
                    value, _ = json.JSONDecoder().raw_decode(text[brace:])
                    if isinstance(value, dict):
                        arguments = value
                except json.JSONDecodeError:
                    pass
            calls.append(ToolCall(name, arguments))
    return calls


def confirmed_obligations(trace: Trace, task: TaskSpec, before: int | None = None) -> tuple[Obligation, ...]:
    """Write actions the agent proposed and the user explicitly confirmed."""
    events = trace.before(before) if before is not None else trace.events
    found: list[Obligation] = []
    proposals: list[ToolCall] = []
    for event in events:
        if event.is_agent and event.text:
            proposals = _proposed_calls(event.text, task) or proposals
        elif event.kind == USER:
            if proposals and is_affirmative(event.content):
                for call in proposals:
                    obligation = Obligation(call.name, call.arguments, source='confirmed')
                    if obligation not in found:
                        found.append(obligation)
            proposals = []
    return tuple(found)
