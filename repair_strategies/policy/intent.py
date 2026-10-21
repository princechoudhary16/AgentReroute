"""Intent contracts: what the user asked for, made explicit and checkable.

Intent-plan misalignment is an agent solving a different problem than the one
posed. A contract states the goal, the entities in scope, the actions that are
required and those that are forbidden, so every planned action can be validated
against it instead of re-asking the agent to "remember the goal".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..context import TaskSpec
from ..evidence.claims import EMAIL, ID, argument_claims, extract_claims
from ..evidence.tool_output import is_error
from ..llm import LLMClient, LLMError, parse_json_object
from ..trace import USER, ToolCall, Trace
from .plan_state import Obligation, confirmed_obligations


@dataclass(frozen=True)
class IntentContract:
    goal: str
    entities: frozenset[str] = frozenset()          # lower-case identifiers in scope
    required: tuple[Obligation, ...] = ()
    forbidden_tools: frozenset[str] = frozenset()
    constraints: tuple[str, ...] = ()
    source: str = 'rules'
    notes: tuple[str, ...] = field(default=(), compare=False)

    def validate(self, call: ToolCall, task: TaskSpec, extra_entities=frozenset()) -> list[str]:
        """Reasons a call breaks the contract (empty when it is consistent with it).

        ``extra_entities`` are identifiers that became in scope later (found by lookups about
        in-scope entities during the replay, see ``scoped_entities``).
        """
        issues = []
        if call.name in self.forbidden_tools:
            issues.append(f'{call.name} is forbidden by the user\'s request')
        if task.is_write(call.name):
            required_tools = {o.tool for o in self.required}
            if required_tools and call.name not in required_tools:
                issues.append(f'{call.name} is not one of the requested actions ({", ".join(sorted(required_tools))})')
            if self.entities:
                allowed = self.entities | frozenset(extra_entities)
                for claim in argument_claims(call, 0):
                    if claim.kind in (ID, EMAIL) and claim.value not in allowed:
                        issues.append(f'{claim.argument}={claim.raw} is outside the user\'s request')
        return issues

    def render(self) -> str:
        lines = [f'Goal: {self.goal}']
        if self.required:
            lines.append('Required actions: ' + '; '.join(o.label() for o in self.required))
        if self.forbidden_tools:
            lines.append('Do not call: ' + ', '.join(sorted(self.forbidden_tools)))
        if self.constraints:
            lines.append('Constraints: ' + '; '.join(self.constraints))
        if self.entities:
            lines.append('Only act on: ' + ', '.join(sorted(self.entities)))
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        return {'goal': self.goal, 'entities': sorted(self.entities), 'required': [o.to_dict() for o in self.required],
                'forbidden_tools': sorted(self.forbidden_tools), 'constraints': list(self.constraints),
                'source': self.source}

    @classmethod
    def from_dict(cls, data: Mapping) -> 'IntentContract':
        return cls(goal=data.get('goal', ''), entities=frozenset(data.get('entities', ())),
                   required=tuple(Obligation.from_dict(o) for o in data.get('required', ())),
                   forbidden_tools=frozenset(data.get('forbidden_tools', ())),
                   constraints=tuple(data.get('constraints', ())), source=data.get('source', 'rules'))


def scoped_entities(trace: Trace, before: int) -> frozenset[str]:
    """Identifiers the user named, plus those returned by lookups about them (one hop at a time)."""
    scope: set[str] = set()
    for event in trace.before(before):
        if event.kind == USER:
            scope |= {c.value for c in extract_claims(event.content, event.step_index) if c.kind in (ID, EMAIL)}
    if trace.instruction:
        scope |= {c.value for c in extract_claims(trace.instruction, 0) if c.kind in (ID, EMAIL)}
    for x in trace.tool_exchanges():
        if x.step_index >= before or x.result is None or is_error(x.result.content):
            continue
        arguments = {c.value for c in argument_claims(x.call, x.step_index)}
        if arguments & scope:
            scope |= {c.value for c in extract_claims(x.result.content, x.result.step_index) if c.kind in (ID, EMAIL)}
    return frozenset(scope)


class RuleBasedExtractor:
    """Goal from the user's request; entities by provenance; required actions the user confirmed."""

    def extract(self, trace: Trace, task: TaskSpec, before: int) -> IntentContract:
        users = [e.content for e in trace.before(before) if e.kind == USER and e.content.strip()]
        goal = task.instruction or trace.instruction or (users[0] if users else '')
        return IntentContract(goal=goal.strip(), entities=scoped_entities(trace, before),
                              required=confirmed_obligations(trace, task, before), source='rules')


_PROMPT = ('You extract the user\'s intent from a customer-service conversation. Use only what the user said. '
           'Return one JSON object: {"goal": str, "required_actions": [{"tool": str, "arguments": {}}], '
           '"forbidden_tools": [str], "constraints": [str]}. Only name tools from this list: ')


class LLMContractExtractor:
    """Asks a model for the contract, validates it against the tool list, and falls back to rules."""

    def __init__(self, llm: LLMClient, fallback: RuleBasedExtractor | None = None):
        self.llm = llm
        self.fallback = fallback or RuleBasedExtractor()

    def extract(self, trace: Trace, task: TaskSpec, before: int) -> IntentContract:
        base = self.fallback.extract(trace, task, before)
        turns = [f'[{e.kind}] {e.content if e.kind == USER else e.text}' for e in trace.before(before)
                 if e.kind == USER or (e.is_agent and e.text)]
        messages = [{'role': 'system', 'content': _PROMPT + ', '.join(sorted(task.tools))},
                    {'role': 'user', 'content': '\n'.join(turns)[-12000:]}]
        try:
            data = parse_json_object(self.llm.complete(messages, purpose='intent_contract').text)
            required, notes = [], []
            for item in data.get('required_actions') or []:
                tool = item.get('tool') if isinstance(item, dict) else None
                if tool in task.tools:
                    args = item.get('arguments') if isinstance(item.get('arguments'), dict) else {}
                    required.append(Obligation(tool, args, source='contract'))
                else:
                    notes.append(f'dropped unknown tool {tool!r}')
            forbidden = {t for t in data.get('forbidden_tools') or [] if t in task.tools}
            constraints = tuple(str(c) for c in data.get('constraints') or [] if str(c).strip())
            goal = str(data.get('goal') or base.goal).strip()
        except (LLMError, AttributeError, TypeError) as exc:
            return IntentContract(base.goal, base.entities, base.required, base.forbidden_tools, base.constraints,
                                  'rules', (f'LLM contract failed: {exc}',))
        merged = list(base.required) + [o for o in required if not any(o.tool == b.tool for b in base.required)]
        return IntentContract(goal, base.entities, tuple(merged), frozenset(forbidden), constraints, 'llm',
                              tuple(notes))
