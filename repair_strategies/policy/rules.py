"""Declarative policy rules, checked before a tool call executes.

A reminder in the prompt can be ignored during re-execution; a gate cannot.
Rules are data (see ``resources/retail.json``) and compile against a
``TaskSpec`` so unknown tools or malformed rules fail at load time.

Kinds:

``requires_prior_call``  ``tools`` need a successful call to one of ``any_of`` first
``requires_confirmation`` ``tools`` need an affirmative user turn answering a proposal of that action
``argument_enum``        ``tool.argument`` must be one of ``values``
``at_most_once``         ``tool`` may succeed once per value of ``key`` arguments
``argument_grounded``    ``arguments`` of ``tools`` must appear in user turns or tool outputs
``forbidden_tool``       ``tools`` may not be called
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Mapping

from ..context import TaskSpec
from ..evidence.claims import EvidenceIndex
from ..evidence.tool_output import is_error
from ..trace import USER, ToolCall, Trace

PRIOR, CONFIRM, ENUM, ONCE, GROUNDED, FORBIDDEN = (
    'requires_prior_call', 'requires_confirmation', 'argument_enum', 'at_most_once', 'argument_grounded',
    'forbidden_tool')
KINDS = (PRIOR, CONFIRM, ENUM, ONCE, GROUNDED, FORBIDDEN)
WRITE_ALIAS = '@write'

_AFFIRMATIVE = re.compile(
    r"^\W*(yes|yeah|yep|yup|y|sure|ok|okay|confirm(ed)?|correct|right|please (do|proceed|go ahead)|go ahead|"
    r"proceed|do it|that'?s (right|correct|fine)|sounds good|i confirm)\b", re.I)
_REQUIRED = {PRIOR: ('tools', 'any_of'), CONFIRM: ('tools',), ENUM: ('tool', 'argument', 'values'),
             ONCE: ('tool', 'key'), GROUNDED: ('tools', 'arguments'), FORBIDDEN: ('tools',)}


class RuleError(ValueError):
    pass


def is_affirmative(text: str) -> bool:
    return bool(_AFFIRMATIVE.match(text or ''))


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str
    message: str
    tools: frozenset[str] = frozenset()
    params: Mapping = field(default_factory=dict)

    def applies_to(self, tool: str) -> bool:
        return tool in self.tools

    def to_dict(self) -> dict:
        data = {'id': self.id, 'kind': self.kind, 'message': self.message}
        data.update(self.params)
        if self.kind in (ENUM, ONCE):
            data['tool'] = next(iter(self.tools))
        else:
            data['tools'] = sorted(self.tools)
        return data


@dataclass(frozen=True)
class Violation:
    rule_id: str
    kind: str
    step_index: int
    tool: str
    message: str
    detail: str = ''

    def describe(self) -> str:
        return f'step {self.step_index} {self.tool}: {self.message}' + (f' ({self.detail})' if self.detail else '')


def compile_rules(specs, task: TaskSpec) -> tuple[Rule, ...]:
    rules, ids = [], set()
    known = set(task.tools)

    def resolve(names, rule_id):
        names = [names] if isinstance(names, str) else list(names or ())
        out = set()
        for name in names:
            if name == WRITE_ALIAS:
                out |= task.write_tools
            elif known and name not in known:
                raise RuleError(f'{rule_id}: unknown tool {name!r}')
            else:
                out.add(name)
        if not out:
            raise RuleError(f'{rule_id}: names no tools')
        return frozenset(out)

    for spec in specs:
        if not isinstance(spec, Mapping):
            raise RuleError(f'Rule must be an object: {spec!r}')
        rule_id, kind = spec.get('id'), spec.get('kind')
        if not rule_id or rule_id in ids:
            raise RuleError(f'Rule id missing or duplicated: {rule_id!r}')
        ids.add(rule_id)
        if kind not in KINDS:
            raise RuleError(f'{rule_id}: unknown kind {kind!r}')
        missing = [k for k in _REQUIRED[kind] if k not in spec]
        if missing:
            raise RuleError(f'{rule_id}: missing {missing}')
        params = {k: v for k, v in spec.items() if k not in ('id', 'kind', 'message', 'tools', 'tool')}
        tools = resolve(spec.get('tools', spec.get('tool')), rule_id)
        if kind in (ENUM, ONCE) and len(tools) != 1:
            raise RuleError(f'{rule_id}: {kind} applies to exactly one tool')
        if kind == PRIOR:
            params['any_of'] = sorted(resolve(spec['any_of'], rule_id))
        if kind == ENUM and not isinstance(spec['values'], list):
            raise RuleError(f'{rule_id}: values must be a list')
        if kind in (ONCE, GROUNDED):
            field_name = 'key' if kind == ONCE else 'arguments'
            value = spec[field_name]
            params[field_name] = [value] if isinstance(value, str) else list(value)
        rules.append(Rule(rule_id, kind, str(spec.get('message') or rule_id), tools, params))
    return tuple(rules)


class RuleGate:
    """Evaluates rules for a proposed call given everything that happened before it."""

    def __init__(self, rules: tuple[Rule, ...]):
        self.rules = tuple(rules)

    @classmethod
    def for_task(cls, task: TaskSpec, extra=()) -> 'RuleGate':
        return cls(compile_rules(tuple(task.rules) + tuple(extra), task))

    def check(self, trace: Trace, step_index: int, call: ToolCall) -> list[Violation]:
        violations = []
        for rule in self.rules:
            if rule.applies_to(call.name):
                detail = _CHECKS[rule.kind](rule, trace, step_index, call)
                if detail is not None:
                    violations.append(Violation(rule.id, rule.kind, step_index, call.name, rule.message, detail))
        return violations

    def audit(self, trace: Trace, from_step: int = 1) -> list[Violation]:
        """Every call made from ``from_step`` on, each checked against its own prefix."""
        found = []
        for event in trace.events:
            if event.step_index >= from_step and event.is_agent:
                for call in event.tool_calls:
                    found.extend(self.check(trace, event.step_index, call))
        return found

    def rules_for(self, tools) -> tuple[Rule, ...]:
        tools = set(tools)
        return tuple(r for r in self.rules if r.tools & tools)


# ---------------------------------------------------------------------- rule checks
def _successful(trace: Trace, before: int, names) -> list:
    return [x for x in trace.tool_exchanges()
            if x.step_index < before and x.call.name in names and x.result is not None
            and not is_error(x.result.content)]


def _prior(rule, trace, step, call):
    if _successful(trace, step, set(rule.params['any_of'])):
        return None
    return f'no successful call to {" or ".join(rule.params["any_of"])} before this step'


def _mentions(text: str, call: ToolCall) -> bool:
    lowered = (text or '').lower()
    if call.name.lower() in lowered or call.name.replace('_', ' ').lower() in lowered:
        return True
    values = []
    for value in call.arguments.values():
        values.extend(value if isinstance(value, list) else [value])
    return any(isinstance(v, str) and len(v) >= 4 and v.lower() in lowered for v in values)


def _confirm(rule, trace, step, call):
    prior = trace.before(step)
    for i in range(len(prior) - 1, -1, -1):
        event = prior[i]
        if event.kind != USER or not is_affirmative(event.content):
            continue
        # The affirmative turn must answer an agent message that proposed this action.
        for earlier in reversed(prior[:i]):
            if earlier.kind == USER:
                break
            if earlier.is_agent and earlier.text and _mentions(earlier.text, call):
                return None
    return 'no explicit user confirmation of this action'


def _enum(rule, trace, step, call):
    argument = rule.params['argument']
    if argument not in call.arguments:
        return None                              # a missing argument is a schema issue, not this rule's
    if call.arguments[argument] in rule.params['values']:
        return None
    return f'{argument}={call.arguments[argument]!r} is not one of {rule.params["values"]}'


def _key(call: ToolCall, keys) -> str:
    return json.dumps([call.arguments.get(k) for k in keys], sort_keys=True, default=str)


def _once(rule, trace, step, call):
    keys = rule.params['key']
    if any(_key(x.call, keys) == _key(call, keys) for x in _successful(trace, step, rule.tools)):
        return f'{call.name} already succeeded for {dict(zip(keys, (call.arguments.get(k) for k in keys)))}'
    return None


def _grounded(rule, trace, step, call):
    index = EvidenceIndex.before(trace, step)
    missing = []
    for name in rule.params['arguments']:
        value = call.arguments.get(name)
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, (str, int)) and not isinstance(v, bool) and str(v) and not index.find_text(str(v)):
                missing.append(f'{name}={v}')
    return f'not found in evidence: {", ".join(missing)}' if missing else None


def _forbidden(rule, trace, step, call):
    return 'this tool is not allowed here'


_CHECKS = {PRIOR: _prior, CONFIRM: _confirm, ENUM: _enum, ONCE: _once, GROUNDED: _grounded, FORBIDDEN: _forbidden}
