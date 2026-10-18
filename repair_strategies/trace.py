"""Read-only trace model shared by every analyzer and strategy.

Accepts the faultattribution IR (``steps[].substeps[]``), the benchmark's
public traces (same IR, agent turns written as JSON actions, handoffs as a
second substep), OpenAI-style chat messages, and the orchestrator's ``Trace``
object. Step indices are the original 1-based benchmark indices and are never
treated as list offsets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from .taxonomy import normalize_agent

USER, AGENT, TOOL, SYSTEM = 'user', 'agent', 'tool', 'system'
_ROLE_KINDS = {'user': USER, 'human': USER, 'tool': TOOL, 'function': TOOL, 'system': SYSTEM}


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)
    id: str = ''

    def signature(self) -> str:
        """Canonical identity of a call: name plus sorted arguments."""
        return self.name + ':' + json.dumps(self.arguments, sort_keys=True, ensure_ascii=False, default=str)

    def to_dict(self) -> dict:
        return {'id': self.id, 'name': self.name, 'arguments': self.arguments}


@dataclass(frozen=True)
class Event:
    step_index: int
    sub_index: int
    role: str                          # role as written in the trace
    kind: str                          # user | agent | tool | system
    content: str = ''                  # raw content
    text: str = ''                     # natural-language part (a reply, a handoff message)
    tool_calls: tuple[ToolCall, ...] = ()
    tool_name: str | None = None       # set on tool results
    handoff: dict | None = None        # {'to', 'message', 'context'}
    injected: bool = False
    tool_call_id: str = ''             # set on tool results that name the call they answer

    @property
    def actor(self) -> str:
        return normalize_agent(self.role) or ''

    @property
    def is_agent(self) -> bool:
        return self.kind == AGENT and not self.injected


@dataclass(frozen=True)
class ToolExchange:
    """One tool call and the result event that answered it (``None`` if none did)."""
    step_index: int
    actor: str
    call: ToolCall
    result: Event | None


class TraceError(ValueError):
    pass


@dataclass(frozen=True)
class Trace:
    trajectory_id: str
    domain: str
    events: tuple[Event, ...]
    instruction: str = ''

    # ------------------------------------------------------------------ queries
    @property
    def step_indices(self) -> tuple[int, ...]:
        return tuple(sorted({e.step_index for e in self.events}))

    @property
    def last_index(self) -> int:
        return self.events[-1].step_index if self.events else 0

    def has_step(self, index: int) -> bool:
        return any(e.step_index == index for e in self.events)

    def at(self, index: int) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.step_index == index)

    def before(self, index: int) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.step_index < index)

    def from_step(self, index: int) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.step_index >= index)

    def agent_events(self) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.is_agent)

    def user_events(self) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.kind == USER)

    @property
    def agents(self) -> frozenset[str]:
        return frozenset(e.actor for e in self.events if e.is_agent)

    def primary(self, index: int) -> Event | None:
        """The first event of a step (the actor's turn)."""
        events = self.at(index)
        return events[0] if events else None

    def tool_exchanges(self) -> tuple[ToolExchange, ...]:
        """Pair every call with its result: by call id when present, otherwise in order."""
        pending: list[tuple[Event, ToolCall]] = []
        pairs: list[ToolExchange] = []

        def flush():
            for event, call in pending:
                pairs.append(ToolExchange(event.step_index, event.actor, call, None))
            pending.clear()

        for event in self.events:
            if event.kind == TOOL:
                if not pending:
                    continue
                match = next((i for i, (_, c) in enumerate(pending)
                              if c.id and c.id == event.tool_call_id), None)
                if match is None:
                    match = next((i for i, (_, c) in enumerate(pending)
                                  if event.tool_name and c.name == event.tool_name), 0)
                caller, call = pending.pop(match)
                pairs.append(ToolExchange(caller.step_index, caller.actor, call, event))
            elif event.tool_calls:
                flush()
                pending.extend((event, call) for call in event.tool_calls)
            elif event.kind != SYSTEM:
                flush()
        flush()
        return tuple(sorted(pairs, key=lambda p: p.step_index))

    def last_user_step_before(self, index: int) -> int | None:
        steps = [e.step_index for e in self.events if e.kind == USER and e.step_index < index]
        return steps[-1] if steps else None

    def next_agent_step(self, index: int) -> int | None:
        steps = [e.step_index for e in self.events if e.is_agent and e.step_index >= index]
        return steps[0] if steps else None

    def previous_agent_step(self, index: int) -> int | None:
        steps = [e.step_index for e in self.events if e.is_agent and e.step_index <= index]
        return steps[-1] if steps else None

    def rollback_point(self, index: int) -> int | None:
        """The agent step to regenerate so that the event at ``index`` is replaced.

        An agent step is its own rollback point. A tool result rolls back to the
        agent step that issued the call. A user or system turn rolls back to the
        agent's next turn (where it should have reacted differently).
        """
        event = self.primary(index)
        if event is None:
            return None
        if event.is_agent:
            return index
        if event.kind == TOOL:
            return self.previous_agent_step(index)
        return self.next_agent_step(index)

    def final_reply(self) -> Event | None:
        """The last agent message addressed to the user (not a tool call or handoff)."""
        for event in reversed(self.events):
            if event.is_agent and not event.tool_calls and event.handoff is None and event.text.strip():
                return event
        return None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_ir(cls, ir: dict) -> 'Trace':
        if not isinstance(ir, dict) or 'steps' not in ir:
            raise TraceError('IR trace needs a "steps" list')
        events: list[Event] = []
        seen_steps: set[int] = set()
        for step in ir['steps']:
            index = _as_int(step.get('index'), 'step index')
            if index in seen_steps:
                raise TraceError(f'Duplicate step index {index}')
            seen_steps.add(index)
            substeps = step.get('substeps') or []
            if not substeps:
                raise TraceError(f'Step {index} has no substeps')
            step_events = [_event(index, _as_int(sub.get('sub_index', i), 'substep index'), sub)
                           for i, sub in enumerate(substeps, start=1)]
            if any(e.tool_name == 'handoff' for e in step_events):
                # The benchmark writes a handoff twice: as the agent's action and as a
                # dedicated 'handoff' substep. Keep the dedicated one only.
                step_events = [replace(e, handoff=None, text='') if e.handoff and e.tool_name != 'handoff' else e
                               for e in step_events]
            events.extend(step_events)
        events.sort(key=lambda e: (e.step_index, e.sub_index))
        return cls(str(ir.get('trajectory_id', '')), str(ir.get('domain', '')), tuple(events),
                   str(ir.get('instruction', '') or ''))

    @classmethod
    def from_messages(cls, messages: list[dict], trajectory_id='', domain='', instruction='') -> 'Trace':
        steps = []
        for i, message in enumerate(messages, start=1):
            sub = {k: message[k] for k in ('role', 'content', 'tool_calls', 'name', 'tool_call_id', 'injected')
                   if k in message}
            steps.append({'index': i, 'substeps': [{'sub_index': 1, **sub}]})
        return cls.from_ir({'trajectory_id': trajectory_id, 'domain': domain, 'instruction': instruction,
                            'steps': steps})

    @classmethod
    def from_object(cls, trace) -> 'Trace':
        """Duck-typed conversion of an object with ``steps`` of ``role``/``content``/``tool_calls``."""
        steps = []
        for s in trace.steps:
            calls = [{'id': getattr(c, 'id', ''), 'name': c.name, 'arguments': dict(c.arguments)}
                     for c in getattr(s, 'tool_calls', ()) or ()]
            steps.append({'index': s.index, 'substeps': [{
                'sub_index': 1, 'role': s.role, 'content': s.content or '', 'tool_calls': calls,
                'name': getattr(s, 'name', None), 'injected': bool(getattr(s, 'injected', False))}]})
        return cls.from_ir({'trajectory_id': getattr(trace, 'trajectory_id', ''),
                            'domain': getattr(trace, 'domain', ''),
                            'instruction': getattr(trace, 'instruction', ''), 'steps': steps})

    @classmethod
    def from_any(cls, value) -> 'Trace':
        if isinstance(value, Trace):
            return value
        if isinstance(value, dict):
            if 'steps' in value:
                return cls.from_ir(value)
            if 'trajectory' in value:          # an FA case: {'trajectory': IR, 'gold': ...}
                return cls.from_ir(value['trajectory'])
            if 'traj' in value:                # a raw tau-bench record
                return cls.from_messages(value['traj'], value.get('task_id', ''), 'tau')
        if isinstance(value, list):
            return cls.from_messages(value)
        if hasattr(value, 'steps'):
            return cls.from_object(value)
        raise TraceError(f'Cannot read a trace from {type(value).__name__}')

    @classmethod
    def load(cls, path) -> 'Trace':
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if isinstance(data, list) and data and isinstance(data[0], dict) and 'steps' in data[0]:
            if len(data) != 1:
                raise TraceError(f'{path} holds {len(data)} traces; pass one')
            data = data[0]
        return cls.from_any(data)

    def to_ir(self) -> dict:
        steps: dict[int, list] = {}
        for e in self.events:
            sub = {'sub_index': e.sub_index, 'role': e.role, 'content': e.content}
            if e.tool_calls:
                sub['tool_calls'] = [c.to_dict() for c in e.tool_calls]
            if e.tool_name:
                sub['name'] = e.tool_name
            if e.injected:
                sub['injected'] = True
            if e.tool_call_id:
                sub['tool_call_id'] = e.tool_call_id
            steps.setdefault(e.step_index, []).append(sub)
        return {'trajectory_id': self.trajectory_id, 'domain': self.domain, 'instruction': self.instruction,
                'steps': [{'index': i, 'substeps': subs} for i, subs in sorted(steps.items())]}


# ---------------------------------------------------------------------- parsing helpers
def _as_int(value, what) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str) and value.strip().lstrip('-').isdigit():
        return int(value)
    raise TraceError(f'Invalid {what}: {value!r}')


def parse_arguments(raw) -> dict:
    """Tool arguments as a dict; malformed JSON is preserved under ``__raw__`` as evidence."""
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == '':
        return {}
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {'__raw__': raw}
        return value if isinstance(value, dict) else {'__raw__': raw}
    return {'__raw__': str(raw)}


def _tool_calls(raw) -> tuple[ToolCall, ...]:
    calls = []
    for item in raw or ():
        if not isinstance(item, dict):
            continue
        fn = item.get('function') if isinstance(item.get('function'), dict) else item
        name = fn.get('name')
        if not isinstance(name, str) or not name:
            continue
        calls.append(ToolCall(name, parse_arguments(fn.get('arguments')), str(item.get('id', '') or '')))
    return tuple(calls)


def parse_action(content: str) -> dict | None:
    """A benchmark-style JSON action ({"action": "tool"|"handoff"|"reply", ...}), or None."""
    text = (content or '').strip()
    if not text.startswith('{') or '"action"' not in text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and isinstance(value.get('action'), str) else None


def _event(index: int, sub_index: int, sub: dict) -> Event:
    role = str(sub.get('role') or '')
    if not role:
        raise TraceError(f'Step {index}.{sub_index} has no role')
    kind = _ROLE_KINDS.get(role.lower(), AGENT)
    content = sub.get('content')
    if not isinstance(content, str):
        content = '' if content is None else json.dumps(content, ensure_ascii=False, default=str)
    name = sub.get('name')
    calls = _tool_calls(sub.get('tool_calls'))
    text, handoff = content, None
    if kind == AGENT and name == 'handoff':
        data = _json_object(content)
        handoff = {'to': str(data.get('handoff_to', data.get('to', ''))), 'message': str(data.get('message', '')),
                   'context': data.get('context') if isinstance(data.get('context'), dict) else {}}
        text = handoff['message']
    elif kind == AGENT:
        action = parse_action(content)
        if action is not None:
            verb = action['action']
            if verb == 'reply':
                text = _string(action.get('content'))
            elif verb == 'handoff':
                handoff = {'to': str(action.get('to', '')), 'message': _string(action.get('message')),
                           'context': action.get('context') if isinstance(action.get('context'), dict) else {}}
                text = handoff['message']
            elif verb == 'tool':
                text = ''
                if not calls and isinstance(action.get('tool'), str):
                    calls = (ToolCall(action['tool'], parse_arguments(action.get('arguments')), f'call-{index}'),)
    return Event(index, sub_index, role, kind, content, text, calls,
                 str(name) if name and (kind == TOOL or name == 'handoff') else None,
                 handoff, bool(sub.get('injected', False)), str(sub.get('tool_call_id') or ''))


def _string(value) -> str:
    if value is None:
        return ''
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _json_object(text: str) -> dict:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}
