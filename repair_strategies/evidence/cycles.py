"""Loop detection: repeated calls, handoffs or messages with no state change in between."""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..trace import Trace
from .tool_output import is_error

CALL, HANDOFF, MESSAGE = 'call', 'handoff', 'message'


@dataclass(frozen=True)
class Loop:
    kind: str
    actor: str
    signature: str
    steps: tuple[int, ...]           # every occurrence, in order

    @property
    def start(self) -> int:
        return self.steps[0]

    @property
    def first_repeat(self) -> int:
        """The first redundant step: where the agent should have changed course."""
        return self.steps[1]

    def describe(self) -> str:
        what = {CALL: 'the same tool call with the same result', HANDOFF: 'the same handoff',
                MESSAGE: 'the same message'}[self.kind]
        return f'{self.actor} repeated {what} {len(self.steps)} times (steps {", ".join(map(str, self.steps))})'


def detect(trace: Trace, write_tools=frozenset(), min_repeats: int = 3) -> tuple[Loop, ...]:
    """Loops of ``min_repeats`` or more. A successful write in between resets the count."""
    if min_repeats < 2:
        raise ValueError('min_repeats must be at least 2')
    found: list[Loop] = []
    calls: dict[tuple, list[int]] = {}

    def close_calls():
        for (actor, signature, _), steps in calls.items():
            if len(steps) >= min_repeats:
                found.append(Loop(CALL, actor, signature, tuple(steps)))
        calls.clear()

    for x in trace.tool_exchanges():
        if x.orphan:
            continue                          # no call was logged, so nothing was repeated by the agent
        if x.call.name in write_tools and x.result is not None and not is_error(x.result.content):
            close_calls()
            continue
        result = x.result.content if x.result is not None else ''
        calls.setdefault((x.actor, x.call.signature(), result), []).append(x.step_index)

    close_calls()

    handoffs: dict[tuple, list[int]] = {}

    def close_handoffs():
        for (actor, to, message, _), steps in handoffs.items():
            if len(steps) >= min_repeats:
                found.append(Loop(HANDOFF, actor, f'handoff:{to}:{message[:80]}', tuple(steps)))
        handoffs.clear()

    runs: list[tuple[str, str, list[int]]] = []          # consecutive identical messages per actor
    current: dict[str, tuple[str, str, list[int]]] = {}
    writes = {x.step_index for x in trace.tool_exchanges() if x.call.name in write_tools and not x.orphan
              and x.result is not None and not is_error(x.result.content)}
    for event in trace.events:
        if not event.is_agent:
            continue
        if event.step_index in writes:
            close_handoffs()
            current.clear()
        if event.handoff:
            key = (event.actor, event.handoff['to'], event.handoff['message'],
                   json.dumps(event.handoff.get('context') or {}, sort_keys=True))
            handoffs.setdefault(key, []).append(event.step_index)
        elif event.text.strip() and not event.tool_calls:
            run = current.get(event.actor)
            if run is not None and run[1] == event.text:
                run[2].append(event.step_index)
            else:
                run = (event.actor, event.text, [event.step_index])
                current[event.actor] = run
                runs.append(run)

    for actor, text, steps in runs:
        if len(steps) >= min_repeats:
            found.append(Loop(MESSAGE, actor, text[:120], tuple(steps)))
    close_handoffs()
    return tuple(sorted(found, key=lambda loop: (loop.start, loop.kind)))


def loop_at(trace: Trace, step_index: int | None, write_tools=frozenset(), min_repeats: int = 3) -> Loop | None:
    """The loop containing ``step_index`` (or the earliest loop when no step is given)."""
    loops = detect(trace, write_tools, min_repeats)
    if step_index is not None:
        for loop in loops:
            if loop.steps[0] <= step_index <= loop.steps[-1]:
                return loop
    return loops[0] if loops else None
