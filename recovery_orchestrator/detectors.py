"""Cheap rule-based runtime detectors owned by the orchestrator.

FA is post-hoc and makes LLM calls per step, so it cannot run online. These
checks can run on every new step of a replay: repeated tool calls with no state
change (loops) and tool errors. AgentRx has no "Cycles" category, so loops are
reported under the orchestrator-local ``CYCLE`` label.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .rca import Candidate
from .taxonomy import CYCLE
from .trace import Step, Trace


@dataclass(frozen=True)
class Finding:
    kind: str            # 'loop' | 'tool_error'
    step_index: int      # agent step where the problem starts (a rollback point)
    category: str        # suggested category
    detail: str
    agent: str | None = None

    def as_candidate(self, weight: float) -> Candidate:
        return Candidate(self.step_index, self.agent, self.category, 0, weight,
                         rationale=self.detail, source='detector')


def _tool_results(trace: Trace):
    """(agent step, call, tool step) for each tool call, pairing results in order."""
    pending, pairs = [], []
    for step in trace.steps:
        if step.is_agent and step.tool_calls:
            pending.extend((step, call) for call in step.tool_calls)
        elif step.actor == 'tool' and pending:
            agent_step, call = pending.pop(0)
            pairs.append((agent_step, call, step))
        elif step.actor != 'tool':
            pending.clear()
    return pairs


def is_tool_error(content: str) -> bool:
    text = (content or '').strip()
    if text.lower().startswith(('error', 'exception', 'traceback')):
        return True
    if text.startswith('{'):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return False
        return isinstance(data, dict) and bool(data.get('error'))
    return False


def detect_tool_errors(trace: Trace) -> list[Finding]:
    return [Finding('tool_error', agent_step.index, 'InvalidInvocation',
                    f'{call.name} at step {agent_step.index} returned: {tool.content[:200]}',
                    agent_step.actor)
            for agent_step, call, tool in _tool_results(trace) if is_tool_error(tool.content)]


def detect_loops(trace: Trace, min_repeats: int = 3,
                 write_tools: frozenset[str] = frozenset()) -> list[Finding]:
    """Same call with the same output ``min_repeats`` times and no write call in between,
    or the same agent message repeated ``min_repeats`` times in a row by one actor."""
    findings: list[Finding] = []
    seen: dict[tuple[str, str], list[int]] = {}
    for agent_step, call, tool in _tool_results(trace):
        if call.name in write_tools:
            seen.clear()        # state changed; earlier repeats are no longer a loop
            continue
        steps = seen.setdefault((call.key(), tool.content), [])
        steps.append(agent_step.index)
        if len(steps) == min_repeats:
            findings.append(Finding('loop', steps[1], CYCLE,
                                    f'{call.name} called {min_repeats}x with identical result '
                                    f'(steps {", ".join(map(str, steps))})', agent_step.actor))
    last: dict[str, tuple[str, list[int]]] = {}
    for step in trace.steps:
        if not step.is_agent or step.tool_calls or not step.content.strip():
            continue
        content, run = last.get(step.actor, (None, []))
        run = run + [step.index] if step.content == content else [step.index]
        last[step.actor] = (step.content, run)
        if len(run) == min_repeats:
            findings.append(Finding('loop', run[1], CYCLE,
                                    f'{step.role} repeated the same message {min_repeats}x '
                                    f'(steps {", ".join(map(str, run))})', step.actor))
    return sorted(findings, key=lambda f: f.step_index)


class RuntimeMonitor:
    """Incremental wrapper used during replay: call ``observe`` on every new step;
    a non-empty return value means the rollout should be stopped early."""

    def __init__(self, trajectory_id: str, domain: str, min_repeats: int = 3,
                 write_tools: frozenset[str] = frozenset(), stop_on_tool_error: bool = False):
        self.trace = Trace(trajectory_id, domain, [])
        self.min_repeats = min_repeats
        self.write_tools = write_tools
        self.stop_on_tool_error = stop_on_tool_error
        self._reported: set[tuple[str, int]] = set()

    def observe(self, step: Step) -> list[Finding]:
        self.trace.steps.append(step)
        found = detect_loops(self.trace, self.min_repeats, self.write_tools)
        if self.stop_on_tool_error:
            found += detect_tool_errors(self.trace)
        new = [f for f in found if (f.kind, f.step_index) not in self._reported]
        self._reported.update((f.kind, f.step_index) for f in new)
        return new
