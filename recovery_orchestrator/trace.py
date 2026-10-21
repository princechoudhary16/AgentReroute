"""Trace model, rollback points, and prefixes for replay.

A ``Trace`` is built from FA's ``trajectory_ir.json`` (or any list of
chat messages). It never holds gold labels or rewards.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .taxonomy import normalize_agent

NON_AGENT_ROLES = frozenset({'user', 'human', 'tool', 'system'})


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    id: str = ''

    def key(self) -> str:
        return self.name + ':' + json.dumps(self.arguments, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class Step:
    index: int                     # FA step_index (1-based)
    role: str
    content: str = ''
    tool_calls: tuple[ToolCall, ...] = ()
    name: str | None = None        # tool name, for tool-result steps
    injected: bool = False         # True for orchestrator-injected repair guidance

    @property
    def actor(self) -> str:
        return normalize_agent(self.role) or ''

    @property
    def is_agent(self) -> bool:
        return self.actor not in NON_AGENT_ROLES and not self.injected


@dataclass
class Trace:
    trajectory_id: str
    domain: str
    steps: list[Step]
    instruction: str = ''          # the task statement as given to the user / user simulator
    metadata: dict = field(default_factory=dict)

    def step(self, index: int) -> Step:
        for s in self.steps:
            if s.index == index:
                return s
        raise KeyError(f'step {index} not in trace {self.trajectory_id}')

    def has_step(self, index: int) -> bool:
        return any(s.index == index for s in self.steps)

    @property
    def last_index(self) -> int:
        return self.steps[-1].index if self.steps else 0

    @classmethod
    def from_ir(cls, ir: dict) -> 'Trace':
        steps = []
        for step in ir['steps']:
            # FA IR has one substep per step for tau and Magentic-One.
            for sub in step['substeps'][:1]:
                calls = tuple(ToolCall(c.get('name', ''), c.get('arguments') or {}, str(c.get('id', '')))
                              for c in sub.get('tool_calls', []) or [])
                steps.append(Step(index=step['index'], role=sub['role'], content=sub.get('content') or '',
                                  tool_calls=calls, name=sub.get('name')))
        return cls(trajectory_id=str(ir['trajectory_id']), domain=ir.get('domain', ''), steps=steps,
                   instruction=ir.get('instruction', ''))

    @classmethod
    def load_ir(cls, path) -> 'Trace':
        return cls.from_ir(json.loads(Path(path).read_text(encoding='utf-8-sig')))

    @classmethod
    def from_messages(cls, trajectory_id, domain: str, messages: list[dict],
                      instruction: str = '') -> 'Trace':
        """OpenAI-style chat messages (tau-bench ``traj``) -> Trace, 1-based indices."""
        steps = []
        for i, m in enumerate(messages, start=1):
            calls = []
            for c in m.get('tool_calls') or []:
                fn = c.get('function', c)
                args = fn.get('arguments') or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {'__raw__': args}
                calls.append(ToolCall(fn.get('name', ''), args, str(c.get('id', ''))))
            steps.append(Step(index=i, role=m['role'], content=m.get('content') or '',
                              tool_calls=tuple(calls), name=m.get('name'),
                              injected=bool(m.get('injected'))))
        return cls(str(trajectory_id), domain, steps, instruction)


def rollback_point(trace: Trace, step_index: int) -> int | None:
    """Agent step to regenerate so that ``step_index`` is replaced.

    * agent step: itself;
    * tool result: the agent step that issued the call (nearest earlier agent step);
    * user/system step: the first agent step after it (where the agent should have
      reacted differently, e.g. by asking for clarification); if the trace ends
      before any agent reply (e.g. a run cut off by a system failure), the resume
      point ``last_index + 1``.
    Returns None when no such agent step exists.
    """
    if not trace.has_step(step_index):
        return None
    step = trace.step(step_index)
    if step.is_agent:
        return step_index
    if step.actor == 'tool':
        earlier = [s.index for s in trace.steps if s.index < step_index and s.is_agent]
        return earlier[-1] if earlier else None
    later = [s.index for s in trace.steps if s.index > step_index and s.is_agent]
    if later:
        return later[0]
    return trace.last_index + 1 if step.actor in ('user', 'human') else None


@dataclass(frozen=True)
class Prefix:
    """Everything before a rollback point, ready for replay."""
    trajectory_id: str
    rollback_step: int
    steps: tuple[Step, ...]                    # kept steps, index < rollback_step
    tool_calls: tuple[ToolCall, ...]           # calls to re-execute to rebuild env state
    irreversible_in_prefix: tuple[int, ...]    # irreversible steps that are re-executed
    irreversible_downstream: tuple[int, ...]   # irreversible steps the rollback would undo


def irreversible_steps(trace: Trace, write_tools: frozenset[str]) -> list[int]:
    return [s.index for s in trace.steps if any(c.name in write_tools for c in s.tool_calls)]


def build_prefix(trace: Trace, rollback_step: int, write_tools: frozenset[str] = frozenset()) -> Prefix:
    """Steps before ``rollback_step``; ``last_index + 1`` keeps the whole trace (resume)."""
    if not trace.has_step(rollback_step) and rollback_step != trace.last_index + 1:
        raise KeyError(f'rollback step {rollback_step} not in trace')
    kept = tuple(s for s in trace.steps if s.index < rollback_step)
    calls = tuple(c for s in kept for c in s.tool_calls)
    irreversible = irreversible_steps(trace, write_tools)
    return Prefix(trace.trajectory_id, rollback_step, kept, calls,
                  tuple(i for i in irreversible if i < rollback_step),
                  tuple(i for i in irreversible if i >= rollback_step))
