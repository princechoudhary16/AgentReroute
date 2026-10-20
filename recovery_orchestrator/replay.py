"""Replay interfaces: task views, repair directives, rollouts, and the score ledger.

Leakage boundary: an environment may know the ground truth (tau-bench computes
reward from gold actions), but it writes rewards only to its ``ScoreLedger``.
``Rollout`` objects, which the orchestrator and verifier see, carry no reward.
Only ``evaluation`` reads the ledger.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Protocol

from .detectors import Finding, RuntimeMonitor
from .rca import Candidate
from .registry import Strategy
from .trace import Prefix, Step, Trace


@dataclass(frozen=True)
class TaskView:
    """What the decision and repair path may know about a task. No gold, no reward."""
    trajectory_id: str
    domain: str
    instruction: str = ''
    policy: str = ''


@dataclass(frozen=True)
class RepairDirective:
    kind: str                 # 'inject_guidance' | 'retry' | 'none'
    strategy_id: str
    message: str = ''

    def as_step(self, index: int) -> Step:
        return Step(index=index, role='system', content=self.message, injected=True)


UNMODIFIED = RepairDirective('none', 'unmodified_replay')


class Repairer(Protocol):
    """Repair-team interface. A repairer may also define
    ``record_outcome(trajectory_id, verified, rollout)``; the orchestrator then calls it
    after every verified (or failed) attempt so the repairer can learn. It receives the
    verifier's verdict only, never the ground-truth reward."""

    def build(self, strategy: Strategy, candidate: Candidate, trace: Trace) -> RepairDirective: ...


class TemplateRepairer:
    """Mock Repair-team implementation: fills the strategy's guidance template."""

    def build(self, strategy: Strategy, candidate: Candidate, trace: Trace) -> RepairDirective:
        if strategy.kind == 'retry' or not strategy.template:
            return RepairDirective('retry', strategy.id)
        message = strategy.template.format_map(_SafeDict(
            rationale=candidate.rationale.strip() or 'no rationale given',
            step_index=candidate.step_index, category=candidate.category,
            agent=candidate.agent or ''))
        return RepairDirective('inject_guidance', strategy.id, message)


class _SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


@dataclass
class Rollout:
    rollout_id: str
    trajectory_id: str
    rollback_step: int
    strategy_id: str
    steps: list[Step]                       # prefix + injected guidance + regenerated steps;
                                            # indices are unique and increasing (environments renumber)
    tokens: int = 0
    seconds: float = 0.0
    stopped_early: str | None = None        # e.g. 'loop' when the runtime monitor fired
    findings: list[Finding] = field(default_factory=list)

    def as_trace(self, domain: str, instruction: str = '') -> Trace:
        return Trace(self.trajectory_id, domain, list(self.steps), instruction,
                     {'rollout_id': self.rollout_id})


class ScoreLedger:
    """Ground-truth outcome per rollout, written by the environment, read by evaluation."""

    def __init__(self):
        self._rewards: dict[str, float] = {}

    def record(self, rollout_id: str, reward: float) -> None:
        self._rewards[rollout_id] = float(reward)

    def reward(self, rollout_id: str) -> float | None:
        return self._rewards.get(rollout_id)

    def __len__(self):
        return len(self._rewards)


class ReplayEnvironment(Protocol):
    ledger: ScoreLedger
    write_tools: frozenset[str]

    def task_view(self, trajectory_id: str) -> TaskView: ...

    def rollout(self, trace: Trace, prefix: Prefix, directive: RepairDirective,
                monitor: RuntimeMonitor | None, max_tokens: int) -> Rollout: ...


_ids = itertools.count(1)


def new_rollout_id(trajectory_id: str) -> str:
    return f'{trajectory_id}#r{next(_ids)}'


class ScriptedEnvironment:
    """Deterministic offline environment for tests and dry runs.

    ``script(trace, prefix, directive) -> (new_steps, reward, tokens)`` decides what
    the replayed agent does after the rollback point and the ground-truth reward.
    """

    def __init__(self, tasks: dict[str, TaskView], script, write_tools: frozenset[str] = frozenset()):
        self.tasks = tasks
        self.script = script
        self.write_tools = write_tools
        self.ledger = ScoreLedger()
        self.calls: list[dict] = []

    def task_view(self, trajectory_id: str) -> TaskView:
        return self.tasks[trajectory_id]

    def rollout(self, trace, prefix, directive, monitor, max_tokens):
        self.calls.append({'rollback_step': prefix.rollback_step, 'directive': directive})
        new_steps, reward, tokens = self.script(trace, prefix, directive)
        steps = list(prefix.steps)
        if directive.kind == 'inject_guidance':
            steps.append(directive.as_step(prefix.rollback_step))
        rollout = Rollout(new_rollout_id(trace.trajectory_id), trace.trajectory_id,
                          prefix.rollback_step, directive.strategy_id, steps,
                          tokens=min(tokens, max_tokens))
        if monitor is not None:
            for s in prefix.steps:
                monitor.observe(s)
        for step in new_steps:
            # Injected guidance occupies its own index; regenerated steps follow it.
            step = replace(step, index=len(steps) + 1) if steps and step.index <= steps[-1].index else step
            steps.append(step)
            if monitor is not None:
                found = monitor.observe(step)
                if found:
                    rollout.findings.extend(found)
                    rollout.stopped_early = found[0].kind
                    reward = 0.0
                    break
        if tokens > max_tokens:
            rollout.stopped_early = rollout.stopped_early or 'tokens'
            reward = 0.0
        self.ledger.record(rollout.rollout_id, reward)
        return rollout
