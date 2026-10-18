"""The repair engine: pick the next rung of a fault's ladder, attach lessons, learn from outcomes."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

from .context import FAILED, REJECTED, VERIFIED, AttemptRecord, Hypothesis, TaskSpec
from .llm import LLMClient
from .memory.lessons import LessonMemory
from .memory.tool_outcomes import ToolOutcomeMemory
from .plan import InjectMessage, RepairPlan
from .strategies import Skip, Strategy, default_strategies
from .strategies.base import RepairContext, operations_tools
from .trace import Trace

PLANNED, SKIPPED, TRIED, ERROR = 'planned', 'skipped', 'tried', 'error'


@dataclass(frozen=True)
class Candidate:
    """How one rung of the ladder fared for a given run."""
    strategy_id: str
    rung: int
    status: str                       # planned | skipped | tried | error
    reason: str = ''
    plan: RepairPlan | None = None

    def to_dict(self) -> dict:
        return {'strategy_id': self.strategy_id, 'rung': self.rung, 'status': self.status, 'reason': self.reason,
                'plan': self.plan.to_dict() if self.plan else None}


class RepairEngine:
    def __init__(self, task: TaskSpec, strategies: tuple[Strategy, ...] | None = None,
                 tool_memory: ToolOutcomeMemory | None = None, lessons: LessonMemory | None = None,
                 llm: LLMClient | None = None, max_lessons: int = 2):
        self.task = task
        self.strategies = tuple(strategies) if strategies is not None else default_strategies()
        ids = [s.spec.id for s in self.strategies]
        if len(ids) != len(set(ids)):
            raise ValueError('strategy ids must be unique')
        self.tool_memory, self.lessons, self.llm = tool_memory, lessons, llm
        self.max_lessons = max_lessons

    # ------------------------------------------------------------------ selection
    def ladder(self, hypothesis: Hypothesis) -> list[Strategy]:
        found = [s for s in self.strategies if s.spec.matches(hypothesis.fault)]
        return sorted(found, key=lambda s: (s.spec.rung, s.spec.id))

    def strategy(self, strategy_id: str) -> Strategy:
        for s in self.strategies:
            if s.spec.id == strategy_id:
                return s
        raise KeyError(strategy_id)

    def context(self, trace: Trace, hypothesis: Hypothesis, history=()) -> RepairContext:
        return RepairContext(trace, hypothesis, self.task, tuple(history), self.tool_memory, self.lessons, self.llm)

    def _walk(self, trace: Trace, hypothesis: Hypothesis, history) -> Iterator[Candidate]:
        tried = {r.strategy_id for r in history if r.outcome in (FAILED, REJECTED)}
        ctx = self.context(trace, hypothesis, history)
        for strategy in self.ladder(hypothesis):
            spec = strategy.spec
            if spec.id in tried:
                yield Candidate(spec.id, spec.rung, TRIED, 'already tried on this run')
                continue
            if spec.requires_llm and self.llm is None:
                yield Candidate(spec.id, spec.rung, SKIPPED, 'needs an LLM client')
                continue
            try:
                result = strategy.plan(ctx)
            except Exception as exc:          # one faulty strategy must not stop the ladder
                yield Candidate(spec.id, spec.rung, ERROR, f'{type(exc).__name__}: {exc}')
                continue
            if isinstance(result, Skip):
                yield Candidate(spec.id, spec.rung, SKIPPED, result.reason)
            else:
                yield Candidate(spec.id, spec.rung, PLANNED, result.rationale, self._with_lessons(result, trace))

    def evaluate(self, trace: Trace, hypothesis: Hypothesis, history=()) -> list[Candidate]:
        """Every rung of the ladder with its outcome (for inspection and logging)."""
        return list(self._walk(trace, hypothesis, tuple(history)))

    def propose(self, trace: Trace, hypothesis: Hypothesis, history=()) -> RepairPlan | None:
        """The next plan to try, or None when the ladder is exhausted or the fault is already repaired."""
        history = tuple(history)
        if any(r.outcome == VERIFIED and r.category == hypothesis.category for r in history):
            return None
        for candidate in self._walk(trace, hypothesis, history):
            if candidate.status == PLANNED:
                return candidate.plan
        return None

    def plan_for(self, strategy_id: str, trace: Trace, hypothesis: Hypothesis, history=()) -> RepairPlan | Skip:
        """A plan from one named strategy (used when the orchestrator already chose the strategy)."""
        strategy = self.strategy(strategy_id)
        if not strategy.spec.matches(hypothesis.fault):
            return Skip(f'{strategy_id} does not target {hypothesis.category}')
        if strategy.spec.requires_llm and self.llm is None:
            return Skip('needs an LLM client')
        result = strategy.plan(self.context(trace, hypothesis, history))
        return result if isinstance(result, Skip) else self._with_lessons(result, trace)

    # ------------------------------------------------------------------ lessons
    def _with_lessons(self, plan: RepairPlan, trace: Trace) -> RepairPlan:
        if self.lessons is None or plan.terminal or self.max_lessons < 1:
            return plan
        tools = operations_tools(plan.operations)
        step = plan.hypothesis.get('step_index')
        if step is not None:
            tools |= {c.name for e in trace.at(step) for c in e.tool_calls}
        agent = plan.hypothesis.get('agent')
        found = self.lessons.retrieve(plan.category, tools, [agent] if agent else (), self.max_lessons)
        if not found:
            return plan
        text = 'Lessons from earlier verified repairs of this kind:\n' + '\n'.join(f'- {l.text}' for l in found)
        return replace(plan, operations=plan.operations + (InjectMessage('system', text),),
                       lessons=tuple(l.lesson_id for l in found))

    # ------------------------------------------------------------------ feedback
    def record_outcome(self, plan: RepairPlan, verified: bool, rollout: Trace | None = None) -> AttemptRecord:
        """Update memories with what happened to ``plan``. Lessons are written only for verified repairs."""
        if self.tool_memory is not None and rollout is not None:
            self.tool_memory.observe(rollout, key=f'{rollout.domain}/{rollout.trajectory_id}#{plan.plan_id}')
        if self.lessons is not None:
            learned = None
            if verified and plan.rationale and not plan.terminal:
                agent = plan.hypothesis.get('agent')
                learned = self.lessons.add_verified(plan.category, plan.rationale, plan.strategy_id,
                                                    tools=operations_tools(plan.operations),
                                                    agents=[agent] if agent else ()).lesson_id
            for lesson_id in plan.lessons:
                if lesson_id == learned:
                    continue                      # already counted by add_verified
                if verified:
                    self.lessons.record_success(lesson_id)
                else:
                    self.lessons.record_failure(lesson_id)
        return AttemptRecord(plan.strategy_id, plan.category, plan.rollback_step, VERIFIED if verified else FAILED)
