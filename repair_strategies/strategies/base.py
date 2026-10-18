"""Strategy interface, the context every strategy receives, and shared helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property

from ..context import AttemptRecord, Hypothesis, TaskSpec
from ..evidence import claims
from ..evidence.tool_output import is_error
from ..llm import LLMClient
from ..memory.lessons import LessonMemory
from ..memory.tool_outcomes import ToolOutcomeMemory
from ..plan import AcceptanceCheck, Operation, RepairPlan
from ..policy.plan_state import Obligation, confirmed_obligations
from ..policy.rules import RuleGate
from ..taxonomy import FaultType
from ..trace import Event, ToolExchange, Trace

REGISTRY_KINDS = ('reasoning_repair', 'retry', 'clarify', 'loop_break')


@dataclass(frozen=True)
class StrategySpec:
    id: str
    name: str
    categories: tuple[str, ...]
    rung: int                              # position in the per-fault ladder (1 = first choice)
    kind: str                              # orchestrator registry kind
    description: str
    tokens: int = 600                      # estimated extra tokens of the repair itself (not the replay)
    latency_s: float = 0.0
    prior_success: float = 0.5             # P(verified recovery | correct attribution); tuned by evaluation
    subtypes: tuple[str | None, ...] | None = None   # None: any subtype
    reversible: bool = True
    requires_llm: bool = False

    def __post_init__(self):
        if self.kind not in REGISTRY_KINDS:
            raise ValueError(f'{self.id}: kind must be one of {REGISTRY_KINDS}')
        if self.rung < 1:
            raise ValueError(f'{self.id}: rung starts at 1')
        if not 0.0 <= self.prior_success <= 1.0:
            raise ValueError(f'{self.id}: prior_success must be in [0, 1]')

    def matches(self, fault: FaultType) -> bool:
        return fault.category in self.categories and (self.subtypes is None or fault.subtype in self.subtypes)


@dataclass(frozen=True)
class Skip:
    """The strategy does not apply to this run, and why."""
    reason: str


@dataclass
class RepairContext:
    trace: Trace
    hypothesis: Hypothesis
    task: TaskSpec
    history: tuple[AttemptRecord, ...] = ()
    tool_memory: ToolOutcomeMemory | None = None
    lessons: LessonMemory | None = None
    llm: LLMClient | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def step(self) -> int | None:
        return self.hypothesis.step_index

    @cached_property
    def gate(self) -> RuleGate:
        return RuleGate.for_task(self.task)

    @cached_property
    def exchanges(self) -> tuple[ToolExchange, ...]:
        return self.trace.tool_exchanges()

    @cached_property
    def obligations(self) -> tuple[Obligation, ...]:
        return confirmed_obligations(self.trace, self.task)

    def attempts(self, strategy_id: str) -> int:
        return sum(1 for r in self.history if r.strategy_id == strategy_id)

    def agent_step(self) -> int | None:
        """The agent step to regenerate for the attributed event."""
        if self.step is None or not self.trace.has_step(self.step):
            return None
        return self.trace.rollback_point(self.step)

    def provenance(self, step: int) -> claims.ProvenanceReport:
        return claims.provenance(self.trace, step, synonyms=self.task.fact_keywords)

    def turn_steps(self, step: int) -> list[int]:
        """Agent steps from ``step`` up to (not including) the next user turn."""
        out = []
        for event in self.trace.from_step(step):
            if event.kind == 'user' and event.step_index > step:
                break
            if event.is_agent and event.step_index not in out:
                out.append(event.step_index)
        return out

    def consumed_exchange(self, step: int, requery: bool = False) -> ToolExchange | None:
        """The latest answered call before ``step``, preferring the same actor.

        ``requery=True`` excludes orphan results, which cannot be re-executed.
        """
        event = self.trace.primary(step)
        actor = event.actor if event is not None else None
        answered = [x for x in self.exchanges if x.result is not None and x.result.step_index < step]
        if requery:
            answered = [x for x in answered if not x.orphan]
        same = [x for x in answered if x.actor == actor]
        return (same or answered or [None])[-1]


class Strategy:
    spec: StrategySpec

    def plan(self, ctx: RepairContext) -> RepairPlan | Skip:
        raise NotImplementedError

    def build(self, ctx: RepairContext, rollback_step: int | None, operations, acceptance=(), rationale: str = '',
              evidence=(), confidence_factor: float = 1.0) -> RepairPlan:
        expected = max(0.0, min(1.0, self.spec.prior_success * ctx.hypothesis.confidence * confidence_factor))
        return RepairPlan(strategy_id=self.spec.id, category=ctx.hypothesis.category, subtype=ctx.hypothesis.subtype,
                          rollback_step=rollback_step, operations=tuple(operations), acceptance=tuple(acceptance),
                          rationale=rationale, estimated_tokens=self.spec.tokens, expected_success=round(expected, 4),
                          evidence=tuple(evidence), hypothesis=ctx.hypothesis.to_dict())


# ---------------------------------------------------------------------- shared helpers
def check(kind: str, **params) -> AcceptanceCheck:
    return AcceptanceCheck(kind, params)


def compact(value, limit: int = 400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + ' ...'


def record(source: str, lines) -> str:
    """Body of a verified record delivered under an external role."""
    body = '\n'.join(f'- {line}' for line in lines)
    return f'Verified record from {source} (computed from tool outputs, authoritative):\n{body}'


def rule_specs(task: TaskSpec, kinds=None, tools=None) -> tuple[dict, ...]:
    """Task rules filtered by kind and by the tools they govern, as plain specs for GuardCalls."""
    out = []
    for spec in task.rules:
        governed = spec.get('tools', [spec.get('tool')])
        governed = [governed] if isinstance(governed, str) else list(governed or ())
        if kinds is not None and spec.get('kind') not in kinds:
            continue
        if tools is not None and not set(governed) & set(tools) and '@write' not in governed:
            continue
        out.append(dict(spec))
    return tuple(out)


def failed_exchange(exchanges, step: int | None) -> ToolExchange | None:
    """The call that failed at ``step`` (call or result step), else the latest failure before it."""
    errors = [x for x in exchanges if not x.orphan and x.result is not None and is_error(x.result.content)]
    if step is not None:
        for x in errors:
            if step in (x.step_index, x.result.step_index):
                return x
        errors = [x for x in errors if x.step_index <= step]
    return errors[-1] if errors else None


def calls_at(trace: Trace, step: int) -> list[tuple[Event, object]]:
    return [(e, c) for e in trace.at(step) if e.is_agent for c in e.tool_calls]


def operations_tools(operations: tuple[Operation, ...]) -> set[str]:
    tools = set()
    for op in operations:
        for name in ('tool', 'from_tool', 'to_tool', 'name'):
            value = getattr(op, name, None)
            if isinstance(value, str) and value:
                tools.add(value)
    return tools
