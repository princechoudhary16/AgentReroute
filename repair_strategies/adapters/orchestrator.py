"""Integration with the recovery orchestrator (branch ``orchestrator``), by data contract only.

* ``export_registry`` writes the strategy registry in the orchestrator's schema
  (``recovery_orchestrator/registry.py``, schema_version 1).
* ``OrchestratorRepairer`` implements its ``Repairer.build(strategy, candidate, trace)``
  protocol and returns its ``RepairDirective`` shape. Pass
  ``directive_factory=recovery_orchestrator.replay.RepairDirective`` when wiring it in.

The orchestrator calls ``build`` again for a trajectory only after the previous
repair failed verification, so each repeated call records the previous plan as
failed and the next rung of the ladder is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..context import FAILED, AttemptRecord, Hypothesis
from ..engine import RepairEngine
from ..lowering import GUIDANCE, NONE, ORCHESTRATOR, RESAMPLE, RETRY, TERMINAL, Lowered, lower
from ..plan import RepairPlan
from ..strategies import Skip, default_strategies
from ..taxonomy import INCONCLUSIVE
from ..trace import Trace

SCHEMA_VERSION = 1
_KINDS = {GUIDANCE: 'inject_guidance', RETRY: 'retry', RESAMPLE: 'retry', NONE: 'none'}


def export_registry(strategies=None) -> dict:
    """Registry JSON for the orchestrator.

    Inconclusive is not a repair target there, and the fallback is applied by the engine when a
    ladder is exhausted, so neither is exported.
    """
    entries = []
    for strategy in strategies or default_strategies():
        spec = strategy.spec
        if spec.fallback:
            continue                  # the last resort is applied by the engine, never chosen by EV
        targets = [c for c in spec.categories if c != INCONCLUSIVE]
        if not targets:
            continue
        entries.append({
            'id': spec.id, 'name': spec.name, 'kind': spec.kind, 'targets': targets,
            'cost': {'tokens': spec.tokens, 'latency_s': spec.latency_s}, 'prior_success': spec.prior_success,
            'reversible': spec.reversible, 'description': spec.description,
            'template': f'{spec.description} Attributed step {{step_index}} ({{category}}): {{rationale}}'})
    return {'schema_version': SCHEMA_VERSION, 'strategies': entries}


@dataclass(frozen=True)
class RepairDirective:
    """Mirror of the orchestrator's directive, used when no factory is supplied."""
    kind: str                 # inject_guidance | retry | none
    strategy_id: str
    message: str = ''

    def as_step(self, index: int) -> 'InjectedStep':
        return InjectedStep(index, 'system', self.message)


@dataclass(frozen=True)
class InjectedStep:
    index: int
    role: str
    content: str
    tool_calls: tuple = ()
    name: str | None = None
    injected: bool = True

    @property
    def actor(self) -> str:
        return self.role

    @property
    def is_agent(self) -> bool:
        return False


@dataclass
class _Run:
    history: list[AttemptRecord] = field(default_factory=list)
    plan: RepairPlan | None = None
    lowered: Lowered | None = None


class OrchestratorRepairer:
    def __init__(self, engine: RepairEngine, directive_factory=RepairDirective):
        self.engine = engine
        self.factory = directive_factory
        self.runs: dict[str, _Run] = {}

    def build(self, strategy, candidate, trace):
        trace = Trace.from_any(trace)
        run = self.runs.setdefault(trace.trajectory_id, _Run())
        if run.plan is not None:
            run.history.append(AttemptRecord(run.plan.strategy_id, run.plan.category, run.plan.rollback_step, FAILED))
        hypothesis = Hypothesis.from_object(candidate)
        strategy_id = getattr(strategy, 'id', str(strategy))
        plan = None
        if any(s.spec.id == strategy_id for s in self.engine.strategies):
            result = self.engine.plan_for(strategy_id, trace, hypothesis, run.history)
            plan = None if isinstance(result, Skip) else result
        if plan is None:
            plan = self.engine.propose(trace, hypothesis, run.history)
        run.plan = plan
        if plan is None:
            run.lowered = None
            return self._fallback(strategy, strategy_id, hypothesis)
        run.lowered = lowered = lower(plan, ORCHESTRATOR)
        if lowered.kind == TERMINAL:
            # The orchestrator escalates unrepairable categories itself; a directive cannot carry a
            # terminal decision, so a terminal plan replays unchanged and is reported via last_plan().
            return self.factory('none', strategy_id, '')
        return self.factory(_KINDS[lowered.kind], plan.strategy_id, lowered.text)

    def _fallback(self, strategy, strategy_id: str, hypothesis: Hypothesis):
        template = getattr(strategy, 'template', '')
        if template and hypothesis.rationale:
            message = template.replace('{rationale}', hypothesis.rationale) \
                .replace('{step_index}', str(hypothesis.step_index)).replace('{category}', hypothesis.category)
            return self.factory('inject_guidance', strategy_id, message)
        return self.factory('retry', strategy_id, '')

    def last_plan(self, trajectory_id) -> RepairPlan | None:
        run = self.runs.get(str(trajectory_id))
        return run.plan if run else None

    def last_lowered(self, trajectory_id) -> Lowered | None:
        run = self.runs.get(str(trajectory_id))
        return run.lowered if run else None

    def record_outcome(self, trajectory_id, verified: bool, rollout=None) -> None:
        """Report the verifier's verdict on the last plan (enables lessons and tool memory)."""
        run = self.runs.get(str(trajectory_id))
        if run is None or run.plan is None:
            return
        record = self.engine.record_outcome(run.plan, verified, Trace.from_any(rollout) if rollout else None)
        run.history.append(record)
        run.plan = None
