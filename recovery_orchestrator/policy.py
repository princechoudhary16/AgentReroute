"""Decision policies: where to roll back, which strategy, or whether to abstain.

* ``BaselinePolicy``: top-1 hypothesis, its best-prior strategy, retried up to N
  times. Every later improvement must beat it.
* ``EVPolicy``: scores (rollback point, strategy) options by expected value
  ``P(success) * value - cost`` and intervenes only when EV > 0. Rolling back
  to an earlier step covers every candidate downstream of it, so near-tied
  candidates lead to the earliest rollback point. After a verified failure the
  belief in the covered candidates is lowered (Bayes), so the next option moves
  to another hypothesis instead of repeating itself.
* ``UnmodifiedReplayPolicy``: control; same rollback point as another policy,
  no repair applied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .budget import LoopGuard
from .detectors import detect_loops, detect_tool_errors
from .rca import Candidate, RCAPayload
from .registry import Strategy, StrategyRegistry
from .taxonomy import ABSTAIN, ESCALATE, handling
from .trace import Trace, build_prefix, rollback_point

INTERVENE, ABSTAINED, ESCALATED = 'intervene', 'abstain', 'escalate'

UNMODIFIED_STRATEGY = Strategy(id='unmodified_replay', name='Unmodified replay (control)', kind='none',
                               targets=frozenset({'*'}), tokens=0, latency_s=0.0, prior_success=0.0)


@dataclass(frozen=True)
class Option:
    rollback_step: int
    strategy: Strategy
    primary: Candidate                 # candidate whose rationale feeds the repair
    covered: tuple[Candidate, ...]
    p_success: float
    est_tokens: int
    cost: float
    ev: float

    def describe(self) -> dict:
        return {'rollback_step': self.rollback_step, 'strategy': self.strategy.id,
                'primary': {'step_index': self.primary.step_index, 'category': self.primary.category,
                            'source': self.primary.source},
                'covered_steps': [c.step_index for c in self.covered],
                'p_success': round(self.p_success, 4), 'est_tokens': self.est_tokens,
                'cost': round(self.cost, 4), 'ev': round(self.ev, 4)}


@dataclass(frozen=True)
class Decision:
    action: str                        # INTERVENE | ABSTAINED | ESCALATED
    reason: str
    option: Option | None = None


@dataclass(frozen=True)
class EVConfig:
    value_of_recovery: float = 1.0
    cost_per_1k_tokens: float = 0.01
    cost_per_second: float = 0.0
    replay_tokens_per_step: int = 400      # regenerating each step after the rollback point
    verifier_tokens: int = 3000            # one fresh-context verification
    distance_discount: float = 0.97        # guidance applied k steps early works less well
    min_ev: float = 0.0
    abstain_inconclusive_share: float = 0.5
    escalate_min_confidence: float = 0.5   # an unrepairable top hypothesis this sure -> escalate
    detector_weight: float = 0.2           # pseudo-confidence for runtime-detector findings
    respect_irreversible: bool = True      # never roll back across an executed write action
    calibrator: Callable[[float], float] | None = None


def _calibrated(config: EVConfig, share: float) -> float:
    return min(1.0, max(0.0, config.calibrator(share) if config.calibrator else share))


def detector_candidates(trace: Trace, existing: list[Candidate], weight: float,
                        write_tools: frozenset[str]) -> list[Candidate]:
    steps = {c.step_index for c in existing}
    found = detect_loops(trace, write_tools=write_tools) + detect_tool_errors(trace)
    out, seen = [], set()
    for f in found:
        if f.step_index in steps or f.step_index in seen:
            continue
        seen.add(f.step_index)
        out.append(f.as_candidate(weight))
    return out


def _gate(payload: RCAPayload, config: EVConfig) -> Decision | None:
    """Category/abstention rules that apply before any option is scored."""
    top = payload.top
    if payload.fa_decision == 'within_boundary':
        return Decision(ABSTAINED, 'FA divergence gate: within boundary (no repair required)')
    if top is None:
        return Decision(ABSTAINED, 'inconclusive: FA localized no step')
    if payload.inconclusive_share >= config.abstain_inconclusive_share:
        return Decision(ABSTAINED, f'inconclusive_share {payload.inconclusive_share:.2f} '
                                   f'>= {config.abstain_inconclusive_share}')
    mode = handling(top.category)
    if mode == ESCALATE and _calibrated(config, top.share) >= config.escalate_min_confidence:
        return Decision(ESCALATED, f'{top.category} is usually not repairable')
    return None


def _feasible(trace: Trace, step: int | None, write_tools, config: EVConfig) -> str | None:
    """Why a rollback to ``step`` is not allowed, or None."""
    if step is None:
        return 'no agent step to roll back to'
    if config.respect_irreversible and build_prefix(trace, step, write_tools).irreversible_downstream:
        return 'irreversible step downstream'
    return None


def _cost(config: EVConfig, trace: Trace, rollback: int, strategy: Strategy) -> tuple[int, float]:
    replayed = max(1, sum(1 for s in trace.steps if s.index >= rollback))
    tokens = strategy.tokens + config.replay_tokens_per_step * replayed + config.verifier_tokens
    return tokens, tokens / 1000 * config.cost_per_1k_tokens + strategy.latency_s * config.cost_per_second


class BaselinePolicy:
    """Top-1 hypothesis -> its best-prior strategy, retried up to ``retries`` times."""
    name = 'baseline'

    def __init__(self, registry: StrategyRegistry, retries: int = 3, config: EVConfig | None = None,
                 write_tools: frozenset[str] = frozenset()):
        self.registry, self.retries = registry, retries
        self.config = config or EVConfig()
        self.write_tools = write_tools

    def make_guard(self) -> LoopGuard:
        return LoopGuard(allow_repeats=True, stop_on_worse=False)

    def decide(self, payload: RCAPayload, trace: Trace, guard: LoopGuard) -> Decision:
        top = payload.top
        if top is None:
            return Decision(ABSTAINED, 'inconclusive: FA localized no step')
        if len(guard.tried_list) >= self.retries:
            return Decision(ABSTAINED, f'retry limit {self.retries} reached')
        if handling(top.category) in (ESCALATE, ABSTAIN):
            return Decision(ESCALATED, f'{top.category} is usually not repairable')
        strategies = self.registry.for_category(top.category)
        if not strategies:
            return Decision(ESCALATED, f'no strategy targets {top.category}')
        step = rollback_point(trace, top.step_index)
        blocked = _feasible(trace, step, self.write_tools, self.config)
        if blocked:
            return Decision(ESCALATED, blocked)
        strategy = strategies[0]
        tokens, cost = _cost(self.config, trace, step, strategy)
        return Decision(INTERVENE, 'top-1 hypothesis', Option(
            step, strategy, top, (top,), top.share * strategy.prior_success, tokens, cost,
            top.share * strategy.prior_success * self.config.value_of_recovery - cost))

    def observe_failure(self, option: Option) -> None:
        pass


class EVPolicy:
    name = 'ev'

    def __init__(self, registry: StrategyRegistry, config: EVConfig | None = None,
                 write_tools: frozenset[str] = frozenset(), use_detectors: bool = True):
        self.registry = registry
        self.config = config or EVConfig()
        self.write_tools = write_tools
        self.use_detectors = use_detectors
        self._belief: dict[tuple[int, str], float] = {}

    def make_guard(self) -> LoopGuard:
        self._belief = {}
        return LoopGuard()

    def _key(self, c: Candidate):
        return (c.step_index, c.category)

    def belief(self, c: Candidate) -> float:
        return self._belief.get(self._key(c), _calibrated(self.config, c.share))

    def candidates(self, payload: RCAPayload, trace: Trace) -> list[Candidate]:
        cands = list(payload.candidates)
        if self.use_detectors:
            cands += detector_candidates(trace, cands, self.config.detector_weight, self.write_tools)
        return cands

    def options(self, payload: RCAPayload, trace: Trace, guard: LoopGuard) -> tuple[list[Option], list[str]]:
        cfg = self.config
        cands = self.candidates(payload, trace)
        points = {}
        for c in cands:
            points.setdefault(c, rollback_point(trace, c.step_index))
        options, blocked = [], []
        for step in sorted({p for p in points.values() if p is not None}):
            reason = _feasible(trace, step, self.write_tools, cfg)
            if reason:
                blocked.append(f'step {step}: {reason}')
                continue
            covered = [c for c, p in points.items() if p is not None and p >= step]
            for strategy in self.registry:
                if not guard.allowed(step, strategy.id) or (not strategy.reversible and cfg.respect_irreversible):
                    continue
                p, primary = 0.0, None
                for c in covered:
                    if handling(c.category) in (ESCALATE, ABSTAIN) or not strategy.targets_category(c.category):
                        continue
                    term = self.belief(c) * strategy.prior_success * cfg.distance_discount ** (points[c] - step)
                    p += term
                    if primary is None or term > primary[0]:
                        primary = (term, c)
                if primary is None:
                    continue
                tokens, cost = _cost(cfg, trace, step, strategy)
                options.append(Option(step, strategy, primary[1], tuple(covered), min(p, 1.0), tokens,
                                      cost, min(p, 1.0) * cfg.value_of_recovery - cost))
        # Highest EV first; on ties prefer the earlier rollback point (more candidates downstream).
        options.sort(key=lambda o: (-round(o.ev, 9), o.rollback_step, o.strategy.id))
        return options, blocked

    def decide(self, payload: RCAPayload, trace: Trace, guard: LoopGuard) -> Decision:
        gated = _gate(payload, self.config)
        if gated:
            return gated
        options, blocked = self.options(payload, trace, guard)
        if options and options[0].ev > self.config.min_ev:
            return Decision(INTERVENE, f'best EV {options[0].ev:.3f}', options[0])
        if not options and blocked:
            return Decision(ESCALATED, '; '.join(blocked))
        best = f'{options[0].ev:.3f}' if options else 'none'
        return Decision(ABSTAINED, f'no option with EV > {self.config.min_ev} (best {best})')

    def observe_failure(self, option: Option) -> None:
        """P(candidate right | strategy failed) = b(1-q) / (1 - b q), q = P(fix | right)."""
        for c in option.covered:
            if not option.strategy.targets_category(c.category):
                continue
            b = self.belief(c)
            q = option.strategy.prior_success * self.config.distance_discount ** max(0, c.step_index - option.rollback_step)
            self._belief[self._key(c)] = b * (1 - q) / (1 - b * q) if b * q < 1 else 0.0


class UnmodifiedReplayPolicy:
    """Control: the inner policy's first rollback point, replayed without any repair."""

    def __init__(self, inner, retries: int = 1):
        self.inner, self.retries = inner, retries
        self.name = f'unmodified[{inner.name}]'

    def make_guard(self) -> LoopGuard:
        self._guard = self.inner.make_guard()
        return LoopGuard(allow_repeats=True, stop_on_worse=False)

    def decide(self, payload, trace, guard) -> Decision:
        if len(guard.tried_list) >= self.retries:
            return Decision(ABSTAINED, 'control replay done')
        decision = self.inner.decide(payload, trace, self._guard)
        if decision.action != INTERVENE:
            return decision
        o = decision.option
        tokens = o.est_tokens - o.strategy.tokens
        return Decision(INTERVENE, 'unmodified replay control', Option(
            o.rollback_step, UNMODIFIED_STRATEGY, o.primary, o.covered, 0.0, tokens, 0.0, 0.0))

    def observe_failure(self, option) -> None:
        pass
