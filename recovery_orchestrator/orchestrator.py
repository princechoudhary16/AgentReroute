"""Core loop: RCA payload -> decide -> roll back -> repair -> replay -> verify -> log.

Never declares success without the verifier passing the repaired rollout.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .budget import BudgetCaps, BudgetManager
from .detectors import RuntimeMonitor
from .policy import INTERVENE, ESCALATED, Decision, Option, UNMODIFIED_STRATEGY
from .rca import RCAPayload
from .replay import UNMODIFIED, ReplayEnvironment, Repairer, Rollout, TemplateRepairer
from .trace import Trace, build_prefix
from .verifier import Verifier, VerifierResult

RECOVERED, PARTIAL, ESCALATE, ABSTAIN, ABORTED = 'recovered', 'partial', 'escalated', 'abstained', 'aborted'


@dataclass
class Attempt:
    number: int
    option: dict
    rollout_id: str
    directive_kind: str
    tokens: int
    stopped_early: str | None
    verifier: dict | None
    applied_strategy: str = ''           # directive's strategy; a repairer may pick another ladder rung


@dataclass
class Outcome:
    trajectory_id: str
    policy: str
    status: str                          # RECOVERED | PARTIAL | ESCALATE | ABSTAIN | ABORTED
    reason: str
    intervened: bool
    attempts: list[Attempt] = field(default_factory=list)
    final_rollout_id: str | None = None  # recovered rollout, or best partial one
    gaps: tuple[str, ...] = ()
    original_verifier: dict | None = None
    budget: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Orchestrator:
    def __init__(self, env: ReplayEnvironment, verifier: Verifier, policy,
                 repairer: Repairer | None = None, caps: BudgetCaps | None = None,
                 verify_original: bool = True, monitor_loops: bool = True, log_path=None):
        self.env, self.verifier, self.policy = env, verifier, policy
        self.repairer = repairer or TemplateRepairer()
        self.caps = caps or BudgetCaps()
        self.verify_original = verify_original
        self.monitor_loops = monitor_loops
        self.log_path = Path(log_path) if log_path else None

    def run(self, trace: Trace, payload: RCAPayload) -> Outcome:
        if str(payload.trajectory_id) != str(trace.trajectory_id):
            raise ValueError('payload and trace refer to different trajectories')
        budget = BudgetManager(self.caps)
        guard = self.policy.make_guard()
        task = self.env.task_view(trace.trajectory_id)
        outcome = Outcome(trace.trajectory_id, self.policy.name, ABSTAIN, '', False)

        def log(event, **data):
            outcome.events.append({'event': event, 't': round(budget.elapsed, 3), **data})

        log('start', domain=trace.domain, n_steps=len(trace.steps), confidence=payload.confidence,
            inconclusive_share=payload.inconclusive_share,
            candidates=[{'step_index': c.step_index, 'category': c.category, 'share': c.share}
                        for c in payload.candidates])

        reference = None
        if self.verify_original:
            original = Rollout('original', trace.trajectory_id, 0, 'none', list(trace.steps))
            result = self.verifier.verify(task, original)
            budget.charge(result.tokens, verifier=True)
            outcome.original_verifier = _vdict(result)
            reference = result.score
            log('verify_original', **_vdict(result))

        best: tuple[float, Rollout, VerifierResult] | None = None
        stop_reason, last_decision, skipped, skipped_pairs = None, None, [], set()
        while True:
            decision: Decision = self.policy.decide(payload, trace, guard)
            last_decision = decision
            log('decision', action=decision.action, reason=decision.reason,
                option=decision.option.describe() if decision.option else None)
            if decision.action != INTERVENE:
                break
            option = decision.option
            blocked = budget.can_afford(option.rollback_step, option.est_tokens)
            if blocked:
                stop_reason = f'budget: {blocked}'
                log('budget_stop', reason=blocked, **budget.summary())
                break
            directive = self._directive(option, trace)
            if directive.kind == 'none' and directive is not UNMODIFIED:
                # The repairer has no plan this backend can execute (e.g. a terminal
                # compensate/escalate plan). Replaying unchanged would just duplicate the
                # control, so skip it and let the policy pick another option.
                if (option.rollback_step, option.strategy.id) in skipped_pairs:
                    log('no_progress', option=option.describe())
                    break           # policies that allow repeats would ask for it forever
                skipped_pairs.add((option.rollback_step, option.strategy.id))
                skipped.append(directive.strategy_id)
                guard.record(option.rollback_step, option.strategy.id, None, None)
                guard.record(option.rollback_step, directive.strategy_id, None, None)
                log('repair_not_applicable', option=option.describe(), applied_strategy=directive.strategy_id)
                continue
            budget.start_attempt(option.rollback_step)
            outcome.intervened = True
            rollout, result = self._attempt(trace, task, option, directive, budget)
            outcome.attempts.append(Attempt(budget.attempts, option.describe(), rollout.rollout_id,
                                            directive.kind, rollout.tokens, rollout.stopped_early,
                                            _vdict(result), directive.strategy_id))
            log('attempt', number=budget.attempts, rollout_id=rollout.rollout_id,
                applied_strategy=directive.strategy_id, stopped_early=rollout.stopped_early,
                verifier=_vdict(result))
            if rollout.stopped_early == 'budget':
                stop_reason = 'budget: environment refused the replay'
                log('budget_stop', reason='environment', **budget.summary())
                break
            if result.passed:
                outcome.status, outcome.reason = RECOVERED, 'verifier passed'
                outcome.final_rollout_id, outcome.gaps = rollout.rollout_id, result.gaps
                return self._finish(outcome, budget, log)
            score = result.score
            if not rollout.stopped_early and (best is None or score > best[0]):
                best = (score, rollout, result)
            self.policy.observe_failure(option)
            if directive.strategy_id != option.strategy.id:
                # The repairer substituted another strategy; never offer that one again here either.
                guard.record(option.rollback_step, directive.strategy_id, None, None)
            if guard.record(option.rollback_step, option.strategy.id, score, reference) == 'worsened':
                stop_reason = 'loop guard: repair worsened the verifier score'
                log('guard_stop', score=score, reference=reference)
                break

        # Fallbacks. Success is never declared without verification.
        improved = best is not None and (best[0] > reference if reference is not None else best[0] > 0)
        if improved:
            outcome.status, outcome.reason = PARTIAL, 'best repair improved but did not pass verification'
            outcome.final_rollout_id, outcome.gaps = best[1].rollout_id, best[2].gaps
        elif outcome.intervened:
            outcome.status = ABORTED if stop_reason else ESCALATE
            outcome.reason = stop_reason or f'no verified repair ({last_decision.reason})'
        elif stop_reason:
            outcome.status, outcome.reason = ABORTED, stop_reason
        elif skipped:
            outcome.status = ESCALATE
            outcome.reason = f'repairer had no executable plan ({", ".join(skipped)}); {last_decision.reason}'
        elif last_decision.action == ESCALATED:
            outcome.status, outcome.reason = ESCALATE, last_decision.reason
        else:
            outcome.status, outcome.reason = ABSTAIN, last_decision.reason
        return self._finish(outcome, budget, log)

    def _directive(self, option: Option, trace: Trace):
        if option.strategy is UNMODIFIED_STRATEGY:
            return UNMODIFIED
        return self.repairer.build(option.strategy, option.primary, trace)

    def _attempt(self, trace: Trace, task, option: Option, directive, budget: BudgetManager):
        prefix = build_prefix(trace, option.rollback_step, self.env.write_tools)
        monitor = RuntimeMonitor(trace.trajectory_id, trace.domain,
                                 write_tools=self.env.write_tools) if self.monitor_loops else None
        rollout = self.env.rollout(trace, prefix, directive, monitor, budget.remaining_tokens())
        budget.charge(rollout.tokens)
        if rollout.stopped_early:
            # A looping or truncated rollout fails without spending verifier tokens.
            result = VerifierResult(False, 0.0, 'FAIL', (f'stopped early: {rollout.stopped_early}',))
        else:
            result = self.verifier.verify(task, rollout)
            budget.charge(result.tokens, verifier=True)
        if directive is not UNMODIFIED:
            self._report_outcome(trace.trajectory_id, result.passed, rollout)
        return rollout, result

    def _report_outcome(self, trajectory_id, verified: bool, rollout: Rollout) -> None:
        """Tell a learning repairer the verifier's verdict (never the ground-truth reward)."""
        record = getattr(self.repairer, 'record_outcome', None)
        if record is not None:
            record(trajectory_id, verified, rollout)

    def _finish(self, outcome: Outcome, budget: BudgetManager, log) -> Outcome:
        outcome.budget = budget.summary()
        log('end', status=outcome.status, reason=outcome.reason, **outcome.budget)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(outcome.to_dict(), ensure_ascii=False) + '\n')
        return outcome


def _vdict(result: VerifierResult | None) -> dict | None:
    if result is None:
        return None
    return {'passed': result.passed, 'score': result.score, 'verdict': result.verdict,
            'gaps': list(result.gaps), 'tokens': result.tokens}
