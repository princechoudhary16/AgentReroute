"""Run the orchestrator inside the benchmark's recovery harness (``benchmark.scoring.recovery``).

``OrchestratorPolicy`` implements the harness's ``recover(env)`` protocol. It turns
the episode's attribution hypotheses into an ``RCAPayload``, wraps the harness's
``RecoveryEnv`` as a ``ReplayEnvironment``, runs the orchestrator, and delivers
the verified rollout (or the best partial one, if allowed).

This module never imports the ``benchmark`` package. Pass its ``Directive`` class
as ``directive_cls``, as the Repair team's adapter does. The harness keeps the
reward ledger; nothing here can see rewards.
"""
from __future__ import annotations

from ..orchestrator import PARTIAL, RECOVERED, Orchestrator
from ..rca import Candidate, RCAPayload
from ..replay import Rollout, TaskView
from ..tau_retail import WRITE_TOOLS
from ..taxonomy import category_from_mode
from ..trace import Trace

_KINDS = {'inject_guidance': 'guidance', 'retry': 'resample', 'none': 'none'}


def payload_from_hypotheses(trajectory_id, domain: str, hypotheses: list[dict],
                            source: str = 'benchmark') -> RCAPayload:
    """Benchmark submission hypotheses (mode or category, confidence) -> RCAPayload.

    With one source the share is the hypothesis' confidence, as for Track B.
    """
    candidates, seen = [], set()
    for h in hypotheses:
        if h.get('step_index') is None or h['step_index'] in seen:
            continue
        category = category_from_mode(h.get('mode', h.get('category')))
        if category == 'Inconclusive':
            continue
        seen.add(h['step_index'])
        candidates.append(Candidate(int(h['step_index']), h.get('agent'), category, 1,
                                    float(h.get('confidence', 1.0)), h.get('rationale', ''), source=source))
    candidates.sort(key=lambda c: (-c.share, c.step_index))
    return RCAPayload(str(trajectory_id), domain, 'ok', candidates[:3], len(candidates[:3]), 1,
                      0.0 if candidates else 1.0, confidence_kind=f'{source}_confidence')


class BenchmarkEnvironment:
    """``ReplayEnvironment`` over one ``RecoveryEnv``. Replays run whole (the harness has no
    step hook), so the runtime monitor inspects them afterwards and flags loops."""

    write_tools = WRITE_TOOLS
    ledger = None                    # the harness owns the ground truth

    def __init__(self, env, directive_cls, domain: str = 'tau'):
        self.env, self.directive_cls, self.domain = env, directive_cls, domain
        self.raw = {}

    def task_view(self, trajectory_id) -> TaskView:
        view = self.env.task_view or {}
        return TaskView(str(trajectory_id), self.domain, instruction=view.get('first_user_message', ''),
                        policy=view.get('policy', ''))

    def rollout(self, trace, prefix, directive, monitor, max_tokens):
        kind = _KINDS[directive.kind]
        bench = self.directive_cls(rollback_step=max(1, prefix.rollback_step), kind=kind,
                                   text=directive.message if kind == 'guidance' else '')
        try:
            raw = self.env.replay(bench)
        except Exception as exc:          # the harness's BudgetExceeded; anything else is a bug
            if type(exc).__name__ != 'BudgetExceeded':
                raise
            return Rollout(f'budget-{prefix.rollback_step}-{directive.strategy_id}', trace.trajectory_id,
                           prefix.rollback_step, directive.strategy_id, list(prefix.steps),
                           stopped_early='budget')
        self.raw[raw.rollout_id] = raw
        steps = Trace.from_ir(raw.trace).steps
        rollout = Rollout(raw.rollout_id, trace.trajectory_id, prefix.rollback_step, directive.strategy_id,
                          steps, tokens=raw.tokens)
        if monitor is not None:
            for step in steps:
                found = monitor.observe(step)
                if found and step.index >= prefix.rollback_step and rollout.stopped_early is None:
                    rollout.findings.extend(found)
                    rollout.stopped_early = found[0].kind
        return rollout


class CallbackVerifier:
    """Verifier from an observable acceptance check on the raw harness rollout.

    Offline stand-in for ``FreshContextVerifier`` when no LLM is available, e.g. the
    benchmark baselines' ``looks_ok``. It is blind to gold, as a verifier must be.
    """

    def __init__(self, check, environment_ref):
        self.check, self.environment_ref = check, environment_ref

    def verify(self, task, rollout):
        from ..verifier import VerifierResult
        raw = self.environment_ref().raw.get(rollout.rollout_id)
        if raw is None:                   # the original trace, or a rollout we never ran
            return VerifierResult(False, 0.0, 'FAIL', ('not a replay',))
        ok = bool(self.check(raw))
        return VerifierResult(ok, 1.0 if ok else 0.0, 'PASS' if ok else 'FAIL',
                              () if ok else (f'observable check failed (termination {raw.termination})',))


class OrchestratorPolicy:
    """Benchmark ``RecoveryPolicy``: the orchestrator with a pluggable policy, verifier and repairer.

    ``policy_factory()`` -> an orchestrator policy (fresh per episode).
    ``verifier_factory(environment)`` -> a ``Verifier``.
    ``repairer_factory(task_view)`` -> a ``Repairer`` (default: the registry templates).
    ``caps_factory(remaining)`` -> ``BudgetCaps`` from the harness's remaining budget.
    """

    def __init__(self, directive_cls, policy_factory, verifier_factory, repairer_factory=None,
                 caps_factory=None, name='orchestrator', deliver_partial=False, verify_original=False):
        self.directive_cls = directive_cls
        self.policy_factory, self.verifier_factory = policy_factory, verifier_factory
        self.repairer_factory, self.caps_factory = repairer_factory, caps_factory
        self.name, self.deliver_partial, self.verify_original = name, deliver_partial, verify_original
        self.outcomes = {}

    def recover(self, env):
        trace = Trace.from_ir(env.trace)
        payload = payload_from_hypotheses(env.trajectory_id, trace.domain or 'tau', env.hypotheses)
        environment = BenchmarkEnvironment(env, self.directive_cls, trace.domain or 'tau')
        verifier = self.verifier_factory(environment)
        repairer = self.repairer_factory(env.task_view) if self.repairer_factory else None
        caps = self.caps_factory(env.remaining()) if self.caps_factory else None
        orchestrator = Orchestrator(environment, verifier, self.policy_factory(), repairer=repairer, caps=caps,
                                    verify_original=self.verify_original)
        outcome = orchestrator.run(trace, payload)
        self.outcomes[str(env.trajectory_id)] = outcome
        deliver = outcome.status == RECOVERED or (self.deliver_partial and outcome.status == PARTIAL)
        delivered = outcome.final_rollout_id if deliver and outcome.final_rollout_id in environment.raw else None
        status = 'intervened' if outcome.intervened else ('escalated' if outcome.status == 'escalated'
                                                          else 'abstained')
        return {'delivered': delivered, 'status': status, 'notes': f'{outcome.status}: {outcome.reason}'}


__all__ = ['OrchestratorPolicy', 'BenchmarkEnvironment', 'CallbackVerifier', 'payload_from_hypotheses']
