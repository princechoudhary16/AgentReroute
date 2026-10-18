"""Integration with the benchmark recovery harness (``benchmark.scoring.recovery``).

``RepairStrategyPolicy`` implements the harness's ``recover(env)`` protocol:
for each attribution hypothesis it asks the engine for the next rung, lowers
the plan to the benchmark's ``Directive``, replays, and accepts a rollout only
if the plan's acceptance checks pass and the run ended normally. If replays ran
but none was accepted, the last one is delivered (``deliver_unaccepted=False``
escalates instead). If no replay could run at all, the fallback rolls back the
side effects made from the fault onward and escalates. Rewards stay in the
harness's ledger; this policy never sees them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import acceptance
from ..context import FAILED, AttemptRecord, Hypothesis, SpecError, TaskSpec, ToolSpec
from ..engine import RepairEngine
from ..lowering import BENCHMARK, lower
from ..taxonomy import UnknownCategory
from ..trace import Trace


def task_from_view(view: dict, template: TaskSpec | None = None) -> TaskSpec:
    """TaskSpec for one episode: the template's rules and metadata, the episode's request, policy and agents."""
    template = template or TaskSpec.preset('retail')
    agent_tools = {a: tuple(s['name'] for s in schemas) for a, schemas in (view.get('tools') or {}).items()}
    tools = dict(template.tools)
    for schemas in (view.get('tools') or {}).values():
        for schema in schemas:
            if schema['name'] not in tools:
                tools[schema['name']] = ToolSpec.from_schema(schema)
    return TaskSpec(instruction=view.get('first_user_message', '') or '', policy=view.get('policy', '') or '',
                    tools=tools, agent_tools=agent_tools or dict(template.agent_tools),
                    equivalents=dict(template.equivalents), rules=template.rules,
                    compensations=dict(template.compensations), safe_defaults=dict(template.safe_defaults),
                    fact_keywords=dict(template.fact_keywords), name=template.name)


@dataclass
class EpisodeLog:
    attempts: list[dict] = field(default_factory=list)
    skipped_hypotheses: list[str] = field(default_factory=list)


class RepairStrategyPolicy:
    name = 'repair_strategies'

    def __init__(self, directive_cls=None, template: TaskSpec | None = None, max_hypotheses: int = 2,
                 accept_terminations=('user_stop',), deliver_unaccepted: bool = True, engine_factory=None,
                 learn: bool = False):
        """``learn=True`` reports each attempt to the engine (tool memory; lessons from accepted replays).
        Acceptance then stands in for verification, so pass shared memories via ``engine_factory``."""
        self._directive_cls = directive_cls
        self.template = template
        self.max_hypotheses = max_hypotheses
        self.accept_terminations = tuple(accept_terminations)
        self.deliver_unaccepted = deliver_unaccepted
        self.engine_factory = engine_factory or (lambda task: RepairEngine(task))
        self.learn = learn
        self.log: dict[str, EpisodeLog] = {}

    @property
    def directive_cls(self):
        if self._directive_cls is None:
            try:
                from benchmark.env.workflow import Directive
            except ImportError as exc:
                raise RuntimeError('pass directive_cls=benchmark.env.workflow.Directive') from exc
            self._directive_cls = Directive
        return self._directive_cls

    def _hypotheses(self, env, log: EpisodeLog) -> list[Hypothesis]:
        found = []
        for raw in env.hypotheses or []:
            try:
                found.append(Hypothesis.from_mapping(raw))
            except (UnknownCategory, SpecError, TypeError, ValueError) as exc:
                log.skipped_hypotheses.append(f'{raw!r}: {exc}')
        return found[:self.max_hypotheses]

    @staticmethod
    def _affordable(env) -> bool:
        left = env.remaining()
        return all(left.get(k, 1) > 0 for k in ('tokens', 'attempts', 'seconds'))

    def recover(self, env):
        log = self.log.setdefault(str(env.trajectory_id), EpisodeLog())
        hypotheses = self._hypotheses(env, log)
        if not hypotheses:
            return {'delivered': None, 'status': 'abstained', 'notes': 'no usable attribution'}
        task = task_from_view(env.task_view, self.template)
        engine = self.engine_factory(task)
        trace = Trace.from_ir(env.trace)
        last, terminal, tried = None, None, []
        for hypothesis in hypotheses:
            history: list[AttemptRecord] = []
            while self._affordable(env):
                plan = engine.propose(trace, hypothesis, history)
                if plan is None:
                    break
                if plan.terminal:
                    # Held back: another hypothesis may still repair the run.
                    terminal = terminal or plan
                    break
                lowered = lower(plan, BENCHMARK)
                directive = self.directive_cls(rollback_step=lowered.rollback_step, kind=lowered.kind,
                                               text=lowered.text, seed_offset=lowered.seed_offset)
                rollout = env.replay(directive)
                rollout_trace = Trace.from_ir(rollout.trace)
                report = acceptance.evaluate(plan, rollout_trace, task, start=lowered.rollback_step)
                accepted = report.passed and rollout.termination in self.accept_terminations
                log.attempts.append({'strategy': plan.strategy_id, 'rollback_step': plan.rollback_step,
                                     'rollout': rollout.rollout_id, 'accepted': accepted,
                                     'termination': rollout.termination, 'degraded': list(lowered.degraded),
                                     'failures': [f.detail for f in report.failures]})
                if self.learn:
                    engine.record_outcome(plan, accepted, rollout_trace)
                if accepted:
                    return {'delivered': rollout.rollout_id, 'status': 'intervened',
                            'notes': f'{plan.strategy_id} accepted at step {plan.rollback_step}'}
                record = AttemptRecord(plan.strategy_id, plan.category, plan.rollback_step, FAILED)
                history.append(record)
                tried.append(record)
                last = rollout
        if last is not None and self.deliver_unaccepted:
            return {'delivered': last.rollout_id, 'status': 'intervened', 'notes': 'no rollout passed acceptance'}
        # Worst case: nothing was accepted (ladders exhausted or budget spent). Fall back to rolling back
        # the side effects made from the fault onward and escalating, with every attempt on record.
        final = terminal if terminal is not None and not engine.is_fallback(terminal) else \
            engine.fallback(trace, hypotheses[0], tried)
        if final is None:
            return {'delivered': None, 'status': 'abstained', 'notes': 'no repair applies and no fallback is set'}
        lowered = lower(final, BENCHMARK)
        log.attempts.append({'strategy': final.strategy_id, 'terminal': lowered.terminal_action,
                             'summary': lowered.text, 'degraded': list(lowered.degraded)})
        status = 'escalated' if lowered.terminal_action == 'escalate' else 'abstained'
        return {'delivered': None, 'status': status, 'notes': f'{final.strategy_id}: {final.operations[-1].reason}'}
