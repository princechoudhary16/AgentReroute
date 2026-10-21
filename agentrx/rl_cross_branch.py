"""Stage B closed loop: diagnose_mp_rl -> repair_strategies -> benchmark / orchestrator replay -> theta and M_i.

Rewards come only from observable verifiers: repair_strategies acceptance plus rollout termination on the
benchmark path, the orchestrator's fresh-context VerifierResult on the other. Nothing here reads
runner.ledger, episode rewards, gold_state_hash or benchmark/corpus/gold.
"""
import re
import tempfile
from pathlib import Path

from benchmark.env.workflow import Directive
from benchmark.scoring.baselines import heuristic_findings
from benchmark.scoring.recovery import RecoveryPolicy, looks_ok, run_recovery
from benchmark.taxonomy import from_agentrx
from recovery_orchestrator.orchestrator import RECOVERED, Orchestrator
from recovery_orchestrator.policy import EVPolicy
from recovery_orchestrator.rca import RCAPayload
from recovery_orchestrator.registry import StrategyRegistry
from recovery_orchestrator.replay import RepairDirective
from repair_strategies.adapters.benchmark import RepairStrategyPolicy
from repair_strategies.adapters.orchestrator import OrchestratorRepairer, export_registry
from repair_strategies.context import Hypothesis, TaskSpec
from repair_strategies.engine import RepairEngine
from repair_strategies.taxonomy import UnknownCategory
from .io import read_json
from .perspectives import STUCK_TOOL
from .pipeline import diagnose_mp_rl
from .providers import ProviderError
from .rl_policy import DualLoopRLJudge


class _SeedOnlyProvider:
    """evidence=False runs only seed python checks, so the model is never called; a call is a bug."""
    measurement_kind = 'offline_fixture'
    identity = {'provider': 'seed_only'}

    def complete(self, stage, payload):
        raise ProviderError(f'seed-only attribution made a model call ({stage})')


def trace_tools_and_policy(task_view: dict) -> tuple[str, dict]:
    """RecoveryEnv.task_view ({agent: [schema, ...]}, policy) -> (policy, flat {tool_name: schema})."""
    tools = {s['name']: s for schemas in (task_view.get('tools') or {}).values() for s in schemas}
    return task_view.get('policy', '') or '', tools


def _fault_label(c, steps):
    """(category, mode) for a candidate. The AgentRx category alone loses two refinements that pick the repair
    ladder: a grounding violation on a handoff is context_propagation, and a stuck/repeated tool loop is a Cycle."""
    names = {(v['assertion_name'], v['step_index']) for v in c.get('violated_checks', ())}
    step = steps.get(c['step_index'])
    if step and any(n == 'p3_seed_handoff_and_fact_grounding' for n, _ in names) \
            and any(e.get('name') == 'handoff' for e in step['substeps']):
        return 'MisinterpretationOfToolOutput', 'context_propagation'
    for name, index in names:
        events = steps[index]['substeps'] if index in steps else []
        if (name == 'p2_seed_valid_tool_invocation' and any(
                e.get('role') == 'tool' and re.search(STUCK_TOOL, str(e.get('content', ''))) for e in events)) or (
                name == 'p1_seed_specialist_routing_alignment' and any(e.get('tool_calls') for e in events)):
            return 'Cycle', 'cycles'    # the routing seed fires on a tool-call step only for a 3rd identical call
    return c['category'], from_agentrx(c['category'])[0]


def candidates_to_benchmark_hypotheses(ranked_candidates: list[dict], trace: dict | None = None) -> list[dict]:
    """diagnose_mp_rl ranked_candidates -> benchmark hypotheses with both 'category' and 'mode'.

    With the trace, handoff-grounding and tool-loop violations refine the label (see _fault_label)."""
    steps = {s['index']: s for s in (trace or {}).get('steps', [])}
    hypotheses = []
    for c in ranked_candidates:
        category, mode = _fault_label(c, steps) if steps else (c['category'], from_agentrx(c['category'])[0])
        hypotheses.append({'step_index': int(c['step_index']), 'agent': c.get('raw_agent') or c['agent'],
                           'mode': mode, 'category': category,
                           'confidence': round(float(c.get('probability', c.get('confidence', 0.5))), 4),
                           'rationale': c.get('rationale', '')})
    return hypotheses


def _repairable(hypothesis):
    """Hypothesis.from_mapping reads 'mode' as a subtype and rejects one that does not refine the category
    (InvalidInvocation/tool_failure, any out_of_scope), so the repair side then gets the category alone."""
    try:
        Hypothesis.from_mapping(hypothesis)
        return hypothesis
    except UnknownCategory:
        return {k: v for k, v in hypothesis.items() if k != 'mode'}


class _CreditEngine(RepairEngine):
    """Remembers the hypothesis of each proposal: RepairStrategyPolicy replays every non-terminal plan and returns
    right after an accepted replay, so `current` earned the reward and `replayed` lists the steps tried before it."""
    current = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replayed = []

    def propose(self, trace, hypothesis, history=()):
        self.current = hypothesis
        plan = super().propose(trace, hypothesis, history)
        if plan is not None and not plan.terminal:
            self.replayed.append(hypothesis.step_index)
        return plan


def _diagnose(trajectory, policy, tools, provider, rl_judge, evidence, explore):
    config = {'protocol': 'mp_rl', 'evidence': evidence, 'checklist': True, 'recovery_loop': False,
              'rl_explore': explore}
    with tempfile.TemporaryDirectory() as artifacts:
        result = diagnose_mp_rl(trajectory, policy, tools, config, provider, artifacts, rl_judge=rl_judge)
        return result, read_json(Path(artifacts)/'rca_payload.json')


class MPRLRecoveryPolicy(RepairStrategyPolicy, RecoveryPolicy):
    """benchmark RecoveryPolicy: online mp_rl attribution -> repair ladder -> replay -> observable RL update.

    Verified = acceptance.evaluate(...).passed with a 'user_stop' termination; otherwise a rollout that
    looks_ok() scores 0.5. Pre-supplied env.hypotheses (e.g. the harness's oracle) are repaired but never
    learned from.
    """
    name = 'mp_rl_recovery'

    def __init__(self, provider=None, rl_judge: DualLoopRLJudge | None = None, rl_state_path=None,
                 use_llm_synthesis: bool = False, learn: bool = True, max_hypotheses: int = 3,
                 deliver_unaccepted: bool = True):
        if use_llm_synthesis and provider is None:
            raise ValueError('use_llm_synthesis needs a provider')
        super().__init__(directive_cls=Directive, max_hypotheses=max_hypotheses,
                         deliver_unaccepted=deliver_unaccepted, engine_factory=self._engine, learn=learn)
        self.provider = provider if use_llm_synthesis else _SeedOnlyProvider()
        self.use_llm_synthesis, self.rl_state_path = use_llm_synthesis, rl_state_path
        self.rl_judge = rl_judge or (DualLoopRLJudge.load(rl_state_path) if rl_state_path else DualLoopRLJudge())
        self.engine, self.hypotheses, self.rl_log = None, {}, {}

    def _engine(self, task):
        self.engine = _CreditEngine(task)
        return self.engine

    def attribute(self, env) -> list[dict]:
        policy, tools = trace_tools_and_policy(env.task_view)
        result, _ = _diagnose(env.trace, policy, tools, self.provider, self.rl_judge,
                              self.use_llm_synthesis, explore=self.learn)
        tid = str(env.trajectory_id)
        if not any(c.get('violated_checks') for c in result['ranked_candidates']) and not heuristic_findings(env.trace):
            # Nothing observable is wrong: abstain on what looks like a healthy run (no action, no RL update).
            self.rl_judge.pending.pop(tid, None)
            self.hypotheses[tid] = []
            return []
        hypotheses = candidates_to_benchmark_hypotheses(result['ranked_candidates'], env.trace)[:self.max_hypotheses]
        self.hypotheses[tid] = hypotheses
        return hypotheses

    def recover(self, env):
        tid = str(env.trajectory_id)
        learning = self.learn and not env.hypotheses
        if not env.hypotheses:
            env.hypotheses = [_repairable(h) for h in self.attribute(env)]
        self.log.pop(tid, None)            # one episode's attempts only, across training epochs
        self.engine = None
        decision = super().recover(env)
        if tid not in self.rl_judge.pending:
            return decision
        if not learning:
            del self.rl_judge.pending[tid]
            return decision
        rollouts = [env.rollouts[a['rollout']] for a in self.log[tid].attempts if 'rollout' in a]
        verified = any(a.get('accepted') for a in self.log[tid].attempts)
        after = 1.0 if verified else max((0.5 if looks_ok(r) else 0.0 for r in rollouts), default=0.0)
        stopped = bool(rollouts) and not verified and rollouts[-1].termination != 'user_stop'
        step = self.engine.current.step_index if verified else None
        failed = [s for s in self.engine.replayed if s != step] if verified else []
        telemetry = self.rl_judge.record_recovery_outcome(tid, step, verified=verified, verifier_score_before=0.0,
                                                          verifier_score_after=after, stopped_early=stopped,
                                                          failed_steps=failed)
        self.rl_log[tid] = {'verified': verified, 'score_after': after, 'stopped_early': stopped,
                            'credited_step': step, 'failed_steps': telemetry['pairwise_steps'],
                            'reward': telemetry['reward']}
        if self.rl_state_path:
            self.rl_judge.save(self.rl_state_path)
        return decision


class MPRLOrchestratorSession:
    """diagnose_mp_rl -> rca_payload.json -> RCAPayload -> Orchestrator(EVPolicy, OrchestratorRepairer) -> RL update."""

    def __init__(self, env, verifier, provider=None, rl_judge: DualLoopRLJudge | None = None, rl_state_path=None,
                 use_llm_synthesis: bool = False, task: TaskSpec | None = None, registry=None, ev_config=None,
                 caps=None):
        if use_llm_synthesis and provider is None:
            raise ValueError('use_llm_synthesis needs a provider')
        self.provider = provider if use_llm_synthesis else _SeedOnlyProvider()
        self.use_llm_synthesis, self.rl_state_path = use_llm_synthesis, rl_state_path
        self.rl_judge = rl_judge or (DualLoopRLJudge.load(rl_state_path) if rl_state_path else DualLoopRLJudge())
        registry = registry or StrategyRegistry.from_dict(export_registry())
        self.repairer = OrchestratorRepairer(RepairEngine(task or TaskSpec.preset('retail')),
                                             directive_factory=RepairDirective)
        self.orchestrator = Orchestrator(env, verifier, EVPolicy(registry, ev_config, write_tools=env.write_tools),
                                         repairer=self.repairer, caps=caps)

    def run_episode(self, trajectory_ir: dict, trace, policy_str: str = '', tools: dict | None = None):
        """Returns (diagnosis result, orchestrator Outcome, reward or None when attribution produced no action)."""
        tid = str(trace.trajectory_id)
        result, rca = _diagnose(trajectory_ir, policy_str, tools or {}, self.provider, self.rl_judge,
                                self.use_llm_synthesis, explore=False)
        self.repairer.runs.pop(tid, None)  # each episode climbs the repair ladder from its first rung
        outcome = self.orchestrator.run(trace, RCAPayload.from_dict(rca))
        if tid not in self.rl_judge.pending:
            return result, outcome, None
        verified = outcome.status == RECOVERED
        before = (outcome.original_verifier or {}).get('score', 0.0)
        after = max((a.verifier['score'] for a in outcome.attempts if a.verifier), default=before)
        stopped = any(bool(a.stopped_early) for a in outcome.attempts) and not verified
        # Credit the candidate whose rationale fed the verified repair (detector candidates fall back to top-1).
        step = outcome.attempts[-1].option['primary']['step_index'] if verified else None
        step = step if step in {c['step_index'] for c in result['ranked_candidates']} else None
        failed = [a.option['primary']['step_index'] for a in outcome.attempts[:-1]] if verified else []
        telemetry = self.rl_judge.record_recovery_outcome(tid, step, verified=verified, verifier_score_before=before,
                                                          verifier_score_after=after, stopped_early=stopped,
                                                          failed_steps=failed)
        if self.rl_state_path:
            self.rl_judge.save(self.rl_state_path)
        return result, outcome, telemetry['reward']


def train_and_evaluate_benchmark(corpus_dir, train_split='train', test_split='test', epochs=1, rl_state_path=None,
                                 llm_factory=None, use_llm_synthesis=False, limit=None, provider=None) -> dict:
    """Learn theta and M_i on train_split from observable feedback only, then evaluate frozen on test_split.

    harness_recovery_rate in train_history is the harness's ledger-scored metric, reported for the learning
    curve and never fed back into the policy.
    """
    if llm_factory is None:
        from benchmark.env.scripted import scripted_llm as llm_factory
    policy = MPRLRecoveryPolicy(provider, rl_state_path=rl_state_path, use_llm_synthesis=use_llm_synthesis,
                                learn=True)
    history = []
    for epoch in range(1, epochs + 1):
        report = run_recovery(corpus_dir, train_split, policy, llm_factory, limit=limit)
        history.append({'epoch': epoch, 'episodes': policy.rl_judge.episodes,
                        'weights': list(policy.rl_judge.weights),
                        'harness_recovery_rate': report['metrics']['recovery_rate']['value']})
    policy.learn = False
    test_report = run_recovery(corpus_dir, test_split, policy, llm_factory, limit=limit)
    return {'train_history': history, 'test_report': test_report, 'learned_weights': list(policy.rl_judge.weights)}
