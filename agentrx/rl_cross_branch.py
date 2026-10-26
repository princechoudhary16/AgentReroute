"""Stage B closed loop: diagnose_mp_rl -> repair_strategies -> benchmark / orchestrator replay -> theta and M_i.

Rewards come only from observable verifiers: repair_strategies acceptance plus rollout termination on the
benchmark path, the orchestrator's fresh-context VerifierResult on the other. Nothing here reads
runner.ledger, episode rewards, gold_state_hash or benchmark/corpus/gold.

Gold-reading reporting (validation attribution exact-step) lives in ``rl_reporting`` and is called lazily;
the acceptance-vs-ledger gap logged per epoch is assembled from the harness's finished report and the
policy's observable log, never handed back to the policy.
"""
import collections
import math
import sys
import tempfile
import time
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
from .perspectives import repeats_earlier_tool_result
from .pipeline import diagnose_mp_rl
from .providers import ProviderError
from .rl_policy import THETA0, DualLoopRLJudge


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
    ladder: a grounding violation on a handoff is context_propagation, and a stuck/repeated tool loop is a Cycle.

    The loop test is structural (an identical call returning an identical result) rather than a match on
    the injector's own "status": "processing" payload, which is what the seed layer used to key on."""
    names = {(v['assertion_name'], v['step_index']) for v in c.get('violated_checks', ())}
    step = steps.get(c['step_index'])
    if step and any(n == 'p3_seed_handoff_and_fact_grounding' for n, _ in names) \
            and any(e.get('name') == 'handoff' for e in step['substeps']):
        return 'MisinterpretationOfToolOutput', 'context_propagation'
    for name, index in names:
        events = steps[index]['substeps'] if index in steps else []
        if (name == 'p2_seed_valid_tool_invocation' and repeats_earlier_tool_result(steps, index)) or (
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
        # Frozen evaluation (learn=False) loads the checkpoint read-only so recover() cannot write it back.
        self.rl_judge = rl_judge or (DualLoopRLJudge.load(rl_state_path, read_only=not learn) if rl_state_path
                                     else DualLoopRLJudge(read_only=not learn))
        self.engine, self.hypotheses, self.rl_log = None, {}, {}
        self.epoch = None          # set by the training loop; None means do not record consumption

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
        # Which acceptance check rejected the repair. AcceptanceReport.passed is a conjunction over up
        # to 11 kinds, so a run sitting at accept 0.00 is uninformative without knowing which one fires.
        rejected = [kind for a in self.log[tid].attempts for kind in a.get('rejected_by', [])]
        self.rl_log[tid] = {'verified': verified, 'score_after': after, 'stopped_early': stopped,
                            'credited_step': step, 'failed_steps': telemetry['pairwise_steps'],
                            'reward': telemetry['reward'], 'rejected_by': rejected}
        if self.epoch is not None:
            # Recorded before the save, into the dict that save() serializes, so theta and the ledger of
            # what produced it land in one atomic write and cannot disagree after an interruption. Only
            # this path records: the early returns above made no policy update, so replaying those
            # episodes on resume costs time but cannot double-count a gradient.
            consumed = self.rl_judge.progress.setdefault('epoch_consumed', {}).setdefault(str(self.epoch), [])
            if tid not in consumed:
                consumed.append(tid)
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


def _observable_epoch_metrics(policy):
    """Policy-visible training signals for one epoch, from the policy's own observable log (never the ledger).

    acceptance_rate and mean_reward are over episodes the policy acted on (abstentions are not logged)."""
    rl = list(policy.rl_log.values())
    n = len(rl)
    return {'n_acted': n,
            'acceptance_rate': (sum(1.0 for r in rl if r['verified']) / n) if n else 0.0,
            'mean_reward': (sum(r['reward'] for r in rl) / n) if n else 0.0}


class ReachedTarget(Exception):
    """Raised from the per-episode callback to stop training once validation clears the target."""

    def __init__(self, score, episodes):
        super().__init__(f'validation {score:.4f} cleared the target after {episodes} episode(s)')
        self.score, self.episodes = score, episodes


def train_and_evaluate_benchmark(corpus_dir, train_split='train', test_split='test', epochs=1, rl_state_path=None,
                                 llm_factory=None, use_llm_synthesis=False, limit=None, provider=None,
                                 val_split=None, control_sample=25, stop_above=None, val_every=0,
                                 val_limit=None, show_progress=True) -> dict:
    """Learn theta and M_i on train_split from observable feedback only, then evaluate frozen on test_split.

    Per epoch, train_history logs the observable signals (mean_reward, acceptance_rate) alongside the gold
    harness_recovery_rate and their acceptance_minus_ledger_gap - the reward-hacking alarm: observable
    acceptance climbing away from gold recovery. The gap is assembled here in the reporting layer, after the
    policy returns; the ledger is never fed back into the policy. With val_split set, also logs frozen
    attribution exact-step on it for model selection.

    ``control_sample`` caps how many clean/tolerated episodes training sees. Attribution runs full
    constraint synthesis before it decides to abstain, so each control costs a whole episode of LLM
    calls and usually returns no gradient; on this corpus that is 120 of 233 train episodes. A sample
    is kept rather than none, because controls carry the only signal against intervening on a healthy
    run. The frozen test evaluation always scores every control.

    Resume: the checkpoint carries a ledger of which episodes each epoch has already trained on, written
    in the same atomic save as theta. A restart skips completed epochs outright and, inside a partly
    finished epoch, skips the episodes whose gradients theta already holds. Without that ledger a restart
    silently applied them a second time. A resumed epoch's aggregate metrics therefore cover only the
    episodes that process ran, and the row says so in resumed_from_episode; the authoritative numbers come
    from the frozen test evaluation, which always runs its whole split in one pass. ``epochs`` is left out
    of the resume identity on purpose, so raising it continues an existing run rather than being refused.
    """
    if llm_factory is None:
        from benchmark.env.scripted import scripted_llm as llm_factory
    from .rl_reporting import attribution_exact_step    # reporting-side (reads gold); kept off the policy path
    policy = MPRLRecoveryPolicy(provider, rl_state_path=rl_state_path, use_llm_synthesis=use_llm_synthesis,
                                learn=True)
    progress = policy.rl_judge.progress
    identity = {'corpus': str(corpus_dir), 'train': train_split, 'val': val_split, 'test': test_split,
                'limit': limit, 'use_llm_synthesis': bool(use_llm_synthesis),
                'control_sample': control_sample}
    recorded = progress.get('identity')
    if recorded is not None and recorded != identity:
        raise ValueError(f'this checkpoint was trained on different settings ({recorded} != {identity}); '
                         'resuming would fold two different runs into one theta. Use a new checkpoint path.')
    progress['identity'] = identity
    history = list(progress.get('history', []))
    done = {row['epoch'] for row in history}
    for epoch in range(1, epochs + 1):
        if epoch in done:
            continue                        # this epoch's gradients are already in theta
        policy.epoch = epoch
        consumed = progress.setdefault('epoch_consumed', {}).setdefault(str(epoch), [])
        resumed_from = len(consumed)
        started_at = time.monotonic()

        def episode_line(done_n, total, tid, row, _start=started_at):
            """Live per-episode telemetry, and the validation early stop.

            A training run here is hours long, so reporting only at the end means a corrupted reward
            signal or a stalled policy is discovered after the compute is spent rather than in the
            first few minutes.
            """
            rl = list(policy.rl_log.values())
            mean_reward = sum(r['reward'] for r in rl) / len(rl) if rl else 0.0
            accepted = sum(1.0 for r in rl if r['verified']) / len(rl) if rl else 0.0
            drift = math.sqrt(sum((w - t) ** 2 for w, t in zip(policy.rl_judge.weights, THETA0)))
            elapsed = time.monotonic() - _start
            per_episode = elapsed / max(done_n, 1)
            rejects = collections.Counter(k for r in rl for k in r.get('rejected_by', ()))
            worst = f' | rejected_by {rejects.most_common(1)[0][0]}' if rejects else ''
            if show_progress:
                print(f'  [ep {done_n:>3}/{total}] {elapsed / 60:5.1f}m  {per_episode:5.1f}s/ep  '
                      f'eta {per_episode * (total - done_n) / 60:5.1f}m | R {mean_reward:+.3f} | '
                      f'accept {accepted:.2f} | |dtheta| {drift:.4f}{worst} | {tid}',
                      file=sys.stderr, flush=True)
            if val_every and stop_above is not None and val_split and done_n % val_every == 0:
                score = attribution_exact_step(corpus_dir, val_split, policy.rl_judge, use_llm_synthesis,
                                               provider, llm_factory, val_limit)
                if show_progress:
                    print(f'  [val @ {done_n}] exact_step={score} target>{stop_above}',
                          file=sys.stderr, flush=True)
                if score is not None and score > stop_above:
                    raise ReachedTarget(score, done_n)

        reached = None
        try:
            report = run_recovery(corpus_dir, train_split, policy, llm_factory, limit=limit, skip=consumed,
                                  control_sample=control_sample, on_episode=episode_line)
        except ReachedTarget as hit:
            # The consumed ledger was written per episode, so theta and the record of what produced it
            # are already consistent; the epoch simply has no aggregate report.
            reached, report = hit, None
        observable = _observable_epoch_metrics(policy)
        ledger_recovery_rate = report['metrics']['recovery_rate']['value'] if report else None
        history.append({'epoch': epoch, 'episodes': policy.rl_judge.episodes,
                        'weights': list(policy.rl_judge.weights),
                        'harness_recovery_rate': ledger_recovery_rate, **observable,
                        # None when the epoch ran no episodes at all, which a resume can hit if the last
                        # episode was saved before the epoch itself was recorded as done.
                        'acceptance_minus_ledger_gap': (None if ledger_recovery_rate is None else
                                                        observable['acceptance_rate'] - ledger_recovery_rate),
                        'resumed_from_episode': resumed_from,
                        'stopped_on_target': None if reached is None else
                                             {'score': reached.score, 'after_episodes': reached.episodes},
                        'val_attribution_exact_step': (
                            reached.score if reached is not None else
                            attribution_exact_step(corpus_dir, val_split, policy.rl_judge, use_llm_synthesis,
                                                    provider, llm_factory, limit) if val_split else None)})
        progress['history'] = history
        if rl_state_path:
            policy.rl_judge.save(rl_state_path)
        if reached is not None:
            if show_progress:
                print(f'  stopping: {reached}', file=sys.stderr, flush=True)
            break
    policy.epoch = None
    policy.learn = False
    test_report = run_recovery(corpus_dir, test_split, policy, llm_factory, limit=limit)
    return {'train_history': history, 'test_report': test_report, 'learned_weights': list(policy.rl_judge.weights)}
