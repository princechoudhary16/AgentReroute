"""Recovery harness: replayable failed episodes, a policy interface, baselines and metrics.

A recovery policy receives a ``RecoveryEnv`` for one episode. It can read the
public trace, the task view and (optionally) attribution hypotheses, and it can
call ``replay(Directive)`` or ``restart()``. Each call re-executes the workflow:
the prefix before the rollback step is reproduced from the response cache (no
new cost), the directive is applied, and the rest runs live. The policy finally
returns which rollout to deliver.

Rewards are written by the environment into a ledger the policy is never
given; only the harness reads it after the policy returns.

The fault stays in the world during replays (see ``faults.operators``):
environment faults fire again with their transient/persistent semantics, and
agent faults recur whenever the faulty agent sees the exact same context.
"""
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..corpus import load_gold, load_public
from ..env.retail import TOOL_SCHEMAS, gold_state_hash
from ..env.tasks import load_domain
from ..env.workflow import AGENTS, Directive, Workflow, export_trace, outcome
from ..faults.operators import build_operator
from ..io import read_json
from .baselines import heuristic_findings
from .stats import mcnemar, mean, percentile, proportion


@dataclass
class BudgetCaps:
    tokens: int = 60000
    attempts: int = 3
    seconds: float = 600.0


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Rollout:
    rollout_id: str
    directive: dict
    trace: dict
    tokens: int
    termination: str
    replayed_steps: int


@dataclass
class _Usage:
    tokens: int = 0
    attempts: int = 0
    breaches: list = field(default_factory=list)


class RecoveryEnv:
    """What a recovery policy may see and do for one failed episode (no gold)."""

    def __init__(self, trajectory_id, trace, task_view, runner, caps, hypotheses=None):
        self.trajectory_id = trajectory_id
        self.trace = trace                    # public full view of the failed run
        self.task_view = task_view            # first user message, policy, agents, tool schemas
        self.hypotheses = hypotheses or []    # attribution predictions, if the harness was given any
        self.caps = caps
        self._runner = runner
        self._usage = _Usage()
        self._started = time.perf_counter()
        self._ids = itertools.count(1)
        self.rollouts = {}

    @property
    def spent(self):
        return {'tokens': self._usage.tokens, 'attempts': self._usage.attempts,
                'seconds': round(time.perf_counter() - self._started, 3)}

    def remaining(self):
        return {'tokens': self.caps.tokens - self._usage.tokens, 'attempts': self.caps.attempts - self._usage.attempts,
                'seconds': self.caps.seconds - (time.perf_counter() - self._started)}

    def _check(self):
        left = self.remaining()
        for key in ('tokens', 'attempts', 'seconds'):
            if left[key] <= 0:
                raise BudgetExceeded(f'{key} budget exhausted')

    def replay(self, directive):
        """Roll back to ``directive.rollback_step`` and continue with the directive applied."""
        self._check()
        self._usage.attempts += 1
        rid = f'r{next(self._ids)}'
        run, trace = self._runner(rid, directive)
        self._usage.tokens += run.new_tokens
        if self._usage.tokens > self.caps.tokens:
            self._usage.breaches.append('tokens')
        rollout = Rollout(rid, directive.__dict__.copy(), trace, run.new_tokens, run.termination,
                          sum(1 for s in run.steps if s.replayed))
        self.rollouts[rid] = rollout
        return rollout

    def restart(self, seed_offset=1):
        """Full restart from the first step with a fresh seed (resample)."""
        return self.replay(Directive(rollback_step=1, kind='resample', seed_offset=seed_offset))


class RecoveryPolicy:
    name = 'policy'

    def recover(self, env):
        """Return {'delivered': rollout_id | None, 'status': str, 'notes': str}."""
        raise NotImplementedError


def looks_ok(rollout):
    """A blind, observable-only acceptance check used by the baselines (not a gold verifier)."""
    later = {**rollout.trace, 'steps': [s for s in rollout.trace['steps']
                                        if s['index'] >= rollout.directive['rollback_step']]}
    return rollout.termination == 'user_stop' and not heuristic_findings(later)


class NoRecovery(RecoveryPolicy):
    name = 'no_recovery'

    def recover(self, env):
        return {'delivered': None, 'status': 'abstained', 'notes': 'baseline: never intervene'}


class FullRestart(RecoveryPolicy):
    name = 'full_restart'

    def recover(self, env):
        rollout = env.restart(seed_offset=1)
        return {'delivered': rollout.rollout_id, 'status': 'intervened', 'notes': 'blind full restart'}


class RetryLastStep(RecoveryPolicy):
    """Capped retry: resample from the last agent step, accept the first rollout that looks ok."""
    name = 'retry_last_step'

    def __init__(self, retries=2):
        self.retries = retries

    def recover(self, env):
        steps = [s['index'] for s in env.trace['steps'] if s['substeps'][0]['role'] in AGENTS]
        if not steps:
            return {'delivered': None, 'status': 'abstained', 'notes': 'no agent step'}
        last = None
        for attempt in range(1, self.retries + 1):
            try:
                last = env.replay(Directive(rollback_step=steps[-1], kind='resample', seed_offset=attempt))
            except BudgetExceeded:
                break
            if looks_ok(last):
                return {'delivered': last.rollout_id, 'status': 'intervened', 'notes': f'accepted attempt {attempt}'}
        return {'delivered': last.rollout_id if last else None, 'status': 'intervened' if last else 'aborted',
                'notes': 'no attempt passed the check'}


GENERIC_GUIDANCE = ('Review the conversation carefully before acting. Follow the policy, check every tool '
                    'output, pass complete context, and give the user every requested detail.')


class GenericReflection(RecoveryPolicy):
    """Fault-blind reflection: generic guidance from the first agent step (Reflexion-style)."""
    name = 'generic_reflection'

    def recover(self, env):
        first = next((s['index'] for s in env.trace['steps'] if s['substeps'][0]['role'] in AGENTS), 1)
        rollout = env.replay(Directive(rollback_step=first, kind='guidance', text=GENERIC_GUIDANCE))
        return {'delivered': rollout.rollout_id, 'status': 'intervened', 'notes': 'generic guidance at first step'}


MODE_GUIDANCE = {
    'instruction_adherence': 'Follow the policy exactly: perform the confirmed database action with the proper '
                             'tool, and never report success without calling it.',
    'drift': 'Stay on the user goal: route to the specialist that owns the needed tool and call only relevant tools.',
    'cycles': 'Do not repeat the same call. If a tool keeps answering "processing", stop and report the problem.',
    'tool_failure': 'The last tool output may be an error or stale. Call the tool again and check the values '
                    'before acting; report the failure if it persists.',
    'misinterpretation_of_tool_output': 'Re-read the latest tool output and use the exact values (ids, amounts, '
                                        'counts) it contains.',
    'context_propagation': 'Include every identifier the specialist needs (user id, order id, item ids, payment '
                           'method) in the handoff message and context.',
    'missing_details': 'Include every requested fact (amounts, counts, ids) in your answer to the user.',
}


class FixedRules(RecoveryPolicy):
    """Fault-aware fixed rules: roll back to each hypothesised step and apply a mode-specific directive."""
    name = 'fixed_rules'
    RESAMPLE_MODES = ()

    def __init__(self, max_hypotheses=2):
        self.max_hypotheses = max_hypotheses

    def recover(self, env):
        if not env.hypotheses:
            return {'delivered': None, 'status': 'abstained', 'notes': 'no attribution available'}
        last = None
        for h in env.hypotheses[:self.max_hypotheses]:
            if h.get('step_index') is None or h.get('mode') not in MODE_GUIDANCE:
                continue
            directive = Directive(rollback_step=max(1, h['step_index']), kind='guidance',
                                  text=MODE_GUIDANCE[h['mode']])
            try:
                last = env.replay(directive)
            except BudgetExceeded:
                break
            if looks_ok(last):
                return {'delivered': last.rollout_id, 'status': 'intervened',
                        'notes': f'accepted rollback to step {h["step_index"]} ({h["mode"]})'}
        return {'delivered': last.rollout_id if last else None, 'status': 'intervened' if last else 'abstained',
                'notes': 'no rollout passed the check'}


POLICIES = {cls.name: cls for cls in (NoRecovery, FullRestart, RetryLastStep, GenericReflection, FixedRules)}


# ============================================================================== harness
class _EpisodeRunner:
    """Re-executes one generation episode with directives; keeps the reward ledger."""

    def __init__(self, episode, gen_config, llm_factory):
        self.episode = episode
        self.domain = load_domain(gen_config['data_dir'])
        self.task = self.domain.task(episode['task_id'])
        self.gold_hash = gold_state_hash(self.domain.data, self.task['actions'])
        self.llm_factory = llm_factory
        self.ledger = {}
        self.runs = {}

    def _faults(self):
        if not self.episode['operator']:
            return []
        op = build_operator(self.episode['operator'])
        op.fired = list(self.episode['run']['faults_fired'])
        return [op.replay_copy()]

    def __call__(self, rollout_id, directive):
        ep = self.episode
        run = Workflow(self.domain, self.task, self.llm_factory(self.domain, self.task), ep['model'], seed=ep['seed'],
                       temperature=ep['temperature'], faults=self._faults(), directive=directive,
                       max_steps=ep['max_steps']).run()
        self.ledger[rollout_id] = outcome(self.task, run, self.gold_hash)
        self.runs[rollout_id] = run
        first_user = next((s.content for s in run.steps if s.kind == 'user'), '')
        return run, export_trace(run, ep['trajectory_id'], ep['domain'], first_user, 'full')

    def task_view(self, trace):
        return {'first_user_message': trace['instruction'], 'policy': self.domain.policy, 'agents': sorted(AGENTS),
                'tools': {a: [TOOL_SCHEMAS[t] for t in spec['tools']] for a, spec in AGENTS.items()}}


def oracle_hypotheses(gold):
    if gold['status'] != 'failed':
        return []
    c = gold['critical']
    return [{'step_index': c['step_index'], 'agent': c['agent'], 'mode': c['mode'], 'confidence': 1.0}]


def run_recovery(corpus_dir, split, policy, llm_factory, caps=None, hypotheses=None, oracle=False,
                 include_controls=True, limit=None):
    """Evaluate ``policy`` on the replayable episodes of ``split``.

    ``hypotheses``: {tid: [hypothesis, ...]} from an attribution submission.
    ``oracle``: pass gold attribution instead (upper bound condition).
    """
    corpus_dir = Path(corpus_dir)
    caps = caps or BudgetCaps()
    gold = load_gold(corpus_dir, split)
    traces = load_public(corpus_dir, split, 'full')
    index = read_json(corpus_dir / 'gold' / 'episodes.json')
    rows = []
    selected = [tid for tid in sorted(gold) if tid in index
                and (gold[tid]['status'] == 'failed' or include_controls)]
    if limit:
        selected = selected[:limit]
    for tid in selected:
        g = gold[tid]
        ref = index[tid]
        gen_dir = Path(ref['generation_dir'])
        episode = read_json(gen_dir / 'episodes' / f'{ref["episode_id"]}.json')
        gen_config = read_json(gen_dir / 'generation.json')['config']
        runner = _EpisodeRunner(episode, gen_config, llm_factory)
        hyps = oracle_hypotheses(g) if oracle else (hypotheses or {}).get(tid, [])
        env = RecoveryEnv(tid, traces[tid], runner.task_view(traces[tid]), runner, caps, hyps)
        started = time.perf_counter()
        try:
            decision = policy.recover(env)
        except BudgetExceeded as exc:
            decision = {'delivered': None, 'status': 'aborted', 'notes': str(exc)}
        seconds = time.perf_counter() - started
        original = episode['outcome']['reward']
        delivered = decision.get('delivered')
        if delivered is not None and delivered not in runner.ledger:
            raise ValueError(f'{policy.name} delivered unknown rollout {delivered}')
        result = runner.ledger[delivered] if delivered else episode['outcome']
        delivered_run = runner.runs.get(delivered)
        gold_writes = {(a['name'], repr(sorted(a['kwargs'].items()))) for a in runner.task['actions']}
        collateral = 0
        if delivered_run is not None:
            collateral = sum(1 for w in delivered_run.write_calls if w['ok'] and
                             (w['tool'], repr(sorted(w['arguments'].items()))) not in gold_writes)
        rows.append({'trajectory_id': tid, 'status': g['status'], 'mode': (g['critical'] or {}).get('mode'),
                     'level': (g.get('injection') or {}).get('level'), 'original_reward': original,
                     'delivered_reward': result['reward'], 'intervened': env.spent['attempts'] > 0,
                     'decision': decision.get('status'), 'notes': decision.get('notes', ''),
                     'attempts': env.spent['attempts'], 'tokens': env.spent['tokens'],
                     'budget_breach': bool(env._usage.breaches), 'seconds': round(seconds, 3),
                     'collateral_writes': collateral})
    return summarize(policy.name + (':oracle' if oracle else ''), split, rows, caps)


def summarize(name, split, rows, caps):
    failed = [r for r in rows if r['status'] == 'failed']
    controls = [r for r in rows if r['status'] != 'failed']
    recovered = [r for r in failed if r['delivered_reward'] == 1]
    intervened_failed = [r for r in failed if r['intervened']]
    by_mode = {}
    for r in failed:
        by_mode.setdefault(r['mode'] or 'unlabeled', []).append(r)
    metrics = {
        'recovery_rate': proportion([r['delivered_reward'] == 1 for r in failed], len(failed)),
        'conditional_recovery_rate': proportion([r['delivered_reward'] == 1 for r in intervened_failed],
                                                len(intervened_failed)),
        'intervention_rate': proportion([r['intervened'] for r in failed], len(failed)),
        'harmful_intervention_rate': proportion([r['delivered_reward'] < r['original_reward'] for r in controls],
                                                len(controls)),
        'false_intervention_rate': proportion([r['intervened'] for r in controls], len(controls)),
        'collateral_writes_per_episode': mean(r['collateral_writes'] for r in rows),
        'tokens_per_episode': mean(r['tokens'] for r in failed),
        'tokens_per_recovered': (sum(r['tokens'] for r in failed) / len(recovered)) if recovered else None,
        'attempts_per_episode': mean(r['attempts'] for r in failed),
        'p95_seconds': percentile([r['seconds'] for r in rows], 0.95),
        'budget_breaches': sum(r['budget_breach'] for r in rows),
    }
    return {'policy': name, 'split': split, 'caps': caps.__dict__, 'n_failed': len(failed),
            'n_controls': len(controls), 'metrics': metrics,
            'by_mode': {m: {'n': len(rs), 'recovery_rate': mean(1.0 if r['delivered_reward'] == 1 else 0.0 for r in rs),
                            'tokens': mean(r['tokens'] for r in rs)} for m, rs in sorted(by_mode.items())},
            'per_episode': rows}


def compare(report_a, report_b):
    """Paired comparison of two recovery reports on the same failed episodes (McNemar)."""
    a = {r['trajectory_id']: r for r in report_a['per_episode'] if r['status'] == 'failed'}
    b = {r['trajectory_id']: r for r in report_b['per_episode'] if r['status'] == 'failed'}
    shared = sorted(set(a) & set(b))
    test = mcnemar([a[t]['delivered_reward'] == 1 for t in shared], [b[t]['delivered_reward'] == 1 for t in shared])
    return {'a': report_a['policy'], 'b': report_b['policy'], 'n_paired': len(shared), **test}
