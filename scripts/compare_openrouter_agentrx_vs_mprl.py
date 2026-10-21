"""Budgeted, cached head-to-head on OpenRouter: original agentrx (protocol=joint) vs mp_rl + recovery.

    python scripts/compare_openrouter_agentrx_vs_mprl.py --n 6 --max-calls 90

Both systems see identical inputs: for retail benchmark episodes, the public output_only trace plus the policy and
tool schemas from the episode's task view; for external traces, the agentrx case (compacted). The full trace view
(per-step input histories, ~20x larger) only backs the RecoveryEnv replays, which run offline. Every model request
goes through one persistent CachedProvider, so reruns only pay for new requests.

Budget: --max-calls caps live HTTP requests (retries included) for the whole run. Semantic (LLM-judged) constraint
checks cost one request per guarded step, so each system gets at most --max-semantic of them per trace; the rest
are refused and surface as checker errors (reported). Gold labels and ledger rewards are read only by this
harness, after each system has produced its attribution / delivered its rollout.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentrx.io import read_json, write_json  # noqa: E402
from agentrx.pipeline import diagnose, diagnose_mp_rl  # noqa: E402
from agentrx.providers import CachedProvider, HTTPProvider, ProviderError  # noqa: E402
from agentrx.rl_cross_branch import _repairable, candidates_to_benchmark_hypotheses, trace_tools_and_policy  # noqa: E402,E501
from agentrx.rl_policy import DualLoopRLJudge  # noqa: E402
from agentrx.taxonomy import agent, category  # noqa: E402
from benchmark.corpus import load_gold, load_public  # noqa: E402
from benchmark.env.scripted import scripted_llm  # noqa: E402
from benchmark.env.workflow import Directive  # noqa: E402
from benchmark.scoring.baselines import heuristic_findings  # noqa: E402
from benchmark.scoring.recovery import BudgetCaps, RecoveryEnv, _EpisodeRunner  # noqa: E402
from benchmark.taxonomy import from_agentrx  # noqa: E402
from repair_strategies.adapters.benchmark import RepairStrategyPolicy  # noqa: E402

BASE_URL = 'https://openrouter.ai/api/v1'
MODEL = 'minimax/minimax-m3.1-flash:free'
CORPUS = ROOT/'runs'/'benchmark_smoke'/'corpus'
EXTERNAL = ROOT/'data'/'benchmark'/'cases.json'
MODES = ('context_propagation', 'drift', 'instruction_adherence', 'cycles', 'misinterpretation_of_tool_output',
         'tool_failure')
JOINT = {'protocol': 'joint', 'generation': 'oneshot', 'evidence': True, 'checklist': True, 'python_timeout': 3,
         'recovery_loop': False}
MP_RL = {'protocol': 'mp_rl', 'evidence': True, 'checklist': True, 'recovery_loop': False, 'python_timeout': 3}
PRIOR_CALLS_PER_TRACE = 14      # x + y estimate before any trace has been measured


# ------------------------------------------------------------------------------ budgeted providers
class LiveGate:
    """Innermost layer: hard ceiling on live HTTP requests, minimum spacing, 429 detection."""

    def __init__(self, http, max_calls, min_interval):
        self.http, self.max_calls, self.min_interval = http, max_calls, min_interval
        self.identity, self.measurement_kind = http.identity, http.measurement_kind
        self.forwarded, self.exhausted, self.rate_limited, self.failures = 0, False, False, []
        self._last = 0.0

    @property
    def last_raw(self):
        return self.http.last_raw

    @property
    def last_telemetry(self):
        return self.http.last_telemetry

    def complete(self, stage, payload):
        remaining = self.max_calls - self.http.http_requests
        if remaining <= 0:
            self.exhausted = True
            raise ProviderError('live call budget exhausted')
        self.http.retries = min(2, remaining - 1)          # retries can never overshoot the ceiling
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        self.forwarded += 1
        try:
            return self.http.complete(stage, payload)
        except ProviderError as exc:
            self.failures.append(f'{stage}: {exc}')
            self.rate_limited |= 'HTTP 429' in str(exc)
            raise


class Ledger:
    """Outermost layer, one per system: counts every model call and caps semantic checks per diagnosis pass."""

    def __init__(self, inner, semantic_cap):
        self.inner, self.semantic_cap = inner, semantic_cap
        self.identity, self.measurement_kind = inner.identity, inner.measurement_kind
        self.calls = self.refused = self._semantic = 0

    def begin(self):
        self._semantic = 0          # a recovery's re-diagnosis replays the same requests in the same order

    def complete(self, stage, payload):
        self.calls += 1
        if stage == 'semantic':
            self._semantic += 1
            if self._semantic > self.semantic_cap:
                self.refused += 1
                raise ProviderError('semantic check skipped: per-trace call cap')
        return self.inner.complete(stage, payload)


class Meter:
    """Per-trace deltas of one system's ledger and the shared live counter."""

    def __init__(self, ledger, http):
        self.ledger, self.http = ledger, http

    def __enter__(self):
        self.start = (self.ledger.calls, self.ledger.refused, self.http.http_requests, time.monotonic())
        return self

    def __exit__(self, *exc):
        calls, refused, live, t0 = self.start
        self.calls, self.refused = self.ledger.calls - calls, self.ledger.refused - refused
        self.live, self.seconds = self.http.http_requests - live, round(time.monotonic() - t0, 2)


# ------------------------------------------------------------------------------ trace selection
def compact(trajectory):
    """The trace without per-step input histories and system prompts (the full view is ~20x larger)."""
    out = {k: v for k, v in trajectory.items() if k != 'system_prompts'}
    out['steps'] = [{k: v for k, v in s.items() if k not in ('input', 'input_digest', 'state_version')}
                    for s in trajectory['steps']]
    return out


def retail_episodes(n):
    """n failed test episodes, round-robin over MODES so every mode is represented (sorted ids within a mode).

    trace_attr (output_only view) is what both systems attribute on; trace (full view) backs the RecoveryEnv.
    """
    gold, traces = load_gold(CORPUS, 'test'), load_public(CORPUS, 'test', 'full')
    traces_out = load_public(CORPUS, 'test', 'output_only')
    index = read_json(CORPUS/'gold'/'episodes.json')
    by_mode = {m: sorted(t for t, g in gold.items() if g['status'] == 'failed' and g['critical']['mode'] == m
                         and t in index) for m in MODES}
    picked = []
    while len(picked) < n and any(by_mode.values()):
        for m in MODES:
            if by_mode[m] and len(picked) < n:
                picked.append(by_mode[m].pop(0))
    return [{'kind': 'retail', 'id': t, 'gold': gold[t], 'trace': traces[t], 'trace_attr': traces_out[t],
             'ref': index[t]} for t in picked]


def external_cases():
    """The shortest tau and magentic case (data/benchmark has no split field; length keeps the budget bounded)."""
    cases = read_json(EXTERNAL)
    picked = []
    for domain in ('tau', 'magentic'):
        group = [c for c in cases if c['trajectory']['domain'] == domain]
        if group:
            picked.append(min(group, key=lambda c: (len(c['trajectory']['steps']), c['trajectory']['trajectory_id'])))
    return [{'kind': c['trajectory']['domain'], 'id': f"{c['trajectory']['domain']}/{c['trajectory']['trajectory_id']}",
             'gold': c['gold'], 'case': c} for c in picked]


def episode_env(item):
    ref = item['ref']
    gen_dir = Path(ref['generation_dir'])
    episode = read_json(gen_dir/'episodes'/f"{ref['episode_id']}.json")
    runner = _EpisodeRunner(episode, read_json(gen_dir/'generation.json')['config'], scripted_llm)
    env = RecoveryEnv(item['id'], item['trace'], runner.task_view(item['trace']), runner,
                      BudgetCaps().for_episode(episode['run']['tokens']))
    return env, runner


# ------------------------------------------------------------------------------ the two systems
def run_agentrx(item, inputs, ledger, http, artifacts):
    trajectory, policy, tools = inputs
    with Meter(ledger, http) as m:
        ledger.begin()
        result = diagnose(trajectory, policy, tools, JOINT, ledger, artifacts)
    pred = result.get('prediction') or {}
    row = {'pred': _pred(pred.get('step_index'), pred.get('agent'), pred.get('category')), 'status': result['status'],
           'checker_errors': len(result['errors']), 'calls': m.calls, 'live': m.live, 'semantic_refused': m.refused,
           'seconds': m.seconds}
    if row['pred'] and pred.get('category') != 'Inconclusive':
        row['pred']['mode'] = from_agentrx(pred['category'])[0]
    if item['kind'] == 'retail':
        env, runner = episode_env(item)
        if row['pred'] and row['pred'].get('step_index') is not None and pred.get('category') != 'Inconclusive':
            env.hypotheses = [_repairable({**row['pred'], 'confidence': 1.0})]
        policy_obj = RepairStrategyPolicy(directive_cls=Directive, max_hypotheses=1)
        row.update(_recover(policy_obj, env, runner, item['id']))
    return row


def run_mp_rl(item, inputs, ledger, http, artifacts):
    trajectory, policy, tools = inputs
    with Meter(ledger, http) as m:
        ledger.begin()
        result = diagnose_mp_rl(trajectory, policy, tools, MP_RL, ledger, artifacts, rl_judge=DualLoopRLJudge())
    ranked = result['ranked_candidates']
    top = candidates_to_benchmark_hypotheses(ranked, trajectory)[:3] if ranked else []
    row = {'pred': _pred(top[0]['step_index'], top[0]['agent'], top[0]['category']) if top else None,
           'top3': [{k: h[k] for k in ('step_index', 'agent', 'category', 'mode', 'confidence')} for h in top],
           'status': result['status'], 'checker_errors': len(result['errors']),
           'violations': len(result['violations']), 'calls': m.calls, 'live': m.live,
           'semantic_refused': m.refused, 'seconds': m.seconds}
    if top:
        row['pred']['mode'] = top[0]['mode']
    if item['kind'] == 'retail':
        env, runner = episode_env(item)
        # MPRLRecoveryPolicy.recover() minus its re-diagnosis: the same abstain rule as attribute(), then the
        # repair engine it subclasses, over the top-3 hypotheses computed above.
        if any(c.get('violated_checks') for c in ranked) or heuristic_findings(env.trace):
            env.hypotheses = [_repairable(h) for h in top]
        row['hypotheses'] = len(env.hypotheses)
        row.update(_recover(RepairStrategyPolicy(directive_cls=Directive, max_hypotheses=3), env, runner, item['id']))
    return row


def _pred(step, who, cat):
    return None if step is None and who is None else {'step_index': step, 'agent': who, 'category': cat}


def _recover(policy, env, runner, tid):
    t0 = time.monotonic()
    decision = policy.recover(env)
    attempts = policy.log[tid].attempts if tid in policy.log else []
    delivered = decision.get('delivered')
    # Harness side, after the policy returned: the gold-scored ledger of the delivered rollout.
    reward = runner.ledger[delivered]['reward'] if delivered else 0.0
    return {'decision': decision['status'], 'notes': decision.get('notes', ''),
            'accepted': any(a.get('accepted') for a in attempts), 'recovered': reward == 1.0,
            'replays': len(env.rollouts), 'recovery_seconds': round(time.monotonic() - t0, 2)}


# ------------------------------------------------------------------------------ scoring and report
def score(item, row):
    gold = item['gold']['critical']
    pred = row.get('pred') or {}
    step = pred.get('step_index')
    ok_steps = set(item['gold'].get('acceptable_steps') or [])
    if item['kind'] == 'retail':
        mode_ok = pred.get('mode') == gold['mode']
    else:
        mode_ok = bool(pred.get('category')) and category(pred['category']) == category(gold['category'])
    row['score'] = {'step_exact': step == gold['step_index'],
                    'step_tolerant': step is not None and (abs(step - gold['step_index']) <= 1 or step in ok_steps),
                    'agent': bool(pred.get('agent')) and agent(pred['agent']) == agent(gold['agent']),
                    'mode': mode_ok}
    if 'top3' in row:
        row['score']['step_hit_at_3'] = any(h['step_index'] == gold['step_index'] for h in row['top3'])


def _rate(rows, key):
    vals = [r['score'][key] for r in rows if key in r.get('score', {})]
    return (sum(vals), len(vals))


def summarize(results, gate, ledgers, args):
    a = [r['agentrx'] for r in results]
    b = [r['mp_rl'] for r in results]
    retail = [r for r in results if r['kind'] == 'retail']
    summary = {}
    for key, label in (('step_exact', 'Step exact accuracy'), ('step_tolerant', 'Step ±1 / acceptable accuracy'),
                       ('agent', 'Agent attribution accuracy'),
                       ('mode', 'Failure mode accuracy (retail mode / external category)')):
        summary[label] = {'agentrx': _rate(a, key), 'mp_rl': _rate(b, key)}
    summary['mp_rl step hit@3'] = {'agentrx': None, 'mp_rl': _rate(b, 'step_hit_at_3')}
    summary['Verified recovery on retail (ledger reward 1)'] = {
        'agentrx': (sum(r['agentrx'].get('recovered', False) for r in retail), len(retail)),
        'mp_rl': (sum(r['mp_rl'].get('recovered', False) for r in retail), len(retail))}
    mean = lambda rows, k: round(statistics.mean(r[k] for r in rows), 2) if rows else None  # noqa: E731
    summary['Mean LLM calls per trace (incl. cached)'] = {'agentrx': mean(a, 'calls'), 'mp_rl': mean(b, 'calls')}
    summary['Mean live HTTP requests per trace'] = {'agentrx': mean(a, 'live'), 'mp_rl': mean(b, 'live')}
    summary['Mean attribution latency per trace (s)'] = {'agentrx': mean(a, 'seconds'), 'mp_rl': mean(b, 'seconds')}
    forwarded = sum(l.calls - l.refused for l in ledgers.values())
    budget = {'max_calls': args.max_calls, 'live_http_requests': gate.http.http_requests,
              'model_calls_total': sum(l.calls for l in ledgers.values()), 'cache_hits': forwarded - gate.forwarded,
              'live_calls_forwarded': gate.forwarded, 'semantic_refused': sum(l.refused for l in ledgers.values()),
              'budget_exhausted': gate.exhausted, 'rate_limited': gate.rate_limited,
              'provider_failures': gate.failures[-10:]}
    return summary, budget


def _fmt_pred(r):
    p = r.get('pred') or {}
    if not p:
        return 'none'
    return f"{p.get('agent')}@{p.get('step_index')} {p.get('mode') or p.get('category')}"


def _fmt_rate(v):
    if v is None:
        return '—'
    if isinstance(v, tuple) or isinstance(v, list):
        return f'{v[0]}/{v[1]}' + (f' ({v[0] / v[1]:.0%})' if v[1] else '')
    return str(v)


def markdown(results, summary, budget, args, stopped):
    lines = [f'# OpenRouter head-to-head: agentrx (joint) vs mp_rl + recovery', '',
             f'Model `{args.model}` · max live requests {args.max_calls} · min interval {args.min_interval}s · '
             f'semantic checks per system per trace ≤ {args.max_semantic}', '']
    if stopped:
        lines += [f'**Stopped early:** {stopped}', '']
    lines += ['## Per trace', '',
              '| trace | mode | gold (agent, step) | agentrx pred | x calls (live) | s | rec | mp_rl pred | '
              'y calls (live) | s | rec |', '|' + '---|' * 11]
    for r in results:
        g, a, b = r['gold'], r['agentrx'], r['mp_rl']
        rec = lambda s: '—' if 'recovered' not in s else ('✔' if s['recovered'] else '✘')  # noqa: E731
        lines.append(f"| {r['id']} | {g.get('mode') or g.get('category')} | {g['agent']}, {g['step_index']} | "
                     f"{_fmt_pred(a)} | {a['calls']} ({a['live']}) | {a['seconds']} | {rec(a)} | {_fmt_pred(b)} | "
                     f"{b['calls']} ({b['live']}) | {b['seconds']} | {rec(b)} |")
    lines += ['', '## Aggregate', '', '| metric | agentrx (joint) | mp_rl |', '|---|---|---|']
    for label, v in summary.items():
        lines.append(f"| {label} | {_fmt_rate(v['agentrx'])} | {_fmt_rate(v['mp_rl'])} |")
    lines += ['', '## Budget telemetry', '', '| item | value |', '|---|---|']
    lines += [f'| {k} | {v} |' for k, v in budget.items() if k != 'provider_failures']
    if budget['provider_failures']:
        lines += ['', 'Last provider failures:', ''] + [f'- `{f}`' for f in budget['provider_failures']]
    lines += ['', 'Recovery: agentrx top-1 → RepairStrategyPolicy(max_hypotheses=1); mp_rl top-3 (frozen θ0, '
                  'MPRLRecoveryPolicy abstain rule) → RepairStrategyPolicy(max_hypotheses=3). "rec" = the delivered '
                  'rollout scores reward 1 on the harness ledger. Scripted agents replay the episodes offline; only '
                  'attribution calls the model.']
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------------------ main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n', type=int, default=6, help='retail benchmark episodes (stratified over modes)')
    parser.add_argument('--max-calls', type=int, default=90, help='hard ceiling on live HTTP requests')
    parser.add_argument('--min-interval', type=float, default=3.2, help='seconds between live requests')
    parser.add_argument('--max-semantic', type=int, default=3, help='semantic checks per system per trace')
    parser.add_argument('--include-external', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--out', default='runs/openrouter_comparison')
    args = parser.parse_args(argv)
    out = Path(args.out)
    # Unbounded, the :free MiniMax model spent 17k-41k reasoning tokens (4-8 minutes) on one synthesis call.
    http = HTTPProvider.from_config({'provider': 'openai-compatible', 'base_url': BASE_URL, 'model': args.model,
                                     'api_key_env': 'OPENROUTER_API_KEY',
                                     'settings': {'temperature': 0, 'reasoning_effort': 'low', 'max_tokens': 4096},
                                     'timeout': 120, 'retries': 2})
    http.retry_delay = 4                                   # backoff 4s, 8s: never faster than --min-interval
    gate = LiveGate(http, args.max_calls, args.min_interval)
    cached = CachedProvider(gate, out/'cache', {'study': 'openrouter_comparison', 'model': args.model})
    ledgers = {'agentrx': Ledger(cached, args.max_semantic), 'mp_rl': Ledger(cached, args.max_semantic)}
    items = retail_episodes(args.n) + (external_cases() if args.include_external else [])
    results, stopped = [], None
    for i, item in enumerate(items, 1):
        done = [r['agentrx']['live'] + r['mp_rl']['live'] + r['mp_rl'].get('recovery_live', 0) for r in results]
        estimate = round(statistics.mean(done)) if done else PRIOR_CALLS_PER_TRACE
        if item['kind'] != 'retail':            # new domain: its policy/tool prompts are not cached yet
            estimate = max(estimate, PRIOR_CALLS_PER_TRACE)
        if http.http_requests + estimate > args.max_calls:
            stopped = (f'budget: {http.http_requests} live requests used + ~{estimate} estimated for '
                       f'{item["id"]} > {args.max_calls}')
            break
        if gate.rate_limited or gate.exhausted:
            stopped = 'OpenRouter rate limit (HTTP 429 after retries)' if gate.rate_limited else 'budget exhausted'
            break
        if item['kind'] == 'retail':
            policy, tools = trace_tools_and_policy(episode_env(item)[0].task_view)   # what MPRLRecoveryPolicy reads
            inputs = (item['trace_attr'], policy, tools)
        else:
            inputs = (compact(item['case']['trajectory']), item['case']['policy'], item['case']['tools'])
        slug = item['id'].replace('/', '_')
        print(f'[{i}/{len(items)}] {item["id"]} ({item["kind"]}) live so far {http.http_requests}', flush=True)
        row = {'id': item['id'], 'kind': item['kind'], 'gold': {**item['gold']['critical'],
                                                                 'acceptable_steps': item['gold'].get('acceptable_steps')}}
        row['agentrx'] = run_agentrx(item, inputs, ledgers['agentrx'], http, out/'artifacts'/'agentrx'/slug)
        row['mp_rl'] = run_mp_rl(item, inputs, ledgers['mp_rl'], http, out/'artifacts'/'mp_rl'/slug)
        for system in ('agentrx', 'mp_rl'):
            score(item, row[system])
        print(f"    agentrx {_fmt_pred(row['agentrx'])} calls={row['agentrx']['calls']} live={row['agentrx']['live']}"
              f" rec={row['agentrx'].get('recovered', '-')} | mp_rl {_fmt_pred(row['mp_rl'])} "
              f"calls={row['mp_rl']['calls']} live={row['mp_rl']['live']} rec={row['mp_rl'].get('recovered', '-')}",
              flush=True)
        results.append(row)
    summary, budget = summarize(results, gate, ledgers, args)
    report = markdown(results, summary, budget, args, stopped)
    write_json(out/'report.json', {'model': args.model, 'args': vars(args), 'stopped': stopped, 'summary': summary,
                                   'budget': budget, 'traces': results})
    (out/'report.md').write_text(report, encoding='utf-8')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print('\n' + report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
