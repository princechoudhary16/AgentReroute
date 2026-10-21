"""Cross-branch integration check: orchestrator x repair-strategies x faultattribution.

Run from the repository root of the orchestrator branch, pointing at checkouts
(e.g. git worktrees) of the other two branches:

    python3 recovery_orchestrator/integration/cross_branch_check.py \
        --repair ../wt-repair --fa ../wt-fa --out runs/cross_branch

The FA checkout needs ``python3 -m agentrx prepare --fetch`` (gold cases) and
``python3 -m agentrx smoke`` (offline FA output). Nothing here calls an LLM.

This is an integration harness, not package code: the packages still never import
each other. It writes ``report.json`` and ``report.md`` to ``--out``.

Sections
  1. registry   repair's export_registry() validated against the orchestrator schema
  2. fa_adapter FA smoke output -> RCAPayload -> decision with the repair registry
  3. dry_run    73 gold-labelled traces (oracle attribution): orchestrator decision and the
                directive OrchestratorRepairer builds on the real trace (no replay)
  4. loop       full orchestrator loop on a synthetic tau trace with the real repairer and a
                scripted environment/verifier: learning hook, ladder advance, leakage
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path


def _commit(root: Path) -> str:
    try:
        return subprocess.run(['git', '-C', str(root), 'log', '-1', '--format=%h %s'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def check_registry(export_registry, StrategyRegistry, categories):
    data = export_registry()
    registry = StrategyRegistry.from_dict(data)          # raises on any schema violation
    coverage = {c: [s.id for s in registry.for_category(c)] for c in categories}
    return registry, {
        'strategies': len(registry),
        'kinds': dict(collections.Counter(s.kind for s in registry)),
        'irreversible': [s.id for s in registry if not s.reversible],
        'coverage': coverage,
        'uncovered': [c for c, ids in coverage.items() if not ids and c != 'Inconclusive'],
    }


def check_fa_adapter(fa_root: Path, registry, ro):
    run = fa_root / 'runs' / 'smoke' / 'experiment'
    if not run.is_dir():
        return {'skipped': f'{run} not found; run `python3 -m agentrx smoke` in the FA checkout'}
    rows = []
    for domain, tid in ro.rca.list_trajectories(run):
        payload = ro.rca.load_payload(run, tid, domain)
        trace = ro.trace.Trace.load_ir(ro.rca.find_runs(run, tid, domain)[0]['dir'] / 'trajectory_ir.json')
        policy = ro.policy.EVPolicy(registry, ro.policy.EVConfig(respect_irreversible=False),
                                    write_tools=ro.tau_retail.WRITE_TOOLS)
        decision = policy.decide(payload, trace, policy.make_guard())
        rows.append({'trajectory_id': tid, 'variants': list(payload.variants), 'total_runs': payload.total_runs,
                     'confidence': payload.confidence, 'top': payload.top and payload.top.category,
                     'action': decision.action,
                     'option': decision.option.describe() if decision.option else None})
    return {'run': str(run), 'traces': rows}


def dry_run(fa_root: Path, registry, ro, rs):
    cases_path = fa_root / 'data' / 'benchmark' / 'cases.json'
    if not cases_path.is_file():
        return {'skipped': f'{cases_path} not found; run `python3 -m agentrx prepare --fetch`'}
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    gold = ro.evaluation.load_gold(cases_path)
    rows = []
    for respect in (False, True):
        policy = ro.policy.EVPolicy(registry, ro.policy.EVConfig(respect_irreversible=respect),
                                    write_tools=ro.tau_retail.WRITE_TOOLS)
        for case in cases:
            trace = ro.trace.Trace.from_ir(case['trajectory'])
            label = gold[trace.trajectory_id]
            decision = policy.decide(ro.evaluation.oracle_payload(label), trace, policy.make_guard())
            row = {'respect_irreversible': respect, 'trajectory_id': trace.trajectory_id, 'domain': trace.domain,
                   'gold_step': label.critical.step_index, 'gold_category': label.critical.category,
                   'action': decision.action, 'reason': decision.reason}
            if decision.action == 'intervene':
                task = rs.TaskSpec.preset('retail') if trace.domain == 'tau' else rs.TaskSpec.from_dict({})
                repairer = rs.OrchestratorRepairer(rs.RepairEngine(task), ro.replay.RepairDirective)
                directive = repairer.build(decision.option.strategy, decision.option.primary, trace)
                lowered = repairer.last_lowered(trace.trajectory_id)
                row.update(rollback_step=decision.option.rollback_step, ev_strategy=decision.option.strategy.id,
                           applied_strategy=directive.strategy_id, directive_kind=directive.kind,
                           lowered=getattr(lowered, 'kind', None), message_chars=len(directive.message))
            rows.append(row)

    def summary(respect):
        group = [r for r in rows if r['respect_irreversible'] is respect]
        out = {}
        for domain in ('tau', 'magentic'):
            g = [r for r in group if r['domain'] == domain]
            acted = [r for r in g if r['action'] == 'intervene']
            out[domain] = {
                'traces': len(g),
                'actions': dict(collections.Counter(r['action'] for r in g)),
                'directive_kinds': dict(collections.Counter(r['directive_kind'] for r in acted)),
                'substituted': sum(r['applied_strategy'] != r['ev_strategy'] for r in acted),
                'not_executable': sorted({(r['gold_category'], r['applied_strategy'])
                                          for r in acted if r['directive_kind'] == 'none'}),
            }
        return out
    return {'summary': {'simulation (allow irreversible)': summary(False),
                        'deployment (respect irreversible)': summary(True)},
            'rows': [r for r in rows if not r['respect_irreversible']]}


def check_loop(registry, ro, rs, helpers):
    """Real repairer in the real loop; replay and verifier are scripted (no LLM)."""
    attempts_seen = []

    def script(trace, prefix, directive):
        attempts_seen.append(directive.strategy_id)
        start = prefix.rollback_step
        if len(attempts_seen) >= 2 and directive.kind != 'none':     # second repair "works"
            return helpers.good_continuation(start), 1.0, 2000
        return helpers.bad_continuation(start), 0.0, 2000

    engine = rs.RepairEngine(rs.TaskSpec.preset('retail'))
    repairer = rs.OrchestratorRepairer(engine, ro.replay.RepairDirective)
    env = helpers.make_env(script=script)
    verifier, llm = helpers.make_verifier()
    policy = ro.policy.EVPolicy(registry, ro.policy.EVConfig(respect_irreversible=False),
                                write_tools=ro.tau_retail.WRITE_TOOLS)
    trace = helpers.syn_trace()
    payload = ro.rca.load_payload(helpers.SYN_RUN, helpers.SYN_ID, 'tau')
    outcome = ro.orchestrator.Orchestrator(env, verifier, policy, repairer=repairer).run(trace, payload)
    history = repairer.runs[helpers.SYN_ID].history
    seen = json.dumps(llm.requests) + json.dumps([c['directive'].message for c in env.calls])
    return {
        'status': outcome.status,
        'attempts': [{'ev_strategy': a.option['strategy'], 'applied': a.applied_strategy,
                      'directive': a.directive_kind, 'verified': a.verifier['passed']} for a in outcome.attempts],
        'repairer_history': [{'strategy': r.strategy_id, 'outcome': r.outcome} for r in history],
        'record_outcome_called': len(history) == len(outcome.attempts),
        'gold_sentinel_leaked': 'GOLDSENTINEL' in seen,
        'reward_in_verifier_or_directives': 'reward' in seen,
    }


def render_md(report: dict) -> str:
    lines = ['# Cross-branch integration check', '']
    for name, commit in report['commits'].items():
        lines.append(f'- `{name}`: {commit}')
    reg = report['registry']
    lines += ['', '## 1. Repair registry → orchestrator schema',
              f"Valid. {reg['strategies']} strategies, kinds {reg['kinds']}. Uncovered categories: "
              f"{reg['uncovered'] or 'none'}. Irreversible: {reg['irreversible']}."]
    fa = report['fa_adapter']
    lines += ['', '## 2. FA smoke output → payload → decision']
    lines += [fa['skipped']] if 'skipped' in fa else [
        f"- `{r['trajectory_id']}`: {r['total_runs']} runs ({', '.join(r['variants'])}), confidence "
        f"{r['confidence']:.2f}, top {r['top']} → **{r['action']}**"
        + (f" at step {r['option']['rollback_step']} with `{r['option']['strategy']}`" if r['option'] else '')
        for r in fa['traces']]
    dr = report['dry_run']
    lines += ['', '## 3. Dry run on 73 gold traces (oracle attribution, no replay)']
    if 'skipped' in dr:
        lines.append(dr['skipped'])
    else:
        for mode, by_domain in dr['summary'].items():
            lines += ['', f'**{mode}**', '', '| | traces | actions | directive kinds | substituted | not executable |',
                      '|---|---|---|---|---|---|']
            for domain, s in by_domain.items():
                ne = '; '.join(f'{c}→{sid}' for c, sid in s['not_executable']) or '—'
                lines.append(f"| {domain} | {s['traces']} | {s['actions']} | {s['directive_kinds']} | "
                             f"{s['substituted']} | {ne} |")
    lp = report['loop']
    lines += ['', '## 4. Full loop with the real repairer (scripted replay/verifier)',
              f"Status **{lp['status']}**; attempts: "
              + ', '.join(f"{a['applied']} ({'pass' if a['verified'] else 'fail'})" for a in lp['attempts'])
              + f". record_outcome called for every attempt: {lp['record_outcome_called']}. "
              f"Gold leaked: {lp['gold_sentinel_leaked']}; reward visible: {lp['reward_in_verifier_or_directives']}."]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--repair', required=True, type=Path, help='repair-strategies checkout root')
    parser.add_argument('--fa', required=True, type=Path, help='faultattribution checkout root')
    parser.add_argument('--out', type=Path, default=Path('runs/cross_branch'))
    args = parser.parse_args(argv)
    here = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(here), str(args.repair.resolve())]

    import recovery_orchestrator.evaluation
    import recovery_orchestrator.orchestrator
    import recovery_orchestrator.policy
    import recovery_orchestrator.rca
    import recovery_orchestrator.replay
    import recovery_orchestrator.tau_retail
    import recovery_orchestrator.trace
    from recovery_orchestrator.registry import StrategyRegistry
    from recovery_orchestrator.taxonomy import CATEGORIES, CYCLE
    from recovery_orchestrator.tests import helpers
    from repair_strategies.adapters.orchestrator import OrchestratorRepairer, export_registry
    from repair_strategies.context import TaskSpec
    from repair_strategies.engine import RepairEngine
    import types
    ro = sys.modules['recovery_orchestrator']
    rs = types.SimpleNamespace(OrchestratorRepairer=OrchestratorRepairer, RepairEngine=RepairEngine,
                               TaskSpec=TaskSpec)

    registry, reg_report = check_registry(export_registry, StrategyRegistry, list(CATEGORIES) + [CYCLE])
    report = {
        'commits': {'orchestrator': _commit(here), 'repair-strategies': _commit(args.repair),
                    'faultattribution': _commit(args.fa)},
        'registry': reg_report,
        'fa_adapter': check_fa_adapter(args.fa, registry, ro),
        'dry_run': dry_run(args.fa, registry, ro, rs),
        'loop': check_loop(registry, ro, rs, helpers),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, default=list) + '\n', encoding='utf-8')
    (args.out / 'registry_from_repair.json').write_text(json.dumps(export_registry(), indent=2) + '\n',
                                                        encoding='utf-8')
    (args.out / 'report.md').write_text(render_md(report), encoding='utf-8')
    print(render_md(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
