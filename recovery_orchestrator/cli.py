"""Command line: build RCA payloads from FA output, inspect decisions, evaluate offline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .policy import BaselinePolicy, EVConfig, EVPolicy
from .rca import find_runs, list_trajectories, load_payload, trace_dir_name
from .registry import StrategyRegistry, default_registry, mock_registry
from .tau_retail import WRITE_TOOLS
from .trace import Trace


def _registry(path):
    if path == 'mock':
        return mock_registry()
    return StrategyRegistry.load(path) if path else default_registry()


def _variants(text):
    return [v for v in text.split(',') if v] if text else None


def _policy(args, registry):
    config = EVConfig(respect_irreversible=not args.allow_irreversible)
    if args.policy == 'baseline':
        return BaselinePolicy(registry, retries=args.retries, config=config, write_tools=WRITE_TOOLS)
    return EVPolicy(registry, config, write_tools=WRITE_TOOLS)


def _trace(args, domain, tid):
    runs = find_runs(args.fa_run, tid, domain, _variants(args.variants))
    return Trace.load_ir(runs[0]['dir'] / 'trajectory_ir.json')


def _dump(value):
    json.dump(value, sys.stdout, indent=2, ensure_ascii=False, default=list)
    sys.stdout.write('\n')


def cmd_payload(args):
    targets = [(args.domain, args.trajectory_id)] if args.trajectory_id else list_trajectories(args.fa_run)
    out = []
    for domain, tid in targets:
        payload = load_payload(args.fa_run, tid, domain, _variants(args.variants))
        out.append(payload.to_dict())
        if args.out:
            path = Path(args.out) / domain / trace_dir_name(tid) / 'rca_payload.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload.to_dict(), indent=2, ensure_ascii=False, default=list) + '\n',
                            encoding='utf-8')
    _dump(out if not args.trajectory_id else out[0])


def cmd_decide(args):
    registry = _registry(args.registry)
    policy = _policy(args, registry)
    payload = load_payload(args.fa_run, args.trajectory_id, args.domain, _variants(args.variants))
    trace = _trace(args, args.domain, args.trajectory_id)
    decision = policy.decide(payload, trace, policy.make_guard())
    result = {'trajectory_id': args.trajectory_id, 'policy': policy.name, 'action': decision.action,
              'reason': decision.reason, 'option': decision.option.describe() if decision.option else None}
    if isinstance(policy, EVPolicy):
        options, blocked = policy.options(payload, trace, policy.make_guard())
        result['ranked_options'] = [o.describe() for o in options[:args.show]]
        result['blocked'] = blocked
    _dump(result)


def cmd_eval_decisions(args):
    from . import evaluation   # gold is loaded only here, never on the decision path
    gold = evaluation.load_gold(args.cases)
    traces = {str(c['trajectory']['trajectory_id']): Trace.from_ir(c['trajectory'])
              for c in json.loads(Path(args.cases).read_text(encoding='utf-8-sig'))}
    registry = _registry(args.registry)
    policy = _policy(args, registry)
    items = []
    if args.oracle:
        items = [(traces[tid], evaluation.oracle_payload(label)) for tid, label in gold.items()]
    else:
        for domain, tid in list_trajectories(args.fa_run):
            if tid in gold:
                items.append((traces[tid], load_payload(args.fa_run, tid, domain, _variants(args.variants))))
    rows = evaluation.offline_decision_eval(items, gold, policy)
    _dump({'summary': evaluation.summarize_decisions(rows),
           'calibration': evaluation.calibration_table(rows), 'rows': rows if args.rows else len(rows)})


def build_parser():
    parser = argparse.ArgumentParser(prog='python3 -m recovery_orchestrator')
    sub = parser.add_subparsers(dest='command', required=True)

    def common(p, need_run=True):
        p.add_argument('--fa-run', required=need_run, help='FA experiment output dir (contains jobs/)')
        p.add_argument('--variants', help='comma-separated FA variants to pool (default: full AgentRx)')
        p.add_argument('--registry', help="strategy registry JSON, or 'mock' (default: bundled Repair-team snapshot)")
        p.add_argument('--policy', choices=['ev', 'baseline'], default='ev')
        p.add_argument('--retries', type=int, default=3)
        p.add_argument('--allow-irreversible', action='store_true',
                       help='simulation only: allow rollback across executed write actions')

    p = sub.add_parser('payload', help='Tier A files -> RCA payload (Tier B aggregates)')
    common(p)
    p.add_argument('--trajectory-id')
    p.add_argument('--domain', default='tau')
    p.add_argument('--out', help='also write <out>/<domain>/<digest>/rca_payload.json')
    p.set_defaults(func=cmd_payload)

    p = sub.add_parser('decide', help='show the policy decision for one trace')
    common(p)
    p.add_argument('--trajectory-id', required=True)
    p.add_argument('--domain', default='tau')
    p.add_argument('--show', type=int, default=5, help='ranked options to print')
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser('eval-decisions', help='offline decision evaluation against gold labels')
    common(p, need_run=False)
    p.add_argument('--cases', required=True, help="FA's data/benchmark/cases.json")
    p.add_argument('--oracle', action='store_true', help='use gold critical failures as the payload')
    p.add_argument('--rows', action='store_true', help='print per-trace rows')
    p.set_defaults(func=cmd_eval_decisions)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'eval-decisions' and not args.oracle and not args.fa_run:
        build_parser().error('--fa-run is required unless --oracle is given')
    args.func(args)
    return 0
