"""Command line: inspect the catalogue, plan repairs, audit traces, check rollouts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import acceptance
from .adapters.orchestrator import export_registry
from .context import AttemptRecord, Hypothesis, TaskSpec
from .engine import RepairEngine
from .evidence import claims, cycles
from .llm import OllamaClient
from .lowering import BENCHMARK, FULL, ORCHESTRATOR, lower
from .plan import RepairPlan
from .policy.rules import RuleGate
from .strategies import default_strategies
from .taxonomy import normalize
from .trace import Trace

TARGETS = {'benchmark': BENCHMARK, 'orchestrator': ORCHESTRATOR, 'full': FULL}


class UsageError(Exception):
    pass


def _json_arg(value: str):
    """Inline JSON or a path to a JSON file."""
    path = Path(value)
    text = path.read_text(encoding='utf-8-sig') if path.is_file() else value
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise UsageError(f'not a JSON value or file: {value[:80]!r} ({exc.msg})') from None


def hypothesis_from(data) -> Hypothesis:
    """FA result.json (``prediction``), a benchmark submission row (``hypotheses``) or a plain mapping."""
    if isinstance(data, dict) and isinstance(data.get('prediction'), dict):
        data = data['prediction']
    elif isinstance(data, dict) and isinstance(data.get('hypotheses'), list):
        if not data['hypotheses']:
            raise UsageError('the submission row has no hypotheses')
        data = data['hypotheses'][0]
    if not isinstance(data, dict):
        raise UsageError('a hypothesis must be a JSON object')
    return Hypothesis.from_mapping(data)


def task_from(args, trace: Trace) -> TaskSpec:
    task = TaskSpec.load(args.task) if args.task else TaskSpec.preset(args.preset)
    return task.with_task(instruction=trace.instruction or task.instruction)


def llm_from(args):
    """Ollama client for the rungs that need a model, or None to skip them."""
    if not args.llm:
        return None
    if args.num_ctx < 1024:
        raise UsageError('--num-ctx must be at least 1024')
    return OllamaClient(args.llm, base_url=args.ollama_url, num_ctx=args.num_ctx)


def _dump(value) -> None:
    json.dump(value, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write('\n')


# ---------------------------------------------------------------------- commands
def cmd_catalog(args):
    rows = []
    for s in default_strategies():
        spec = s.spec
        if args.category and normalize(args.category).category not in spec.categories:
            continue
        rows.append({'id': spec.id, 'rung': spec.rung, 'categories': list(spec.categories),
                     'subtypes': list(spec.subtypes) if spec.subtypes is not None else None, 'kind': spec.kind,
                     'prior_success': spec.prior_success, 'requires_llm': spec.requires_llm, 'fallback': spec.fallback,
                     'description': spec.description})
    _dump(sorted(rows, key=lambda r: (r['categories'][0], r['rung'], r['id'])))


def cmd_registry(args):
    registry = export_registry()
    if args.out:
        Path(args.out).write_text(json.dumps(registry, indent=2) + '\n', encoding='utf-8')
        print(f'wrote {len(registry["strategies"])} strategies to {args.out}')
    else:
        _dump(registry)


def cmd_plan(args):
    trace = Trace.load(args.trace)
    hypothesis = hypothesis_from(_json_arg(args.hypothesis))
    history = [AttemptRecord(**r) for r in _json_arg(args.history)] if args.history else []
    engine = RepairEngine(task_from(args, trace), llm=llm_from(args))
    if args.explain:
        _dump({'hypothesis': hypothesis.to_dict(),
               'ladder': [c.to_dict() if args.verbose else {k: v for k, v in c.to_dict().items() if k != 'plan'}
                          for c in engine.evaluate(trace, hypothesis, history)]})
        return
    plan = engine.propose(trace, hypothesis, history)
    if plan is None:
        _dump({'plan': None, 'reason': 'no applicable strategy left for this fault'})
        return 3
    out = json.loads(plan.to_json())
    if args.target:
        out = {'plan': out, 'lowered': lower(plan, TARGETS[args.target]).to_dict()}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    _dump(out)


def cmd_audit(args):
    trace = Trace.load(args.trace)
    task = task_from(args, trace)
    steps = sorted({e.step_index for e in trace.events if e.is_agent})
    unsupported = [{'step_index': s, 'claims': [c.describe() for c in
                                                claims.provenance(trace, s, synonyms=task.fact_keywords).faulty]}
                   for s in steps]
    _dump({'trajectory_id': trace.trajectory_id,
           'rule_violations': [v.describe() for v in RuleGate.for_task(task).audit(trace)],
           'loops': [loop.describe() for loop in cycles.detect(trace, task.write_tools)],
           'unsupported_claims': [u for u in unsupported if u['claims']]})


def cmd_check(args):
    data = _json_arg(args.plan)
    plan = RepairPlan.from_dict(data.get('plan', data))
    rollout = Trace.load(args.rollout)
    report = acceptance.evaluate(plan, rollout, task_from(args, rollout))
    _dump(report.to_dict())
    return 0 if report.passed else 1


def build_parser():
    parser = argparse.ArgumentParser(prog='python3 -m repair_strategies',
                                     description='Fault-specific, verifiable repair plans for failed agent runs.')
    sub = parser.add_subparsers(dest='command', required=True)

    def task_options(p):
        p.add_argument('--preset', default='retail', help='bundled task spec (default: retail)')
        p.add_argument('--task', help='task spec JSON (tools, rules, compensations); overrides --preset')

    p = sub.add_parser('catalog', help='list the repair strategies')
    p.add_argument('--category', help='only strategies for this fault category or benchmark mode')
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser('registry', help='strategy registry in the orchestrator schema')
    p.add_argument('--out', help='write to this file instead of stdout')
    p.set_defaults(func=cmd_registry)

    p = sub.add_parser('plan', help='plan the next repair for an attributed fault')
    p.add_argument('--trace', required=True, help='trace JSON (FA IR, benchmark trace, FA case, chat messages)')
    p.add_argument('--hypothesis', required=True,
                   help='JSON or file: FA result.json, benchmark submission row, or {step_index, category, ...}')
    p.add_argument('--history', help='JSON or file: attempts already made [{strategy_id, category, rollback_step, '
                                     'outcome}]')
    p.add_argument('--target', choices=sorted(TARGETS), help='also lower the plan for this backend')
    p.add_argument('--explain', action='store_true', help='show every rung of the ladder and why it applies or not')
    p.add_argument('--verbose', action='store_true', help='with --explain, include each rung\'s plan')
    p.add_argument('--llm', metavar='MODEL', help='Ollama model for the rungs that need one, e.g. qwen2.5:7b')
    p.add_argument('--ollama-url', help='Ollama server (default: $OLLAMA_HOST or http://127.0.0.1:11434)')
    p.add_argument('--num-ctx', type=int, default=16384, help='Ollama context window (default: 16384)')
    p.add_argument('--out', help='also write the plan JSON to this file')
    task_options(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser('audit', help='rule violations, loops and unsupported claims in a trace')
    p.add_argument('--trace', required=True)
    task_options(p)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser('check', help='run a plan\'s acceptance checks on a replayed rollout')
    p.add_argument('--plan', required=True, help='plan JSON or file (output of "plan")')
    p.add_argument('--rollout', required=True, help='trace of the replayed run')
    task_options(p)
    p.set_defaults(func=cmd_check)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except (UsageError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
