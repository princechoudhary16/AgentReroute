"""Command line: ``python3 -m benchmark <command>``.

  check-ollama      verify the Ollama server and model
  fetch-tau         download tau-bench retail data and tasks (pinned, hashed)
  generate          clean runs, eligibility, fault injection, labels
  build-corpus      splits, public traces, hidden gold, faultattribution export
  natural-queue     export natural failures for manual annotation
  import-agentrx    add the AgentRx release (from faultattribution `prepare`) as an external split
  import-whowhen    add Who&When logs as an external split
  baseline          produce an attribution submission from a baseline
  from-agentrx-run  convert a faultattribution run directory into a submission
  score             score a submission against a split's gold
  recover           run a recovery policy on a split's replayable episodes
  report            Markdown summary of score/recover reports
  dashboard         self-contained HTML dashboard
  smoke             offline end-to-end run with scripted agents (no model, no network)
"""
import argparse
import json
import sys
from pathlib import Path

from .corpus import build, load_gold, load_public, natural_queue
from .env.llm import DEFAULT_HOST, DEFAULT_MODEL, CachedLLM, LLMError, OllamaClient
from .env.scripted import scripted_llm
from .env.tasks import fetch_tau_retail
from .generate import generate
from .importers import import_agentrx, import_whowhen
from .io import read_json, write_json
from .scoring.attribution import score_attribution
from .scoring.baselines import (from_agentrx_run, heuristic_baseline, last_step_baseline, llm_judge_baseline,
                                random_baseline, step_by_step_baseline)
from .scoring.recovery import POLICIES, BudgetCaps, run_recovery
from .scoring.report import markdown

BASELINES = ('random', 'last_step', 'heuristic', 'llm_judge', 'step_by_step')


def _ollama(args, cache):
    return CachedLLM(OllamaClient(args.host), cache)


def cmd_check_ollama(args):
    client = OllamaClient(args.host)
    models = client.available_models()
    print(f'Ollama at {args.host}: {len(models)} model(s): {", ".join(models) or "none"}')
    if args.model not in models and f'{args.model}:latest' not in models:
        print(f'Model {args.model} is not pulled. Run: ollama pull {args.model}', file=sys.stderr)
        return 2
    print(f'Model {args.model} is available (digest {client.model_digest(args.model)[:12]}).')
    return 0


def cmd_fetch_tau(args):
    manifest = fetch_tau_retail(args.out, args.ref)
    print(f'Fetched tau-bench retail at {manifest["commit"]}: {manifest["n_tasks"]} tasks -> {args.out}')
    return 0


def cmd_generate(args):
    config = read_json(args.config) if args.config else {}
    for key in ('backend', 'model', 'host', 'data_dir', 'max_tasks'):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.seeds:
        config['seeds'] = args.seeds
    if args.operators:
        config['operators'] = args.operators
    summary = generate(config, args.out)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_build_corpus(args):
    manifest = build(args.gen, args.out, natural_labels=args.natural_labels)
    print(json.dumps({k: manifest[k] for k in ('splits', 'dropped', 'agentrx_export')}, indent=2))
    return 0


def cmd_natural_queue(args):
    n = natural_queue(args.gen, args.out)
    print(f'{n} natural failures written to {args.out} (fill labels_template.json, then pass it to build-corpus)')
    return 0


def cmd_import_agentrx(args):
    print(f'imported {import_agentrx(args.cases, args.corpus)} AgentRx release trajectories')
    return 0


def cmd_import_whowhen(args):
    print(f'imported {import_whowhen(args.dir, args.corpus)} Who&When trajectories')
    return 0


def cmd_baseline(args):
    traces = load_public(args.corpus, args.split, args.view)
    if args.method == 'random':
        sub = random_baseline(traces, args.view, args.seed)
    elif args.method == 'last_step':
        sub = last_step_baseline(traces, args.view)
    elif args.method == 'heuristic':
        sub = heuristic_baseline(traces, args.view)
    else:
        llm = _ollama(args, Path(args.out).parent / 'llm_cache')
        fn = llm_judge_baseline if args.method == 'llm_judge' else step_by_step_baseline
        sub = fn(traces, llm, args.model, args.view, args.seed)
    sub['split'] = args.split
    write_json(args.out, sub)
    print(f'{args.method}: {len(sub["predictions"])} predictions -> {args.out}')
    return 0


def cmd_from_agentrx_run(args):
    sub = from_agentrx_run(args.run_dir, args.variant, args.repetition, args.system)
    write_json(args.out, sub)
    print(f'{len(sub["predictions"])} predictions -> {args.out}')
    return 0


def cmd_score(args):
    gold = load_gold(args.corpus, args.split)
    report = score_attribution(read_json(args.submission), gold, args.seed)
    report['split'] = args.split
    write_json(args.out, report)
    m = report['metrics']
    print(f'{report["system"]} on {args.split} (n={report["denominator"]}): step {m["step_accuracy"]["value"]:.3f}, '
          f'agent {m["agent_accuracy"]["value"]:.3f}, mode {m["mode_accuracy"]["value"] or 0:.3f}')
    return 0


def _llm_factory(args):
    if args.backend == 'scripted':
        return scripted_llm
    shared = _ollama(args, args.cache_dir or Path(args.out).parent / 'llm_cache')
    return lambda domain, task: shared


def cmd_recover(args):
    hypotheses = None
    if args.predictions:
        hypotheses = {row['trajectory_id']: row.get('hypotheses', [])
                      for row in read_json(args.predictions)['predictions']}
    policy = POLICIES[args.policy]()
    caps = BudgetCaps(tokens=args.max_tokens, attempts=args.max_attempts, seconds=args.max_seconds)
    report = run_recovery(args.corpus, args.split, policy, _llm_factory(args), caps, hypotheses, args.oracle,
                          not args.failed_only, args.limit)
    if args.predictions and not args.oracle:
        report['policy'] += f'+{Path(args.predictions).stem}'
    write_json(args.out, report)
    print(f'{report["policy"]} on {args.split}: recovery {report["metrics"]["recovery_rate"]["value"]}, '
          f'harm {report["metrics"]["harmful_intervention_rate"]["value"]}')
    return 0


def cmd_report(args):
    attribution = [read_json(p) for p in args.attribution or []]
    recovery = [read_json(p) for p in args.recovery or []]
    text = markdown(attribution, recovery, note=args.note)
    if args.out:
        Path(args.out).write_text(text + '\n', encoding='utf-8')
    print(text)
    return 0


def cmd_dashboard(args):
    from .dashboard import build_dashboard
    path = build_dashboard(args.corpus, args.attribution or [], args.recovery or [], args.out, args.title)
    print(f'dashboard -> {path}')
    return 0


def cmd_smoke(args):
    """Offline end-to-end: scripted agents -> corpus -> baselines -> scoring -> recovery -> dashboard."""
    from .dashboard import build_dashboard
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'generate.log', 'w', encoding='utf-8') as log:
        generate({'backend': 'scripted', 'model': 'scripted', 'seeds': [0]}, out / 'generation', log=log)
    manifest = build([out / 'generation'], out / 'corpus')
    reports, recoveries = [], []
    for split in ('calibration', 'test'):
        traces = load_public(out / 'corpus', split)
        gold = load_gold(out / 'corpus', split)
        for sub in (random_baseline(traces), last_step_baseline(traces), heuristic_baseline(traces)):
            sub['split'] = split
            report = score_attribution(sub, gold)
            report['split'] = split
            write_json(out / 'reports' / f'attribution-{split}-{sub["system"]}.json', report)
            if split == 'test':
                reports.append(report)
    heuristic = {r['trajectory_id']: r['hypotheses'] for r in heuristic_baseline(load_public(out / 'corpus', 'test'))['predictions']}
    for name, kwargs in [('no_recovery', {}), ('full_restart', {}), ('retry_last_step', {}),
                         ('generic_reflection', {}), ('fixed_rules', {'hypotheses': heuristic}),
                         ('fixed_rules', {'oracle': True})]:
        report = run_recovery(out / 'corpus', 'test', POLICIES[name](), scripted_llm, **kwargs)
        if 'hypotheses' in kwargs:
            report['policy'] += '+heuristic'
        write_json(out / 'reports' / f'recovery-test-{report["policy"].replace(":", "-")}.json', report)
        recoveries.append(report)
    note = 'OFFLINE FIXTURE: scripted agents on the bundled 8-task retail fixture. Verifies the pipeline only.'
    (out / 'report.md').write_text(markdown(reports, recoveries, note=note) + '\n', encoding='utf-8')
    build_dashboard(out / 'corpus', reports, recoveries, out / 'dashboard.html', 'AgentReroute benchmark (offline smoke)')
    print(json.dumps({'splits': manifest['splits'], 'dropped': manifest['dropped']}, indent=2))
    print(f'report -> {out / "report.md"}\ndashboard -> {out / "dashboard.html"}')
    return 0


def parser():
    p = argparse.ArgumentParser(prog='python3 -m benchmark', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    def model_args(s, model=True):
        s.add_argument('--host', default=DEFAULT_HOST)
        if model:
            s.add_argument('--model', default=DEFAULT_MODEL)

    s = sub.add_parser('check-ollama'); model_args(s); s.set_defaults(fn=cmd_check_ollama)
    s = sub.add_parser('fetch-tau'); s.add_argument('--out', default='benchmark/data/tau_retail'); s.add_argument('--ref', default='main')
    s.set_defaults(fn=cmd_fetch_tau)

    s = sub.add_parser('generate'); s.add_argument('--config', type=Path); s.add_argument('--out', required=True)
    s.add_argument('--backend', choices=('ollama', 'scripted')); s.add_argument('--model'); s.add_argument('--host')
    s.add_argument('--data-dir', dest='data_dir'); s.add_argument('--max-tasks', dest='max_tasks', type=int)
    s.add_argument('--seeds', type=int, nargs='+'); s.add_argument('--operators', nargs='+')
    s.set_defaults(fn=cmd_generate)

    s = sub.add_parser('build-corpus'); s.add_argument('--gen', nargs='+', required=True); s.add_argument('--out', required=True)
    s.add_argument('--natural-labels', dest='natural_labels'); s.set_defaults(fn=cmd_build_corpus)
    s = sub.add_parser('natural-queue'); s.add_argument('--gen', nargs='+', required=True); s.add_argument('--out', required=True)
    s.set_defaults(fn=cmd_natural_queue)
    s = sub.add_parser('import-agentrx'); s.add_argument('--cases', required=True); s.add_argument('--corpus', required=True)
    s.set_defaults(fn=cmd_import_agentrx)
    s = sub.add_parser('import-whowhen'); s.add_argument('--dir', required=True); s.add_argument('--corpus', required=True)
    s.set_defaults(fn=cmd_import_whowhen)

    s = sub.add_parser('baseline'); s.add_argument('--corpus', required=True); s.add_argument('--split', default='test')
    s.add_argument('--method', choices=BASELINES, required=True); s.add_argument('--view', default='output_only',
                                                                               choices=('full', 'output_only'))
    s.add_argument('--seed', type=int, default=0); s.add_argument('--out', required=True); model_args(s)
    s.set_defaults(fn=cmd_baseline)
    s = sub.add_parser('from-agentrx-run'); s.add_argument('--run-dir', required=True); s.add_argument('--out', required=True)
    s.add_argument('--variant'); s.add_argument('--repetition', type=int); s.add_argument('--system')
    s.set_defaults(fn=cmd_from_agentrx_run)
    s = sub.add_parser('score'); s.add_argument('--corpus', required=True); s.add_argument('--split', default='test')
    s.add_argument('--submission', required=True); s.add_argument('--out', required=True); s.add_argument('--seed', type=int, default=0)
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser('recover'); s.add_argument('--corpus', required=True); s.add_argument('--split', default='test')
    s.add_argument('--policy', choices=sorted(POLICIES), required=True); s.add_argument('--predictions')
    s.add_argument('--oracle', action='store_true'); s.add_argument('--failed-only', action='store_true')
    s.add_argument('--limit', type=int); s.add_argument('--backend', choices=('ollama', 'scripted'), default='ollama')
    s.add_argument('--cache-dir'); s.add_argument('--max-tokens', type=int, default=BudgetCaps.tokens)
    s.add_argument('--max-attempts', type=int, default=BudgetCaps.attempts)
    s.add_argument('--max-seconds', type=float, default=BudgetCaps.seconds); s.add_argument('--out', required=True)
    model_args(s, model=False); s.set_defaults(fn=cmd_recover)

    s = sub.add_parser('report'); s.add_argument('--attribution', nargs='*'); s.add_argument('--recovery', nargs='*')
    s.add_argument('--note'); s.add_argument('--out'); s.set_defaults(fn=cmd_report)
    s = sub.add_parser('dashboard'); s.add_argument('--corpus', required=True); s.add_argument('--attribution', nargs='*')
    s.add_argument('--recovery', nargs='*'); s.add_argument('--out', default='dashboard.html'); s.add_argument('--title')
    s.set_defaults(fn=cmd_dashboard)
    s = sub.add_parser('smoke'); s.add_argument('--out', default='benchmark/runs/smoke'); s.set_defaults(fn=cmd_smoke)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except LLMError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
