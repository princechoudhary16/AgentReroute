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
  oracle-ceiling    preflight: what an oracle attributor recovers on a split (the ceiling)
  report            Markdown summary of score/recover reports
  dashboard         self-contained HTML dashboard
  evaluate          standard suite on a split: baselines, recovery, comparisons, robustness, report, dashboard
  robustness        agent robustness per fault (MAS-FIRE RS, ToolMaze TSR/PRR/RC)
  train             learn theta and prompt memory on a corpus split, then evaluate frozen
  smoke             offline end-to-end run with scripted agents (no model, no network)
"""
import argparse
import json
import sys
import time
from pathlib import Path

from .corpus import build, generation_dir, load_gold, load_public, natural_queue
from .env.llm import (DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_NUM_CTX, CachedLLM, GeminiActor, LLMError,
                      OllamaClient, request)
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
    """Cached Ollama client, for the judge baselines (not an actor: no action contract is declared)."""
    return CachedLLM(OllamaClient(args.host, num_ctx=getattr(args, 'num_ctx', None) or DEFAULT_NUM_CTX), cache)


def _actor(args, cache, fallbacks=()):
    """The benchmark actor behind a response cache: local Ollama, or the Gemini bridge.

    The model string must match the corpus being replayed, because CachedLLM keys on it: replaying with
    a different actor than generation used turns every prefix into a cache miss, which is paid for twice
    and loses the reproducible prefix that makes an injected run comparable to its clean pair.
    """
    if getattr(args, 'backend', 'ollama') == 'gemini':
        actor = GeminiActor(args.model, settings={'temperature': 0.0},
                            lenient_log=str(Path(cache).parent / 'lenient.jsonl'))
    else:
        actor = OllamaClient(args.host, num_ctx=getattr(args, 'num_ctx', None) or DEFAULT_NUM_CTX)
    return CachedLLM(actor, cache, fallbacks=fallbacks)


def _generation_caches(corpus):
    """Every generation cache behind this corpus, so replays hit instead of re-buying actor tokens.

    A corpus merged from several generation runs has one cache per run. This used to return a single
    directory and gave up entirely when there was more than one, so training against a merged corpus
    started from an empty cache and paid again for responses generation had already bought. They are
    handed to CachedLLM as read-only fallbacks rather than as the write target, so training cannot
    mutate a generation artifact.
    """
    try:
        index = read_json(Path(corpus) / 'gold' / 'episodes.json')
    except (OSError, ValueError, TypeError):
        return []
    caches = {generation_dir(row, corpus) / 'llm_cache' for row in index.values()}
    return sorted(c for c in caches if c.exists())


def cmd_check_ollama(args):
    client = OllamaClient(args.host)
    models = client.available_models()
    print(f'Ollama at {args.host}: {len(models)} model(s): {", ".join(models) or "none"}')
    if args.model not in models and f'{args.model}:latest' not in models:
        print(f'Model {args.model} is not pulled. Run: ollama pull {args.model}', file=sys.stderr)
        return 2
    print(f'Model {args.model} is available (digest {client.model_digest(args.model)[:12]}).')
    # A model listing and a digest do not prove the server can complete: send one real round-trip.
    req = request(args.model, [{'role': 'user', 'content': 'Reply with the single word: ok'}], json_mode=False)
    start = time.perf_counter()
    try:
        completion = client.complete(req)
    except LLMError as exc:
        print(f'Completion failed: {exc}', file=sys.stderr)
        return 2
    latency = time.perf_counter() - start
    if not completion.text.strip():
        print(f'Completion returned empty text after {latency:.1f}s', file=sys.stderr)
        return 2
    print(f'Completion ok: {latency:.1f}s, {completion.tokens_in}+{completion.tokens_out} tokens, '
          f'reply {completion.text.strip()[:80]!r}')
    return 0


def cmd_fetch_tau(args):
    manifest = fetch_tau_retail(args.out, args.ref)
    print(f'Fetched tau-bench retail at {manifest["commit"]}: {manifest["n_tasks"]} tasks -> {args.out}')
    return 0


def cmd_generate(args):
    config = read_json(args.config) if args.config else {}
    for key in ('backend', 'model', 'host', 'data_dir', 'max_tasks', 'num_ctx', 'task_slice'):
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


def _llm_factory(args, cache=None):
    if args.backend == 'scripted':
        return scripted_llm
    cache = args.cache_dir or cache or Path(args.out).parent / 'llm_cache'
    shared = _actor(args, cache, fallbacks=_generation_caches(getattr(args, 'corpus', None)))
    return lambda domain, task: shared


def cmd_recover(args):
    hypotheses = None
    if args.predictions:
        hypotheses = {row['trajectory_id']: row.get('hypotheses', [])
                      for row in read_json(args.predictions)['predictions']}
    policy = POLICIES[args.policy](min_confidence=args.min_confidence) if args.policy == 'fixed_rules' \
        else POLICIES[args.policy]()
    caps = BudgetCaps(tokens=args.max_tokens, token_multiple=args.token_multiple, attempts=args.max_attempts,
                      seconds=args.max_seconds)
    report = run_recovery(args.corpus, args.split, policy, _llm_factory(args), caps, hypotheses, args.oracle,
                          not args.failed_only, args.limit)
    if args.predictions and not args.oracle:
        report['policy'] += f'+{Path(args.predictions).stem}'
    write_json(args.out, report)
    print(f'{report["policy"]} on {args.split}: recovery {report["metrics"]["recovery_rate"]["value"]}, '
          f'harm {report["metrics"]["harmful_intervention_rate"]["value"]}')
    return 0


def cmd_oracle_ceiling(args):
    """Preflight: what an oracle attributor recovers on a split - the ceiling a learned policy competes with.

    Reuses gold oracle hypotheses and fixed_rules:oracle (writes no new oracle). If the oracle recovers
    nothing, no learned policy can earn reward and the corpus, not the learner, is at fault. A scripted
    corpus is offline_fixture; its numbers verify plumbing only and are not a result.
    """
    gold = load_gold(args.corpus, args.split)
    label = 'offline_fixture' if 'offline_fixture' in {g.get('measurement_kind') for g in gold.values()} else 'model'
    caps = BudgetCaps(tokens=args.max_tokens, token_multiple=args.token_multiple, attempts=args.max_attempts,
                      seconds=args.max_seconds)
    report = run_recovery(args.corpus, args.split, POLICIES['fixed_rules'](), _llm_factory(args), caps,
                          oracle=True, limit=args.limit)
    m = report['metrics']
    print(json.dumps({'measurement_kind': label, 'split': args.split, 'policy': report['policy'],
                      'n_failed': report['n_failed'], 'oracle_recovery_rate': m['recovery_rate'],
                      'oracle_conditional_recovery_rate': m['conditional_recovery_rate']}, indent=2))
    if label == 'offline_fixture':
        print('NOTE: offline_fixture - plumbing check only, not a result.', file=sys.stderr)
    if not m['recovery_rate']['value']:
        print('WARNING: oracle recovers nothing - no learned policy can earn reward on this corpus.',
              file=sys.stderr)
    return 0


def _attributor(args, directory):
    """Attribution-side provider for constraint synthesis, cached per request.

    Separate from the actor on purpose: the actor replays episodes and needs the corpus's own model,
    while synthesis is the reasoning job and can run on a different, larger one. Both can be the same
    endpoint. Imported lazily because agentrx.rl_cross_branch imports this package.
    """
    from agentrx.providers import CachedProvider, HTTPProvider, OllamaProvider
    if args.backend == 'gemini':
        provider = HTTPProvider.from_config({'provider': 'gemini', 'model': args.model,
                                             'settings': {'temperature': 0.0}})
    else:
        provider = OllamaProvider.from_config({'provider': 'ollama', 'model': args.model,
                                               'base_url': args.host, 'num_ctx': args.num_ctx,
                                               'settings': {'temperature': 0.0}})
    return CachedProvider(provider, directory, namespace=f'synthesis:{args.model}')


def cmd_train(args):
    """Learn theta and the prompt memory on one split from observable feedback, then freeze and test.

    The checkpoint is written, so an existing one is refused unless --resume is given: the frozen
    baseline in checkpoints/ is what published numbers were measured against, and training over it in
    place would destroy that without any record.
    """
    from agentrx.rl_cross_branch import train_and_evaluate_benchmark
    if args.use_llm_synthesis and args.backend == 'scripted':
        # The scripted backend is an offline fixture with no attribution provider behind it, and
        # _attributor would quietly try to reach a model instead.
        print('error: --use-llm-synthesis needs a model backend (ollama or gemini), not scripted.',
              file=sys.stderr)
        return 2
    out = Path(args.out)
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists() and not args.resume:
        print(f'error: {checkpoint} exists and training writes to it. Pass --resume to continue from it, '
              f'or choose a new path to keep the existing checkpoint reproducible.', file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or (out / 'llm_cache')
    reused = _generation_caches(args.corpus)
    provider = _attributor(args, out / 'synthesis_cache') if args.use_llm_synthesis else None
    print(f'train={args.train_split} val={args.val_split} test={args.test_split} epochs={args.epochs} '
          f'backend={args.backend} synthesis={args.use_llm_synthesis}\nactor cache: {cache}\n'
          f'reusing {len(reused)} generation cache(s): {", ".join(str(c) for c in reused) or "none"}')
    result = train_and_evaluate_benchmark(
        args.corpus, train_split=args.train_split, test_split=args.test_split, epochs=args.epochs,
        rl_state_path=str(checkpoint), llm_factory=_llm_factory(args, cache),
        use_llm_synthesis=args.use_llm_synthesis, limit=args.limit, provider=provider,
        val_split=args.val_split, control_sample=args.control_sample,
        stop_above=args.stop_above, val_every=args.val_every, val_limit=args.val_limit,
        show_progress=not args.quiet)
    summary = {'corpus': str(args.corpus), 'backend': args.backend, 'model': args.model,
               'epochs': args.epochs, 'use_llm_synthesis': args.use_llm_synthesis,
               'splits': {'train': args.train_split, 'val': args.val_split, 'test': args.test_split},
               **result}
    write_json(out / 'train_val_summary.json', summary)
    for row in result['train_history']:
        # acceptance_minus_ledger_gap is the reward-hacking alarm: observable acceptance climbing away
        # from the harness's own recovery rate means the policy is learning to satisfy the cheap check.
        print(f'  epoch {row["epoch"]}: episodes={row["episodes"]} acted={row["n_acted"]} '
              f'mean_reward={row["mean_reward"]:.3f} acceptance={row["acceptance_rate"]:.3f} '
              f'harness_recovery={row["harness_recovery_rate"]} '
              f'gap={row["acceptance_minus_ledger_gap"]:+.3f} val_exact_step={row["val_attribution_exact_step"]}')
    print(f'checkpoint -> {checkpoint}\nsummary    -> {out / "train_val_summary.json"}')
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


def cmd_evaluate(args):
    """Standard suite on one split: baselines, recovery conditions, comparisons, robustness, report, dashboard."""
    from .evaluate import evaluate
    judge = None
    if args.backend == 'ollama':
        judge = _ollama(args, args.cache_dir or Path(args.out) / 'llm_cache')
        factory = lambda domain, task: judge  # noqa: E731
    else:
        factory = scripted_llm
    caps = BudgetCaps(tokens=args.max_tokens, token_multiple=args.token_multiple, attempts=args.max_attempts,
                      seconds=args.max_seconds)
    evaluate(args.corpus, args.split, factory, args.out, judge_llm=judge, judge_model=args.model, caps=caps,
             note=args.note, title=args.title, min_confidence=args.min_confidence)
    print(f'report -> {Path(args.out) / "report.md"}\ndashboard -> {Path(args.out) / "dashboard.html"}')
    return 0


def cmd_robustness(args):
    from .generate import load_episodes
    from .scoring.robustness import markdown as robustness_markdown
    from .scoring.robustness import robustness
    report = robustness([e for d in args.gen for e in load_episodes(d)])
    write_json(args.out, report)
    print(robustness_markdown(report))
    return 0


def cmd_smoke(args):
    """Offline end-to-end: scripted agents -> corpus -> evaluate (baselines, recovery, robustness, dashboard)."""
    from .evaluate import evaluate
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'generate.log', 'w', encoding='utf-8') as log:
        generate({'backend': 'scripted', 'model': 'scripted', 'seeds': [0]}, out / 'generation', log=log)
    manifest = build([out / 'generation'], out / 'corpus')
    note = 'OFFLINE FIXTURE: scripted agents on the bundled 8-task retail fixture. Verifies the pipeline only.'
    with open(out / 'evaluate.log', 'w', encoding='utf-8') as log:
        evaluate(out / 'corpus', 'test', scripted_llm, out, note=note, title='AgentReroute benchmark (offline smoke)',
                 log=log)
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
    s.add_argument('--backend', choices=('ollama', 'scripted', 'gemini')); s.add_argument('--model'); s.add_argument('--host')
    s.add_argument('--data-dir', dest='data_dir'); s.add_argument('--max-tasks', dest='max_tasks', type=int)
    s.add_argument('--num-ctx', dest='num_ctx', type=int,
                   help=f'ollama context window (default {DEFAULT_NUM_CTX}); overflow raises, never truncates')
    s.add_argument('--seeds', type=int, nargs='+'); s.add_argument('--operators', nargs='+')
    s.add_argument('--task-slice', dest='task_slice',
                   help='K/N: generate only every Nth task starting at K, for running N processes side by side into N output dirs that build-corpus then merges')
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
    s.add_argument('--limit', type=int)
    s.add_argument('--backend', choices=('ollama', 'gemini', 'scripted'), default='ollama')
    s.add_argument('--num-ctx', dest='num_ctx', type=int, default=DEFAULT_NUM_CTX)
    s.add_argument('--cache-dir', help='default: the generation cache behind this corpus, so replays hit')
    s.add_argument('--max-tokens', type=int, default=None,
                                                  help='fixed per-episode token cap (default: relative)')
    s.add_argument('--token-multiple', type=float, default=BudgetCaps.token_multiple,
                   help='relative cap: this many times the original run tokens')
    s.add_argument('--min-confidence', type=float, default=0.0, help='fixed_rules: ignore weaker hypotheses')
    s.add_argument('--max-attempts', type=int, default=BudgetCaps.attempts)
    s.add_argument('--max-seconds', type=float, default=BudgetCaps.seconds); s.add_argument('--out', required=True)
    model_args(s, model=False); s.set_defaults(fn=cmd_recover)

    s = sub.add_parser('oracle-ceiling'); s.add_argument('--corpus', required=True); s.add_argument('--split', default='train')
    s.add_argument('--backend', choices=('ollama', 'gemini', 'scripted'), default='ollama')
    s.add_argument('--cache-dir'); s.add_argument('--num-ctx', dest='num_ctx', type=int, default=DEFAULT_NUM_CTX)
    s.add_argument('--out'); s.add_argument('--limit', type=int); s.add_argument('--max-tokens', type=int, default=None)
    s.add_argument('--token-multiple', type=float, default=2.0); s.add_argument('--max-attempts', type=int, default=BudgetCaps.attempts)
    s.add_argument('--max-seconds', type=float, default=BudgetCaps.seconds); model_args(s, model=False)
    s.set_defaults(fn=cmd_oracle_ceiling)

    s = sub.add_parser('train'); s.add_argument('--corpus', required=True); s.add_argument('--out', required=True)
    s.add_argument('--checkpoint', required=True); s.add_argument('--resume', action='store_true')
    s.add_argument('--backend', choices=('ollama', 'gemini', 'scripted'), default='ollama')
    s.add_argument('--epochs', type=int, default=2); s.add_argument('--limit', type=int)
    s.add_argument('--stop-above', type=float,
                   help='stop once validation exact-step beats this. Measure the bar on the SAME data: '
                        'run agentrx over <corpus>/agentrx/<val-split>/cases.json, not its published '
                        'number from another dataset.')
    s.add_argument('--val-every', type=int, default=0,
                   help='evaluate validation every N training episodes (0 = only at epoch end). '
                        'Each evaluation costs a pass over the validation split, so keep N large.')
    s.add_argument('--val-limit', type=int, help='cap episodes per mid-epoch validation pass')
    s.add_argument('--quiet', action='store_true', help='suppress the per-episode progress line')
    s.add_argument('--control-sample', type=int, default=25,
                   help='clean/tolerated episodes to train on (they cost a full episode of synthesis '
                        'and usually yield no gradient); 0 trains on faults only. Test scores all.')
    s.add_argument('--train-split', dest='train_split', default='train')
    s.add_argument('--val-split', dest='val_split', default='calibration')
    s.add_argument('--test-split', dest='test_split', default='test')
    s.add_argument('--use-llm-synthesis', dest='use_llm_synthesis', action='store_true',
                   help='synthesise LLM constraints; without it only seed checks run and the prompt '
                        'memory receives nothing (constraints_used filters to origin == llm)')
    s.add_argument('--cache-dir'); s.add_argument('--num-ctx', dest='num_ctx', type=int, default=DEFAULT_NUM_CTX)
    model_args(s); s.set_defaults(fn=cmd_train)

    s = sub.add_parser('report'); s.add_argument('--attribution', nargs='*'); s.add_argument('--recovery', nargs='*')
    s.add_argument('--note'); s.add_argument('--out'); s.set_defaults(fn=cmd_report)
    s = sub.add_parser('dashboard'); s.add_argument('--corpus', required=True); s.add_argument('--attribution', nargs='*')
    s.add_argument('--recovery', nargs='*'); s.add_argument('--out', default='dashboard.html'); s.add_argument('--title')
    s.set_defaults(fn=cmd_dashboard)
    s = sub.add_parser('evaluate'); s.add_argument('--corpus', required=True); s.add_argument('--split', default='test')
    s.add_argument('--out', required=True)
    s.add_argument('--backend', choices=('ollama', 'gemini', 'scripted'), default='ollama')
    s.add_argument('--num-ctx', dest='num_ctx', type=int, default=DEFAULT_NUM_CTX)
    s.add_argument('--cache-dir', help='LLM cache shared with generation (recommended: the generation cache)')
    s.add_argument('--max-tokens', type=int, default=None); s.add_argument('--token-multiple', type=float, default=2.0)
    s.add_argument('--max-attempts', type=int, default=BudgetCaps.attempts)
    s.add_argument('--max-seconds', type=float, default=BudgetCaps.seconds)
    s.add_argument('--min-confidence', type=float, default=0.0); s.add_argument('--note')
    s.add_argument('--title', default='AgentReroute benchmark'); model_args(s); s.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser('robustness'); s.add_argument('--gen', nargs='+', required=True); s.add_argument('--out', required=True)
    s.set_defaults(fn=cmd_robustness)
    s = sub.add_parser('smoke'); s.add_argument('--out', default='benchmark/runs/smoke'); s.set_defaults(fn=cmd_smoke)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except LLMError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
