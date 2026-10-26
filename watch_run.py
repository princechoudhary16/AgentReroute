#!/usr/bin/env python3
"""Live accuracy for an in-flight `agentrx run`, read-only.

The runner prints each trajectory as it starts but no running score, so a long run gives you no signal
until it ends. This joins the per-trace result.json files already on disk against data/benchmark gold
and prints running step accuracy per variant, so a run that is going nowhere can be stopped early
rather than discovered three hours later.

It only reads. It cannot disturb the run it is watching.

    python3 watch_run.py runs/matched-qwen32b-tau            # once
    python3 watch_run.py runs/matched-qwen32b-tau --follow   # refresh until interrupted
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

CORPUS = Path('data/benchmark/cases.json')


def gold_index(path=CORPUS):
    """trajectory_id -> (gold step index, gold category), per domain-agnostic id."""
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    cases = raw.get('cases', []) if isinstance(raw, dict) else raw
    index = {}
    for case in cases:
        critical = (case.get('gold') or {}).get('critical') or {}
        tid = str(case['gold'].get('trajectory_id') or case['trajectory']['trajectory_id'])
        index[tid] = (critical.get('step_index'), critical.get('category'))
    return index


def corpus_for_run(run_dir, override=None):
    """The corpus a run was scored against, taken from the run itself where possible.

    This used to default to data/benchmark unconditionally. A run over any other corpus then
    matched no trajectory_id, every gold lookup returned (None, None), and the monitor reported a
    confident 0.0% for a variant that was actually at 73.7%. The corpus is now derived from the
    run, and a mismatch is reported instead of being scored as a miss.
    """
    if override:
        return Path(override)
    try:
        identity = json.loads((Path(run_dir)/'experiment.json').read_text(encoding='utf-8'))
        wanted = identity['identity']['corpus_hash']
    except (OSError, ValueError, KeyError, TypeError):
        return Path(CORPUS)
    for manifest in sorted(Path('.').glob('**/manifest.json')):
        if any(part in ('.git', 'node_modules', 'requests') for part in manifest.parts):
            continue
        try:
            if json.loads(manifest.read_text(encoding='utf-8')).get('corpus_hash') == wanted:
                candidate = manifest.parent/'cases.json'
                if candidate.exists():
                    return candidate
        except (OSError, ValueError):
            continue
    return Path(CORPUS)


def tally(run_dir, gold):
    """Per (variant, domain): completed traces, step hits, category hits, and the newest mtime."""
    rows = defaultdict(lambda: {'n': 0, 'step': 0, 'category': 0, 'abstain': 0, 'latest': 0.0})
    for result in Path(run_dir).glob('jobs/*/rep-*/*/*/result.json'):
        try:
            data = json.loads(result.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue                       # being written right now; it will be counted next refresh
        variant = result.parents[3].name
        key = (variant, data.get('domain', '?'))
        row = rows[key]
        row['n'] += 1
        row['latest'] = max(row['latest'], result.stat().st_mtime)
        prediction = data.get('prediction') or {}
        if prediction.get('step_index') is None:
            # A dict carrying step_index None is an abstention, and it is truthy, so testing the dict
            # itself silently scored every abstention as a wrong answer instead.
            row['abstain'] += 1
            continue
        tid = str(data.get('trajectory_id'))
        if tid not in gold:
            # No gold for this id means the wrong corpus, not a wrong answer. Counted and reported.
            row['unknown_gold'] = row.get('unknown_gold', 0) + 1
            continue
        want_step, want_category = gold[tid]
        row['step'] += prediction.get('step_index') == want_step
        row['category'] += prediction.get('category') == want_category
    return rows


def render(run_dir, gold, started, corpus=None):
    rows = tally(run_dir, gold)
    if not rows:
        return f'{run_dir}: no completed traces yet'
    out = [f'{run_dir}   {time.strftime("%H:%M:%S")}']
    out.append(f'{"variant":<22}{"domain":<10}{"done":>5}{"step_acc":>10}{"cat_acc":>9}{"abstain":>9}')
    for (variant, domain), row in sorted(rows.items()):
        n = row['n']
        out.append(f'{variant:<22}{domain:<10}{n:>5}'
                   f'{100 * row["step"] / n:>9.1f}%{100 * row["category"] / n:>8.1f}%{row["abstain"]:>9}')
    unknown = sum(r.get('unknown_gold', 0) for r in rows.values())
    done = sum(r['n'] for r in rows.values())
    if unknown:
        out.append(f'WARNING: {unknown}/{done} traces have no gold entry in {corpus}. '
                   f'Those are counted as misses, so every accuracy above is a FLOOR, not the '
                   f'real value. Pass --cases <the corpus this run scored against>.')
    elapsed = time.time() - started
    if done and elapsed > 0:
        rate = done / (elapsed / 60)
        out.append(f'{done} trace(s) since watching began, {rate:.2f}/min')
    out.append('reference: agentrx published 31.03% step accuracy on tau, 18.18% on magentic')
    return '\n'.join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--follow', action='store_true', help='refresh until interrupted')
    parser.add_argument('--every', type=float, default=30.0, help='seconds between refreshes')
    parser.add_argument('--cases', default=str(CORPUS))
    args = parser.parse_args(argv)

    if not Path(args.run_dir).exists():
        print(f'{args.run_dir} does not exist yet', file=sys.stderr)
        return 2
    corpus = corpus_for_run(args.run_dir, args.cases if args.cases != str(CORPUS) else None)
    if not corpus.exists():
        print(f'corpus {corpus} not found', file=sys.stderr)
        return 2
    gold, started = gold_index(corpus), time.time()
    print(f'scoring against {corpus}', flush=True)
    while True:
        print(render(args.run_dir, gold, started, corpus), flush=True)
        if not args.follow:
            return 0
        print('-' * 68, flush=True)
        time.sleep(args.every)


if __name__ == '__main__':
    raise SystemExit(main())
