#!/usr/bin/env python3
"""Score a finished experiment the way the report needs it: accuracy with intervals, the recall
decomposition that explains it, and paired significance against a reference variant.

Raw percentages on 29 traces are misleading: one trace is 3.4 points. So every accuracy carries a
95% Wilson interval, and every comparison against the reference is an exact paired McNemar test on
the same trajectories. The recall decomposition splits each variant's accuracy into the part
explained by whether the true fault was in the candidate pool at all, which is what the
measurements in this project kept pointing at.

Usage:  python3 analyze_run.py runs/gemini-final [--reference agentrx]
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

REFERENCE_PUBLISHED = {'tau': 31.03, 'magentic': 18.18}


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - margin) / d * 100), min(100.0, (centre + margin) / d * 100))


def mcnemar(a_hits, b_hits, keys):
    """Exact two-sided binomial test on discordant pairs. a = reference, b = challenger."""
    b_wins = sum(1 for k in keys if not a_hits.get(k) and b_hits.get(k))
    a_wins = sum(1 for k in keys if a_hits.get(k) and not b_hits.get(k))
    n = a_wins + b_wins
    if n == 0:
        return b_wins, a_wins, 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_wins, b_wins) + 1))
    return b_wins, a_wins, min(1.0, 2 * tail / 2 ** n)


def gold_from_corpus(run_dir):
    """Gold straight from the corpus the run was scored against, so this works MID-RUN.

    predictions.csv is only written when the experiment finishes, and the recall column is the
    early kill signal for a long run, so waiting for it defeats the purpose.
    """
    try:
        identity = json.loads((Path(run_dir)/'experiment.json').read_text(encoding='utf-8'))
        wanted = identity['identity']['corpus_hash']
    except (OSError, ValueError, KeyError, TypeError):
        return {}, None
    for manifest in sorted(Path('.').glob('**/manifest.json')):
        if any(part in ('.git', 'node_modules', 'requests', 'jobs') for part in manifest.parts):
            continue
        try:
            if json.loads(manifest.read_text(encoding='utf-8')).get('corpus_hash') != wanted:
                continue
        except (OSError, ValueError):
            continue
        cases_path = manifest.parent/'cases.json'
        if not cases_path.exists():
            continue
        raw = json.loads(cases_path.read_text(encoding='utf-8'))
        cases = raw.get('cases', []) if isinstance(raw, dict) else raw
        out = {}
        for case in cases:
            critical = (case.get('gold') or {}).get('critical') or {}
            out[(case['trajectory']['domain'], str(case['trajectory']['trajectory_id']))] = \
                str(critical.get('step_index'))
        return out, cases_path
    return {}, None


def load_evidence(csv_path):
    """Rebuild the same rows from an export_evidence.py CSV.

    Per-job result.json files run to ~800 KB each, so a 4-variant run is ~230 MB and re-reading it
    for every analysis is slow and cannot be committed. The exported CSV carries exactly the fields
    used below, so the committed evidence reproduces every table on its own.
    """
    import csv as _csv
    rows, gold = {}, {}
    for r in _csv.DictReader(open(csv_path, encoding='utf-8')):
        key = (r['variant'], r['domain'], r['trajectory_id'])
        gold[(r['domain'], r['trajectory_id'])] = r['gold_step']
        rows[key] = {'hit': r['hit'] == '1',
                     'gold_in_pool': r['gold_in_pool'] == '1',
                     'pool': int(r['pool_size'] or 0),
                     'abstain': r['abstained'] == '1',
                     'violations': [int(x) for x in (r['violations_per_candidate'] or '').split()],
                     'fell_back': r['selector_fell_back'] == 'True',
                     'selector_error': r['selector_status'] == 'error'}
    return rows, gold


def load(run_dir):
    """(variant, domain, trajectory_id) -> {hit, gold_in_pool, pool, in_pool_pick}."""
    rows = {}
    gold, source = gold_from_corpus(run_dir)
    if gold:
        print(f'gold from {source}', file=sys.stderr)
    predictions = Path(run_dir) / 'predictions.csv'
    if predictions.exists():
        import csv
        for r in csv.DictReader(open(predictions)):
            gold[(r['domain'], r['trajectory_id'])] = r['gold_step']
    for job in sorted((Path(run_dir) / 'jobs').iterdir()):
        for f in (job / 'rep-0').rglob('result.json'):
            d = json.loads(f.read_text(encoding='utf-8'))
            key = (job.name, d['domain'], str(d['trajectory_id']))
            g = gold.get((d['domain'], str(d['trajectory_id'])))
            pool = [c['step_index'] for c in d.get('ranked_candidates', [])]
            pred = (d.get('prediction') or {}).get('step_index')
            sel = d.get('selection') or {}
            rows[key] = {'hit': g is not None and str(pred) == str(g),
                         'gold_in_pool': g is not None and str(g) in {str(x) for x in pool},
                         'pool': len(pool), 'abstain': pred is None,
                         'violations': [len(c.get('violated_checks') or [])
                                        for c in d.get('ranked_candidates', [])],
                         'fell_back': bool(sel.get('fell_back')),
                         'selector_error': sel.get('status') == 'error'}
    return rows, gold


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', help='a run directory, or an export_evidence.py CSV')
    ap.add_argument('--reference', default='agentrx', help='variant every other is compared against')
    args = ap.parse_args(argv)
    if not Path(args.run_dir).exists():
        print(f'{args.run_dir} does not exist', file=sys.stderr)
        return 2
    if args.run_dir.endswith('.csv'):
        rows, gold = load_evidence(args.run_dir)
    else:
        rows, gold = load(args.run_dir)
    if not rows:
        print(f'{args.run_dir}: no completed traces yet', file=sys.stderr)
        return 2
    if not gold:
        print('WARNING: no gold found for this run, from predictions.csv or from the corpus. '
              'Every accuracy would read 0, so refusing to print one.', file=sys.stderr)
        return 2
    incomplete = {v: sum(1 for k in rows if k[0] == v) for v in sorted({k[0] for k in rows})}
    if any(n < max(incomplete.values()) for n in incomplete.values()):
        print(f'NOTE: run in progress, traces per variant: {incomplete}. '
              'Read the recall column; accuracies on partial arms are not comparable yet.',
              file=sys.stderr)

    variants = sorted({k[0] for k in rows})
    domains = sorted({k[1] for k in rows})

    print(f'\n=== {args.run_dir} ===')
    for domain in domains:
        keys = sorted({k[2] for k in rows if k[1] == domain})
        print(f'\n--- {domain}  (n={len(keys)})'
              f'{"   agentrx published: %.2f%%" % REFERENCE_PUBLISHED[domain] if domain in REFERENCE_PUBLISHED else ""}')
        print(f'  {"variant":<20}{"acc":>8}{"95% CI":>16}{"recall":>9}{"in-pool":>10}'
              f'{"out-pool":>10}{"pool":>7}{"viol/c":>8}{"fellback":>10}')
        tainted = []
        hits = {}
        for v in variants:
            present = [k for k in keys if (v, domain, k) in rows]
            if not present:
                continue
            r = [rows[(v, domain, k)] for k in present]
            n = len(r)
            k_hit = sum(x['hit'] for x in r)
            hits[v] = {k: rows[(v, domain, k)]['hit'] for k in present}
            gip = [x for x in r if x['gold_in_pool']]
            ngip = [x for x in r if not x['gold_in_pool']]
            lo, hi = wilson(k_hit, n)
            flat = [c for x in r for c in x['violations']]
            fb = sum(x['fell_back'] for x in r)
            errs = sum(x.get('selector_error', False) for x in r)
            # A synthesis arm whose provider calls failed degrades silently to the seed
            # constraints (synthesize_perspective_constraints keeps the seeds on failure), so the
            # row looks like a valid seeds-only result instead of a failed synthesis one. The
            # giveaway is the pool: seeds average ~1.2 candidates on magentic and ~4.5 on tau,
            # while synthesis reaches 5-12. Judged together with the selector errors.
            mark = ' <-- TAINTED' if errs and errs / n >= 0.2 else ''
            if mark:
                tainted.append((v, errs, n))
            print(f'  {v:<20}{100*k_hit/n:>7.1f}%{f"[{lo:.1f}, {hi:.1f}]":>16}'
                  f'{100*len(gip)/n:>8.1f}%'
                  f'{(100*sum(x["hit"] for x in gip)/len(gip) if gip else 0):>9.0f}%'
                  f'{(100*sum(x["hit"] for x in ngip)/len(ngip) if ngip else 0):>9.0f}%'
                  f'{sum(x["pool"] for x in r)/n:>7.2f}{(sum(flat)/len(flat) if flat else 0):>8.2f}'
                  f'{f"{fb}/{n}":>10}{mark}')
        if tainted:
            print()
            for v, errs, n in tainted:
                print(f'  DO NOT REPORT {v}: the selector call FAILED on {errs}/{n} traces '
                      f'(transport errors, e.g. rate limiting). Those traces silently fell back to '
                      f'the theta top-1 ranking, so this row measures theta, not the selector.')
        if args.reference in hits:
            print(f'\n  paired McNemar against {args.reference} (exact, two-sided):')
            for v in variants:
                if v == args.reference or v not in hits:
                    continue
                shared = [k for k in keys if k in hits[v] and k in hits[args.reference]]
                w, l, p = mcnemar(hits[args.reference], hits[v], shared)
                mark = 'SIGNIFICANT' if p < 0.05 else 'not significant'
                print(f'    {v:<20} wins {w:>2}, losses {l:>2}, n={len(shared):>3}  p={p:.4f}  {mark}')
            # Identity, not a forecast: recall*in-pool + (1-recall)*out-pool IS the accuracy. It is
            # printed to show how much of each variant's score rides on pool recall rather than on
            # selection quality. The predictive use is cross-variant: take one variant's in-pool and
            # out-pool rates, substitute another's recall, and you get an estimate worth testing.
            print('\n  accuracy decomposed by pool recall (identity, not a forecast):')
            for v in variants:
                present = [k for k in keys if (v, domain, k) in rows]
                r = [rows[(v, domain, k)] for k in present]
                if not r:
                    continue
                gip = [x for x in r if x['gold_in_pool']]
                ngip = [x for x in r if not x['gold_in_pool']]
                rec = len(gip)/len(r)
                ip = (sum(x['hit'] for x in gip)/len(gip)) if gip else 0
                op = (sum(x['hit'] for x in ngip)/len(ngip)) if ngip else 0
                print(f'    {v:<20}{100*(rec*ip+(1-rec)*op):>7.1f}%  actual '
                      f'{100*sum(x["hit"] for x in r)/len(r):>5.1f}%')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
