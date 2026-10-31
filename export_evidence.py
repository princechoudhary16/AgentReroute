#!/usr/bin/env python3
"""Distil a finished experiment into compact, committable evidence.

Each per-job result.json embeds every check row and runs to roughly 800 KB, so a 4-variant run over
73 cases is about 230 MB of artifacts. That is too much to put in git, but the artifacts are also
the only proof behind the reported numbers. This writes one row per (variant, domain, trajectory)
carrying everything analyze_run.py needs: the prediction, the gold label, and the pool facts that
explain the score. A few hundred kilobytes reproduces every figure in the report.

Usage:  python3 export_evidence.py runs/final results/final-evidence.csv
"""
import argparse, csv, json, sys
from pathlib import Path


def gold_index(run_dir):
    try:
        identity = json.loads((Path(run_dir) / 'experiment.json').read_text(encoding='utf-8'))
        wanted = identity['identity']['corpus_hash']
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    for manifest in sorted(Path('.').glob('**/manifest.json')):
        if any(p in ('.git', 'node_modules', 'requests', 'jobs') for p in manifest.parts):
            continue
        try:
            if json.loads(manifest.read_text(encoding='utf-8')).get('corpus_hash') != wanted:
                continue
        except (OSError, ValueError):
            continue
        cases_path = manifest.parent / 'cases.json'
        if not cases_path.exists():
            continue
        raw = json.loads(cases_path.read_text(encoding='utf-8'))
        cases = raw.get('cases', []) if isinstance(raw, dict) else raw
        return {(c['trajectory']['domain'], str(c['trajectory']['trajectory_id'])):
                (c.get('gold') or {}).get('critical') or {} for c in cases}
    return {}


FIELDS = ['variant', 'domain', 'trajectory_id', 'status', 'gold_step', 'gold_agent', 'gold_category',
          'predicted_step', 'predicted_agent', 'predicted_category', 'hit', 'agent_hit', 'category_hit',
          'abstained', 'pool_size', 'gold_in_pool', 'gold_rank', 'violations_per_candidate',
          'selector_status', 'selector_fell_back', 'selector_in_pool', 'recovery_resolution',
          'recovery_accepted_rank']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('out')
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    if not (run / 'jobs').exists():
        print(f'{run}: no jobs directory', file=sys.stderr)
        return 2
    gold = gold_index(run)
    if not gold:
        print(f'{run}: could not resolve the corpus, so gold columns would be empty', file=sys.stderr)
        return 2
    rows = []
    for variant_dir in sorted((run / 'jobs').iterdir()):
        for f in sorted((variant_dir / 'rep-0').rglob('result.json')) if (variant_dir / 'rep-0').exists() else []:
            d = json.loads(f.read_text(encoding='utf-8'))
            domain, tid = d['domain'], str(d['trajectory_id'])
            g = gold.get((domain, tid), {})
            pred = d.get('prediction') or {}
            sel = d.get('selection') or {}
            rec = d.get('recovery') or {}
            pool = [c['step_index'] for c in d.get('ranked_candidates', [])]
            rank = pool.index(g['step_index']) + 1 if g.get('step_index') in pool else ''
            rows.append({
                'variant': variant_dir.name, 'domain': domain, 'trajectory_id': tid,
                'status': d.get('status'),
                'gold_step': g.get('step_index'), 'gold_agent': g.get('agent'),
                'gold_category': g.get('category'),
                'predicted_step': pred.get('step_index'), 'predicted_agent': pred.get('agent'),
                'predicted_category': pred.get('category'),
                'hit': int(pred.get('step_index') == g.get('step_index') and pred.get('step_index') is not None),
                'agent_hit': int(pred.get('agent') == g.get('agent') and pred.get('agent') is not None),
                'category_hit': int(pred.get('category') == g.get('category') and pred.get('category') is not None),
                'abstained': int(pred.get('step_index') is None),
                'pool_size': len(pool), 'gold_in_pool': int(bool(rank)), 'gold_rank': rank,
                'violations_per_candidate': ' '.join(str(len(c.get('violated_checks') or []))
                                                     for c in d.get('ranked_candidates', [])),
                'selector_status': sel.get('status', ''), 'selector_fell_back': sel.get('fell_back', ''),
                'selector_in_pool': sel.get('in_pool', ''),
                'recovery_resolution': rec.get('resolution', ''),
                'recovery_accepted_rank': rec.get('accepted_rank', '')})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault((r['variant'], r['domain']), []).append(r)
    print(f'{out}  ({out.stat().st_size/1024:.0f} KB, {len(rows)} rows)')
    for (v, dom), group in sorted(by.items()):
        hits = sum(r['hit'] for r in group)
        print(f'  {v:<20}{dom:<10}{len(group):>4} traces  {100*hits/len(group):>5.1f}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())
