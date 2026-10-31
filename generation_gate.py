"""Progress and actor-hygiene gate for one or more generation directories.

Safe to run at any time, including mid-run: it reads index.jsonl and lenient.jsonl, both of which are
appended per episode and per draw, so it does not wait for summary.json at the end of a run.

    python3 generation_gate.py benchmark/runs/gen-v2-0
    python3 generation_gate.py benchmark/runs/gen-v2-*          # all four slices at once

The gate that matters is the unrecovered rate. An unrecovered actor reply is one where every draw of
the repair ladder failed, which marks the episode 'degraded' and drops it at corpus build. A low rate
costs a few episodes; a high rate means the corpus is being thinned by decoding failures rather than by
the faults under study, and the run should stop.

Standard library only.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

GATE = 0.02


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass          # a half-written final line during a live run is expected
    return rows


def report(directory):
    directory = Path(directory)
    index = read_jsonl(directory / 'index.jsonl')
    lenient = read_jsonl(directory / 'lenient.jsonl')
    statuses = collections.Counter(r.get('status') for r in index)
    outcomes = collections.Counter(e.get('outcome') for e in lenient)
    tasks = {r.get('task_id') for r in index}
    config = {}
    meta = directory / 'generation.json'
    if meta.exists():
        config = json.loads(meta.read_text(encoding='utf-8')).get('config', {})

    print(f'== {directory}')
    if not index:
        print('   no episodes yet (index.jsonl empty or absent)')
        return None
    newest = max((directory / 'episodes').glob('*.json'), key=lambda p: p.stat().st_mtime, default=None)
    oldest = min((directory / 'episodes').glob('*.json'), key=lambda p: p.stat().st_mtime, default=None)
    if newest is not None and oldest is not None and newest is not oldest:
        span = newest.stat().st_mtime - oldest.stat().st_mtime
        rate = len(index) / span if span > 0 else 0
        idle = time.time() - newest.stat().st_mtime
        print(f'   {len(index)} episodes over {len(tasks)} task(s), {span / 3600:.2f} h elapsed, '
              f'{rate * 3600:.0f} episodes/h, last write {idle / 60:.1f} min ago')
        planned = config.get('max_tasks')
        slice_spec = config.get('task_slice')
        if planned and slice_spec:
            index_k, count = (int(x) for x in str(slice_spec).split('/'))
            mine = len(range(index_k, planned, count))
            if tasks and rate > 0:
                per_task = span / len(tasks)
                print(f'   slice {slice_spec}: {len(tasks)}/{mine} tasks done, '
                      f'eta {(mine - len(tasks)) * per_task / 3600:.1f} h')
    print(f'   statuses: {dict(sorted(statuses.items(), key=lambda kv: -kv[1]))}')

    committed = sum(statuses[s] for s in ('clean', 'failed', 'tolerated'))
    degraded = statuses.get('degraded', 0)
    if outcomes:
        print(f'   actor draws: {dict(sorted(outcomes.items()))}')
    unrecovered = outcomes.get('unrecovered', 0)
    denominator = committed + degraded
    rate = unrecovered / denominator if denominator else 0.0
    verdict = 'PASS' if rate < GATE else 'FAIL'
    print(f'   unrecovered {unrecovered} over {denominator} episodes = {rate:.2%}  '
          f'[{verdict}, gate <{GATE:.0%}]')
    if degraded:
        print(f'   {degraded} episode(s) marked degraded; corpus build will drop these')
    if statuses.get('natural_failure'):
        print(f"   {statuses['natural_failure']} clean run(s) failed the task outright "
              f"(eligibility, not a decoding problem)")
    return rate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+')
    args = ap.parse_args()
    rates = [r for r in (report(d) for d in args.dirs) if r is not None]
    if len(rates) > 1:
        worst = max(rates)
        print(f'\nworst slice: {worst:.2%}  [{"PASS" if worst < GATE else "FAIL"}]')
    return 0 if all(r < GATE for r in rates) else 1


if __name__ == '__main__':
    sys.exit(main())
