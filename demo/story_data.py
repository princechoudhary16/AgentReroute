"""Offline presentation evidence, validated against the source revision linked in the UI.

Gold is read here only to validate already-scored, committed research exports. This
module is not imported by any inference or model-input path.
"""
import csv
import hashlib
from collections import Counter
from pathlib import Path

from analyze_run import mcnemar, wilson

SOURCE_COMMIT = '0519e24940eb3ad8149083ce5b410708fbc285ff'
SOURCE_ROOT = f'https://github.com/princechoudhary16/AgentReroute/blob/{SOURCE_COMMIT}/'
REPORT_URL = SOURCE_ROOT + 'results/README.md'
RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
SOURCE_DIGESTS = {
    'final': 'c8e2efd6282abae69f7e72b742a15bb539d654c4a4da25c66ec5c96a6d333c1d',
    'selector': 'b491314894ad56092dc0b90d241db8d84f14dde4a1d4e960513edc3304d54cb2',
}
DOMAINS = {'tau': 'Tau · retail', 'magentic': 'Magentic · web'}
LABELS = {'agentrx': 'AgentRx · joint', 'seeds_select': 'AgentReroute · fixed checks',
          'synth_select_p10': 'AgentReroute · written checks',
          'synth_select_p5': 'Written checks · p5', 'mp_rl_top1': 'Frozen linear top-1',
          'mp_rl_select_pool': 'LLM · candidate pool only',
          'mp_rl_select_trace': 'LLM · full trace'}


def source_url(kind='final'):
    return SOURCE_ROOT + f'results/{kind}-evidence.csv'


def load_evidence(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f'Evidence source digest differs from pinned revision: {path.name}')
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    seen, gold = set(), {}
    for row in rows:
        key = (row['variant'], row['domain'], row['trajectory_id'])
        case = key[1:]
        if key in seen:
            raise ValueError(f'duplicate evidence row: {key}')
        seen.add(key)
        if case in gold and gold[case] != row['gold_step']:
            raise ValueError(f'Conflicting gold for {case}')
        gold[case] = row['gold_step']
        if row['hit'] not in ('0', '1') or (row['hit'] == '1') != (row['predicted_step'] == row['gold_step']):
            raise ValueError(f'Inconsistent exact-step scoring for {key}')
    return rows


def summarize(rows, domain, variant):
    arm = [r for r in rows if r['domain'] == domain and r['variant'] == variant]
    if not arm:
        raise ValueError(f'No evidence for {domain}/{variant}')
    n = len(arm)
    hits = sum(r['hit'] == '1' for r in arm)
    errors = sum(r['selector_status'] == 'error' for r in arm)
    inside = [r for r in arm if r['gold_in_pool'] == '1']
    in_hits = sum(r['hit'] == '1' for r in inside)
    # Match analyze_run.py: transport errors, not every fallback flag, taint an arm.
    complete = {r['trajectory_id'] for r in arm} == {r['trajectory_id'] for r in rows if r['domain'] == domain and r['variant'] == 'agentrx'}
    return dict(domain=domain, variant=variant, label=LABELS[variant], n=n, hits=hits,
                accuracy=100 * hits / n, ci=wilson(hits, n),
                fallbacks=sum(r['selector_fell_back'] == 'True' for r in arm),
                selector_errors=errors, statuses=dict(Counter(r['status'] for r in arm)),
                reportable=complete and errors / n < .2, complete=complete,
                in_pool_n=len(inside), in_pool_hits=in_hits, out_pool_hits=hits-in_hits,
                recall=100 * len(inside) / n,
                mean_pool=sum(int(r['pool_size'] or 0) for r in arm) / n)


def paired(rows, reference, challenger, domain=None):
    def arm(variant):
        return {(r['domain'], r['trajectory_id']): r for r in rows
                if r['variant'] == variant and (domain is None or r['domain'] == domain)}
    a, b = arm(reference), arm(challenger)
    if not a or a.keys() != b.keys():
        raise ValueError('Paired comparison requires matching, nonempty case sets')
    if any(a[key]['gold_step'] != b[key]['gold_step'] for key in a):
        raise ValueError('Paired comparison has inconsistent gold')
    wins, losses, p = mcnemar({k: r['hit'] == '1' for k, r in a.items()},
                              {k: r['hit'] == '1' for k, r in b.items()}, a.keys())
    return dict(wins=wins, losses=losses, p=p, n=len(a),
                delta=100 * (wins-losses) / len(a))


FINAL_ROWS = load_evidence(RESULTS_DIR / 'final-evidence.csv', SOURCE_DIGESTS['final'])
SELECTOR_ROWS = load_evidence(RESULTS_DIR / 'selector-evidence.csv', SOURCE_DIGESTS['selector'])
HEADLINE = {domain: [summarize(FINAL_ROWS, domain, variant)
                     for variant in ('agentrx', 'seeds_select', 'synth_select_p10')
                     if summarize(FINAL_ROWS, domain, variant)['reportable']]
            for domain in DOMAINS}
ABLATIONS = {domain: [summarize(SELECTOR_ROWS, domain, variant)
                      for variant in ('mp_rl_top1', 'mp_rl_select_pool', 'mp_rl_select_trace')]
             for domain in DOMAINS}
EXCLUDED = [summarize(FINAL_ROWS, domain, variant) for domain in DOMAINS
            for variant in ('synth_select_p5', 'synth_select_p10')
            if not summarize(FINAL_ROWS, domain, variant)['reportable']]
