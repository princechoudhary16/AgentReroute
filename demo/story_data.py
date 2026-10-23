"""Research-page data derived from pinned benchmark summaries, not live runs.

The source report is docs/evaluation-and-hld.md on mp_rl. Its saved numeric
summaries take precedence where rounded or conflicting values appear in prose.
"""
import json
from pathlib import Path

SOURCE_COMMIT = 'e96aed0cd401f507851c3a9d860a57b263ec5c80'
SOURCE_ROOT = f'https://github.com/princechoudhary16/AgentReroute/blob/{SOURCE_COMMIT}/'
REPORT_URL = SOURCE_ROOT + 'docs/evaluation-and-hld.md'
RESOURCE_DIR = Path(__file__).with_name('resources') / 'benchmark_35case'
BACKENDS = {
    'local': {'label': 'Local · Qwen 2.5 Coder 7B', 'model': 'qwen2.5-coder:7b',
              'provider': 'Ollama', 'file': 'ollama_test_summary.json', 'time_scale': 20,
              'note': 'Higher fault-location accuracy and recovery, with longer mean diagnosis time. AgentRx’s low replay-token ratio includes early halts; read it alongside recovery.'},
    'hosted': {'label': 'Hosted · MiniMax M3.1 Flash', 'model': 'minimax/minimax-m3.1-flash:free',
               'provider': 'OpenRouter', 'file': 'openrouter_test_summary.json', 'time_scale': 120,
               'note': 'Both methods recovered 25 of 35 failed cases. AgentReroute identified more exact fault steps, with a lower recovery-token ratio and shorter mean diagnosis time.'},
}
SUMMARIES = {key: json.loads((RESOURCE_DIR / info['file']).read_text()) for key, info in BACKENDS.items()}


def source_url(backend):
    return SOURCE_ROOT + 'checkpoints/results/benchmark_35case/' + BACKENDS[backend]['file']


def reduction(baseline, candidate):
    """Relative reduction from unrounded measurements, displayed to one decimal."""
    return round(100 * (1 - candidate / baseline), 1)


def metrics(backend):
    s = SUMMARIES[backend]
    ax, mp = s['agentrx_attribution'], s['mprl_attribution']
    ar, mr = s['agentrx_recovery'], s['mprl_recovery']
    n = s['n_failed_cases']
    def metric(id, title, description, values, unit, scale, precision, direction, counts=None):
        return dict(id=id, title=title, description=description, agentrx=values[0],
                    agentreroute=values[1], unit=unit, scale=scale, precision=precision,
                    direction=direction, counts=counts)
    return {
        'quality': [
            metric('location', 'Find the decision that went wrong.', 'Exact fault-location accuracy',
                   (100 * ax['exact_step_acc'], 100 * mp['exact_step_acc']), '%', 100, 1, 'Higher is better',
                   (f"{round(ax['exact_step_acc'] * n)}/{n} traces", f"{round(mp['exact_step_acc'] * n)}/{n} traces")),
            metric('recovery', 'Turn the diagnosis into a completed task.', 'Successful task recovery',
                   (100 * ar['recovery_rate']['value'], 100 * mr['recovery_rate']['value']), '%', 100, 1, 'Higher is better',
                   (f"{round(ar['recovery_rate']['value'] * n)}/{n} recovered", f"{round(mr['recovery_rate']['value'] * n)}/{n} recovered")),
        ],
        'efficiency': [
            metric('cost', 'Read replay cost alongside recovery.', 'Replay tokens / original-run tokens',
                   (ar['cost_vs_original'], mr['cost_vs_original']), '×', 1.5, 2, 'Lower is better'),
            metric('time', 'Time spent finding the fault.', 'Mean diagnosis latency',
                   (ax['mean_latency_s'], mp['mean_latency_s']), 's', BACKENDS[backend]['time_scale'], 2, 'Lower is better'),
        ],
    }


BENCHMARKS = {backend: metrics(backend) for backend in BACKENDS}
HOSTED = SUMMARIES['hosted']
IMPACT = {
    'cost': reduction(HOSTED['agentrx_recovery']['cost_vs_original'], HOSTED['mprl_recovery']['cost_vs_original']),
    'time': reduction(HOSTED['agentrx_attribution']['mean_latency_s'], HOSTED['mprl_attribution']['mean_latency_s']),
}


def detail_rows(backend):
    """Full comparison, including neutral and unfavorable results."""
    s = SUMMARIES[backend]
    a, m = s['agentrx_attribution'], s['mprl_attribution']
    ar, mr = s['agentrx_recovery'], s['mprl_recovery']
    pct = lambda v: f'{100 * v:.1f}%'
    rows = []
    for key, name in [('exact_step_acc', 'Exact fault step'), ('acceptable_step_acc', 'Acceptable fault step'),
                      ('category_acc', 'Fault category'), ('joint_exact_acc', 'Step + category match')]:
        rows.append((name, pct(a[key]), pct(m[key])))
    rows.append(('Retail failure mode', 'Not in saved summary', pct(m['mode_acc'])))
    for key, name in [('recovery_rate', 'Recovery · 35 failed cases'),
                      ('conditional_recovery_rate', 'Recovery when intervened'),
                      ('intervention_rate', 'Interventions · failed cases'),
                      ('harmful_intervention_rate', 'Harmful interventions · 17 controls'),
                      ('false_intervention_rate', 'Interventions · 17 controls')]:
        rows.append((name, pct(ar[key]['value']), pct(mr[key]['value'])))
    rows.extend([
        ('Replay tokens per recovered case', f"{ar['tokens_per_recovered']:,.0f}", f"{mr['tokens_per_recovered']:,.0f}"),
        ('Replay-token ratio vs original run', f"{ar['cost_vs_original']:.2f}×", f"{mr['cost_vs_original']:.2f}×"),
        ('Budget breaches', str(ar['budget_breaches']), str(mr['budget_breaches'])),
        ('Mean diagnosis time', f"{a['mean_latency_s']:.2f}s", f"{m['mean_latency_s']:.2f}s"),
    ])
    return rows
