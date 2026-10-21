"""Agent robustness to each fault, before any outside recovery (MAS-FIRE and ToolMaze style).

Computed from generation episodes (gold side). Every injected episode comes from
a task whose clean runs all succeeded, so the share that still succeeds under a
fault is MAS-FIRE's Robustness Score RS_f, and, because the fault always fired,
also ToolMaze's Perturbation Recovery Rate for tool faults.

* ``robustness_score``  tolerated / (tolerated + failed), per variant and per mode
* ``recovery_cost``     ToolMaze RC: 1 − c* / max(c, c*) for a run that succeeded,
                        1 for a failure; c = tool calls from the fault to the end,
                        c* = the same count in the paired clean run
* ``toolmaze``          the four tool-fault cells and the explicit − implicit gaps
"""
from collections import defaultdict

from .stats import mean, proportion

TOOLMAZE_CELLS = {
    'P1 explicit-transient': ('tool_error', 'explicit_transient'),
    'P2 explicit-permanent': ('tool_error', 'explicit_persistent'),
    'P3 implicit-transient': ('stale_output', 'silent_transient'),
    'P4 implicit-permanent': ('stale_output', 'silent_persistent'),
}


def _variant(op):
    params = ','.join(f'{k}={v}' for k, v in sorted(op['params'].items()))
    return f'{op["name"]}[{params}]' if params else op['name']


def _tool_calls_after(run, step_index):
    return sum(1 for s in run['steps'] if s['kind'] == 'tool' and s['index'] >= step_index)


def recovery_cost(episode, clean):
    if episode['status'] != 'tolerated':
        return 1.0
    fired = episode['run']['faults_fired']
    site = fired[0]['step_index'] if fired else 1
    c = _tool_calls_after(episode['run'], site)
    c_star = _tool_calls_after(clean['run'], site) if clean else c
    if max(c, c_star) == 0:
        return 0.0
    return round(1 - c_star / max(c, c_star), 4)


def robustness(episodes, seed=0):
    """``episodes``: generation episode dicts (any number of generation dirs)."""
    clean = {(e['task_id'], e['seed']): e for e in episodes if e['operator'] is None and e['status'] == 'clean'}
    injected = [e for e in episodes if e['operator'] and e['status'] in ('failed', 'tolerated')]
    by_variant, by_mode = defaultdict(list), defaultdict(list)
    for e in injected:
        by_variant[_variant(e['operator'])].append(e)
        by_mode[e['operator']['mode']].append(e)

    def summary(group):
        costs = [recovery_cost(e, clean.get((e['task_id'], e['seed']))) for e in group]
        return {'n': len(group), 'failed': sum(e['status'] == 'failed' for e in group),
                'robustness_score': proportion([e['status'] == 'tolerated' for e in group], len(group), seed),
                'recovery_cost': mean(costs)}

    variants = {k: {**summary(g), 'mode': g[0]['operator']['mode'], 'level': g[0]['operator']['level']}
                for k, g in sorted(by_variant.items())}
    cells = {}
    for cell, (name, value) in TOOLMAZE_CELLS.items():
        group = [e for e in injected if e['operator']['name'] == name and e['operator']['params'].get('cell') == value]
        if group:
            cells[cell] = summary(group)

    def rs(cell):
        return cells[cell]['robustness_score']['value'] if cell in cells else None

    gaps = {}
    for label, explicit, implicit in [('transient', 'P1 explicit-transient', 'P3 implicit-transient'),
                                      ('permanent', 'P2 explicit-permanent', 'P4 implicit-permanent')]:
        if rs(explicit) is not None and rs(implicit) is not None:
            gaps[label] = round(rs(explicit) - rs(implicit), 4)
    tasks = {e['task_id'] for e in episodes if e['operator'] is None}
    eligible = {t for t in tasks if all(e['status'] == 'clean' for e in episodes
                                        if e['operator'] is None and e['task_id'] == t)}
    return {'tasks': len(tasks), 'eligible_tasks': len(eligible),
            'eligible_fraction': len(eligible) / len(tasks) if tasks else None,
            'injected': len(injected), 'overall': summary(injected) if injected else None,
            'by_mode': {k: summary(g) for k, g in sorted(by_mode.items())}, 'by_variant': variants,
            'toolmaze': {'cells': cells, 'explicit_minus_implicit': gaps}}


def markdown(report):
    def pct(m):
        v = m['robustness_score']['value'] if isinstance(m, dict) else m
        return '–' if v is None else f'{100 * v:.0f}%'

    lines = ['## Agent robustness (before recovery)', '',
             f'{report["eligible_tasks"]} of {report["tasks"]} tasks eligible; {report["injected"]} activated faults. '
             'Robustness score = share of injected runs that still succeed (MAS-FIRE RS; ToolMaze PRR for tool '
             'faults). Recovery cost = ToolMaze RC (0 optimal, 1 failed).', '',
             '| Fault variant | Mode | n | Failed | Robustness | Recovery cost |', '|---|---|---:|---:|---:|---:|']
    for name, v in report['by_variant'].items():
        lines.append(f'| {name} | {v["mode"]} | {v["n"]} | {v["failed"]} | {pct(v)} | '
                     f'{v["recovery_cost"]:.2f} |')
    cells = report['toolmaze']['cells']
    if cells:
        lines += ['', '| ToolMaze cell | n | Success (TSR = PRR) | Recovery cost |', '|---|---:|---:|---:|']
        for cell, v in cells.items():
            lines.append(f'| {cell} | {v["n"]} | {pct(v)} | {v["recovery_cost"]:.2f} |')
        gaps = report['toolmaze']['explicit_minus_implicit']
        if gaps:
            lines += ['', 'Explicit − implicit success gap: ' +
                      ', '.join(f'{k} {100 * v:+.0f} points' for k, v in gaps.items()) + '.']
    return '\n'.join(lines) + '\n'
