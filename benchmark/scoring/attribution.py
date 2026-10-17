"""Attribution and detection scoring against hidden gold.

Fixed denominators: every failed trajectory in the split counts, whether the
submission predicted it, abstained, or omitted it (omissions count as misses).
"""
from collections import defaultdict

from ..schema import validate_submission
from ..taxonomy import MODES, OUT_OF_SCOPE, agent_name
from .stats import expected_calibration_error, macro_f1, mean, proportion

STEP_RADII = (1, 3)


def _top(hyps):
    return next((h for h in hyps if h['mode'] != 'inconclusive' and h['step_index'] is not None), None)


def score_attribution(submission, gold, seed=0):
    """Score a submission dict against ``gold`` ({tid: gold record})."""
    predictions = validate_submission(submission)
    unknown = sorted(set(predictions) - set(gold))
    failed = {tid: g for tid, g in gold.items() if g['status'] == 'failed'}
    rows = []
    for tid, g in sorted(failed.items()):
        pred = predictions.get(tid, {'hypotheses': [], 'alarm_step': None})
        hyps = pred['hypotheses']
        top = _top(hyps)
        crit = g['critical']
        row = {'trajectory_id': tid, 'gold_step': crit['step_index'], 'gold_agent': crit['agent'],
               'gold_mode': crit['mode'], 'fault_step': g.get('fault_step'), 'source': g.get('source'),
               'state': 'missing' if tid not in predictions else ('abstained' if top is None else 'predicted'),
               'pred_step': top['step_index'] if top else None, 'pred_agent': top['agent'] if top else None,
               'pred_mode': top['mode'] if top else None, 'confidence': top['confidence'] if top else 0.0,
               'injection': g.get('injection')}
        if top:
            distance = abs(top['step_index'] - crit['step_index'])
            row.update({
                'step_distance': distance,
                'step_exact': distance == 0,
                **{f'step_within_{r}': distance <= r for r in STEP_RADII},
                'fault_step_exact': g.get('fault_step') is not None and top['step_index'] == g['fault_step'],
                'acceptable_step': top['step_index'] in set(g.get('acceptable_steps', [crit['step_index']])),
                'agent_correct': agent_name(top['agent']) == agent_name(crit['agent']),
                'mode_correct': crit['mode'] is not None and top['mode'] == crit['mode'],
            })
            row['joint_correct'] = row['step_exact'] and row['agent_correct'] and row['mode_correct']
        row['top3_step'] = any(h['step_index'] == crit['step_index'] for h in hyps if h['step_index'] is not None)
        row['top3_mode'] = crit['mode'] is not None and any(h['mode'] == crit['mode'] for h in hyps)
        rows.append(row)

    def rate(key, subset=None):
        subset = rows if subset is None else subset
        return proportion([r.get(key, False) for r in subset], len(subset), seed)

    with_mode = [r for r in rows if r['gold_mode'] is not None]
    metrics = {
        'step_accuracy': rate('step_exact'),
        **{f'step_within_{r}': rate(f'step_within_{r}') for r in STEP_RADII},
        'fault_step_accuracy': rate('fault_step_exact', [r for r in rows if r['fault_step'] is not None]),
        'acceptable_step_accuracy': rate('acceptable_step'),
        'top3_step_accuracy': rate('top3_step'),
        'agent_accuracy': rate('agent_correct'),
        'mode_accuracy': rate('mode_correct', with_mode),
        'top3_mode_accuracy': rate('top3_mode', with_mode),
        'joint_accuracy': rate('joint_correct', with_mode),
        'average_step_distance': mean(r['step_distance'] for r in rows if 'step_distance' in r),
    }
    scored = [r for r in with_mode if r['state'] == 'predicted']
    f1, per_mode = macro_f1([r['gold_mode'] for r in with_mode],
                            [r['pred_mode'] or 'inconclusive' for r in with_mode], list(MODES) + [OUT_OF_SCOPE])
    metrics['mode_macro_f1'] = f1
    ece, ece_table = expected_calibration_error([r['confidence'] for r in rows if r['state'] == 'predicted'],
                                                [r['step_exact'] for r in rows if r['state'] == 'predicted'])
    metrics['step_ece'] = ece
    confusion = defaultdict(lambda: defaultdict(int))
    for r in with_mode:
        confusion[r['gold_mode']][r['pred_mode'] or f'<{r["state"]}>'] += 1
    breakdown = {}
    for key, fn in [('by_mode', lambda r: r['gold_mode'] or 'unlabeled'),
                    ('by_source', lambda r: r['source'] or 'unknown'),
                    ('by_method', lambda r: (r['injection'] or {}).get('method', 'none')),
                    ('by_level', lambda r: (r['injection'] or {}).get('level', 'none')),
                    ('by_timing', lambda r: (r['injection'] or {}).get('timing') or 'none')]:
        groups = defaultdict(list)
        for r in rows:
            groups[fn(r)].append(r)
        breakdown[key] = {name: {'n': len(group), 'step_accuracy': mean(1.0 if r.get('step_exact') else 0.0 for r in group),
                                 'agent_accuracy': mean(1.0 if r.get('agent_correct') else 0.0 for r in group),
                                 'mode_accuracy': mean(1.0 if r.get('mode_correct') else 0.0 for r in group)}
                          for name, group in sorted(groups.items())}
    report = {'system': submission['system'], 'view': submission.get('view'), 'denominator': len(rows),
              'states': {s: sum(1 for r in rows if r['state'] == s) for s in ('predicted', 'abstained', 'missing')},
              'unknown_predictions': unknown, 'metrics': metrics, 'per_mode': per_mode,
              'calibration': ece_table, 'confusion': {k: dict(v) for k, v in confusion.items()},
              'breakdown': breakdown, 'per_trajectory': rows, 'n_scored_with_mode': len(scored)}
    if any(p['alarm_step'] is not None for p in predictions.values()) or submission.get('detection'):
        report['detection'] = score_detection(predictions, gold, seed)
    return report


def score_detection(predictions, gold, seed=0):
    """Online detection: alarms on failed runs (at/after the fault) vs alarms on controls."""
    tp = premature = missed = false_alarm = quiet = 0
    delays, rows = [], []
    for tid, g in sorted(gold.items()):
        alarm = predictions.get(tid, {}).get('alarm_step')
        if g['status'] == 'failed':
            onset = g.get('fault_step', g['critical']['step_index'])
            if alarm is None:
                missed += 1
                state = 'missed'
            elif alarm < onset:
                premature += 1
                state = 'premature'
            else:
                tp += 1
                delays.append(alarm - onset)
                state = 'detected'
        else:
            state = 'false_alarm' if alarm is not None else 'quiet'
            false_alarm += alarm is not None
            quiet += alarm is None
        rows.append({'trajectory_id': tid, 'status': g['status'], 'alarm_step': alarm, 'state': state})
    n_failed = tp + premature + missed
    n_controls = false_alarm + quiet
    precision = tp / (tp + premature + false_alarm) if tp + premature + false_alarm else None
    return {'precision': precision, 'recall': tp / n_failed if n_failed else None,
            'false_alarm_rate': false_alarm / n_controls if n_controls else None,
            'premature_rate': premature / n_failed if n_failed else None,
            'mean_delay_steps': mean(delays), 'n_failed': n_failed, 'n_controls': n_controls,
            'per_trajectory': rows}
