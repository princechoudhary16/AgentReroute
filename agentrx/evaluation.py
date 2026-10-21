"""Fixed-denominator benchmark metrics and across-repetition summaries."""
import statistics
from collections import Counter
from .pipeline import validate_prediction
from .taxonomy import agent

METRIC_NAMES = ('step_accuracy', 'acc_at_1', 'acc_at_3', 'acc_at_5', 'average_step_distance',
                'agent_accuracy', 'critical_category_accuracy', 'any_category_accuracy',
                'earliest_category_accuracy', 'terminal_category_accuracy')


def key(row):
    return row['domain'], str(row['trajectory_id'])


def score(cases, predictions):
    intended = {(c['trajectory']['domain'], c['trajectory']['trajectory_id']): c for c in cases}
    if len(intended) != len(cases):
        raise ValueError('Duplicate intended corpus IDs')
    rows = {}
    for prediction in predictions:
        k = key(prediction)
        if k not in intended or k in rows:
            raise ValueError(f'Unknown or duplicate prediction ID {k}')
        rows[k] = prediction
    counts = Counter({name:0 for name in METRIC_NAMES if name != 'average_step_distance'})
    stats = Counter()
    distances, confusion, details = [], {}, []
    for k, case in intended.items():
        row = rows.get(k)
        gold = case['gold']['critical']
        prediction, state = None, 'missing'
        if row is None:
            stats['missing_jobs'] += 1
        elif row.get('status') == 'error' or row.get('prediction') is None:
            stats['failed_jobs'] += 1; state = 'failed'
        else:
            try:
                raw = row['prediction']
                prediction = validate_prediction(raw, case['trajectory'], require_category=raw.get('category') is not None)
                if prediction.get('category') == 'Inconclusive':
                    stats['abstentions'] += 1; state = 'abstention'
                else:
                    state = 'valid'
                    stats['valid_step_predictions'] += 1
                    if prediction.get('category'):
                        stats['category_predictions'] += 1
                    if row.get('status') == 'partial_checker_errors':
                        stats['partial_checker_jobs'] += 1
            except (ValueError, TypeError, KeyError):
                stats['invalid_predictions'] += 1; state = 'invalid'
        predicted_label = prediction.get('category', '<no category>') if state == 'valid' else '<'+state+'>'
        bucket = confusion.setdefault(gold['category'], {})
        bucket[predicted_label] = bucket.get(predicted_label, 0) + 1
        detail = {'domain': k[0], 'trajectory_id': k[1], 'state': state,
                  'gold': gold, 'prediction': prediction}
        if state == 'valid':
            distance = abs(prediction['step_index'] - gold['step_index'])
            distances.append(distance)
            counts['step_accuracy'] += distance == 0
            for radius in (1,3,5): counts[f'acc_at_{radius}'] += distance <= radius
            counts['agent_accuracy'] += agent(prediction['agent']) == agent(gold['agent'])
            label = prediction.get('category')
            failures = case['gold']['failures']
            first = min(f['step_index'] for f in failures)
            last = max(f['step_index'] for f in failures)
            counts['critical_category_accuracy'] += label == gold['category']
            counts['any_category_accuracy'] += any(label == f['category'] for f in failures)
            # If several failures share a boundary step, any label at that step matches.
            counts['earliest_category_accuracy'] += any(label == f['category'] for f in failures if f['step_index'] == first)
            counts['terminal_category_accuracy'] += any(label == f['category'] for f in failures if f['step_index'] == last)
            detail['step_distance'] = distance
        details.append(detail)
    denominator = len(cases)
    metrics = {name: count/denominator if denominator else None for name,count in counts.items()}
    metrics['average_step_distance'] = statistics.mean(distances) if distances else None
    return {'denominator': denominator, 'numerators': dict(counts), 'metrics': metrics,
            **{name:stats[name] for name in ('valid_step_predictions', 'category_predictions', 'missing_jobs',
                'failed_jobs', 'invalid_predictions', 'abstentions', 'partial_checker_jobs')},
            'confusion_matrix': confusion, 'per_trajectory': details,
            'tie_policy': 'any category at earliest/terminal temporal step',
            'distance_denominator': len(distances)}


def aggregate(repetitions):
    if not repetitions:
        return {}
    result = {}
    for name in repetitions[0]['metrics']:
        values = [r['metrics'][name] for r in repetitions if r['metrics'][name] is not None]
        result[name] = {'mean': statistics.mean(values) if values else None,
                        'std': statistics.stdev(values) if len(values)>1 else (0 if values else None),
                        'n': len(values), 'expected_repetitions': len(repetitions)}
    return result
