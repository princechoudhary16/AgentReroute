"""Statistics for small paired samples: bootstrap CIs, exact McNemar, calibration, macro-F1."""
import math
import random
import statistics


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def bootstrap_ci(values, n_resamples=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean. Returns (low, high) or (None, None)."""
    values = list(values)
    if not values:
        return None, None
    if len(set(values)) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(k)) / k for _ in range(n_resamples))
    low = means[int(alpha / 2 * n_resamples)]
    high = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return low, high


def proportion(successes, total, seed=0):
    """Mean with bootstrap CI for a 0/1 list, as a report row."""
    values = [1.0 if s else 0.0 for s in successes]
    low, high = bootstrap_ci(values, seed=seed)
    return {'value': mean(values), 'ci95': [low, high], 'n': len(values), 'denominator': total}


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired 0/1 outcomes. Returns counts and p-value."""
    if len(a) != len(b):
        raise ValueError('paired samples must have equal length')
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    if n == 0:
        return {'only_a': 0, 'only_b': 0, 'p_value': 1.0}
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {'only_a': only_a, 'only_b': only_b, 'p_value': min(1.0, 2 * tail)}


def expected_calibration_error(confidences, correct, bins=10):
    pairs = list(zip(confidences, correct))
    if not pairs:
        return None, []
    table, total = [], 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [(c, k) for c, k in pairs if low <= c < high or (b == bins - 1 and c == 1.0)]
        if not members:
            continue
        conf = mean(c for c, _ in members)
        acc = mean(1.0 if k else 0.0 for _, k in members)
        total += len(members) / len(pairs) * abs(conf - acc)
        table.append({'bin': [low, high], 'n': len(members), 'confidence': conf, 'accuracy': acc})
    return total, table


def macro_f1(gold, predicted, labels):
    """Macro-F1 over ``labels`` (classes absent from gold are skipped)."""
    per = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, predicted) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, predicted) if g == label and p != label)
        if tp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp + fn}
    return (mean(v['f1'] for v in per.values()) if per else None), per


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def stdev(values):
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0
