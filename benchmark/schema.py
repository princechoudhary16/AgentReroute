"""Validation for the four public data contracts.

* trace       observable trajectory, compatible with the faultattribution IR
* gold        hidden label for one trajectory (scoring side only)
* submission  attribution / detection predictions from a system under test
* episode     replay record used by the recovery harness (scoring side only)
"""
from .taxonomy import INCONCLUSIVE, LABELS, mode

VIEWS = ('full', 'output_only')
GOLD_STATUSES = ('failed', 'tolerated', 'clean')


class SchemaError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise SchemaError(message)


def _int(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f'{name} must be an int >= {minimum}, got {value!r}')
    return value


def validate_trace(trace):
    _require(isinstance(trace, dict), 'trace must be an object')
    for key in ('trajectory_id', 'domain', 'instruction', 'steps'):
        _require(key in trace, f'trace missing {key}')
    _require(isinstance(trace['steps'], list) and trace['steps'], 'trace needs at least one step')
    previous = -1
    for step in trace['steps']:
        index = _int(step.get('index'), 'step.index')
        _require(index > previous, 'step indices must be strictly increasing')
        previous = index
        subs = step.get('substeps')
        _require(isinstance(subs, list) and subs, f'step {index} needs substeps')
        seen = set()
        for event in subs:
            sub = _int(event.get('sub_index'), 'sub_index', 1)
            _require(sub not in seen, f'duplicate sub_index in step {index}')
            seen.add(sub)
            _require(isinstance(event.get('role'), str) and event['role'], f'step {index} event needs role')
            _require(isinstance(event.get('content'), str), f'step {index} event content must be a string')
    if 'view' in trace:
        _require(trace['view'] in VIEWS, f'unknown view {trace["view"]!r}')
    return trace


def step_indices(trace):
    return {s['index'] for s in trace['steps']}


def agents_in(trace):
    return {e['role'] for s in trace['steps'] for e in s['substeps']}


def validate_gold(gold, trace=None):
    for key in ('trajectory_id', 'status', 'source'):
        _require(key in gold, f'gold missing {key}')
    _require(gold['status'] in GOLD_STATUSES, f'unknown gold status {gold["status"]!r}')
    if gold['status'] != 'failed':
        _require(gold.get('critical') is None, 'only failed trajectories carry a critical failure')
        return gold
    critical = gold.get('critical')
    _require(isinstance(critical, dict), 'failed gold needs critical')
    _int(critical.get('step_index'), 'critical.step_index')
    _require(isinstance(critical.get('agent'), str) and critical['agent'], 'critical.agent required')
    _require(critical.get('mode') in LABELS or critical.get('mode') is None,
             f'critical.mode must be one of {LABELS} or null (source without categories)')
    if gold.get('fault_step') is not None:
        _int(gold['fault_step'], 'fault_step')
    accept = gold.get('acceptable_steps', [])
    _require(isinstance(accept, list) and all(type(a) is int for a in accept), 'acceptable_steps must be ints')
    if trace is not None:
        indices = step_indices(trace)
        for step in [critical['step_index'], *accept] + ([gold['fault_step']] if gold.get('fault_step') is not None else []):
            _require(step in indices, f'gold step {step} absent from trajectory {gold["trajectory_id"]}')
    return gold


def normalize_hypothesis(raw):
    _require(isinstance(raw, dict), 'hypothesis must be an object')
    label = mode(raw.get('mode', INCONCLUSIVE)) if raw.get('mode') is not None else INCONCLUSIVE
    step = raw.get('step_index')
    if label != INCONCLUSIVE:
        _int(step, 'hypothesis.step_index')
    confidence = raw.get('confidence', 1.0)
    _require(isinstance(confidence, (int, float)) and 0 <= confidence <= 1, 'confidence must be in [0, 1]')
    return {'step_index': step, 'agent': str(raw.get('agent') or ''), 'mode': label,
            'confidence': float(confidence), 'rationale': str(raw.get('rationale', ''))}


def validate_submission(sub):
    """Returns {trajectory_id: {'hypotheses': [...], 'alarm_step': int|None}}."""
    _require(isinstance(sub, dict), 'submission must be an object')
    _require(isinstance(sub.get('system'), str) and sub['system'], 'submission needs a system name')
    _require(isinstance(sub.get('predictions'), list), 'submission needs a predictions list')
    out = {}
    for row in sub['predictions']:
        tid = str(row.get('trajectory_id', ''))
        _require(tid, 'prediction needs trajectory_id')
        _require(tid not in out, f'duplicate prediction for {tid}')
        hyps = [normalize_hypothesis(h) for h in row.get('hypotheses', [])][:3]
        hyps.sort(key=lambda h: -h['confidence'])
        alarm = row.get('alarm_step')
        if alarm is not None:
            _int(alarm, 'alarm_step')
        out[tid] = {'hypotheses': hyps, 'alarm_step': alarm}
    return out
