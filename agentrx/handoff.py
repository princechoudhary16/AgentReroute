"""Versioned recovery contract, separate from legacy run-vote payloads."""
from .boundary import compute_boundary, boundary_decision, validate_threshold
from .predictions import validate_candidates
from .io import read_json

SCHEMA_VERSION = 'agentrx-recovery-v1'
TAXONOMY_VERSION = 'agentrx-9-v1'
CONFIDENCE_KIND = 'uncalibrated_model_estimate'


def build_handoff(trajectory, boundary, candidates=(), *, measurement_kind='fixture_replay',
                  status='ok', attribution_status=None, abstention_reason=None):
    decision = boundary_decision(boundary)
    if attribution_status is None:
        attribution_status = ('error' if status == 'error' else 'ranked' if candidates else
                              'abstained' if decision == 'repair_required' else 'not_requested')
    if attribution_status != 'ranked' and not abstention_reason:
        abstention_reason = {'error': 'Attribution unavailable because the pipeline failed.',
                             'abstained': 'Insufficient evidence to attribute a critical step.',
                             'not_requested': 'Divergence gate did not request attribution.'}[attribution_status]
    payload = {'schema_version': SCHEMA_VERSION, 'trajectory_id': trajectory['trajectory_id'],
               'domain': trajectory['domain'], 'measurement_kind': measurement_kind,
               'status': status, 'decision': decision, 'boundary': boundary,
               'taxonomy_version': TAXONOMY_VERSION, 'confidence_kind': CONFIDENCE_KIND,
               'attribution_status': attribution_status, 'candidates': list(candidates),
               'abstention_reason': abstention_reason}
    return validate_handoff(payload, trajectory)


def validate_handoff(payload, trajectory, checks=None):
    if not isinstance(payload, dict):
        raise ValueError('Handoff must be an object')
    expected = {'schema_version', 'trajectory_id', 'domain', 'measurement_kind', 'status', 'decision',
                'boundary', 'taxonomy_version', 'confidence_kind', 'attribution_status',
                'candidates', 'abstention_reason'}
    if set(payload) != expected:
        raise ValueError('Handoff fields do not match the contract')
    for field, value in [('schema_version', SCHEMA_VERSION), ('taxonomy_version', TAXONOMY_VERSION),
                         ('confidence_kind', CONFIDENCE_KIND), ('trajectory_id', trajectory['trajectory_id']),
                         ('domain', trajectory['domain'])]:
        if payload[field] != value:
            raise ValueError(f'Invalid handoff {field}')
    if not isinstance(payload['measurement_kind'], str) or not payload['measurement_kind'].strip():
        raise ValueError('Handoff needs measurement_kind')
    b = payload['boundary']
    if not isinstance(b, dict):
        raise ValueError('Handoff boundary must be an object')
    validate_threshold(b.get('threshold'))
    template = compute_boundary([], b['threshold'])
    if set(b) != set(template) or b['comparison'] != '>':
        raise ValueError('Boundary fields do not match the contract')
    counters = set(template) - {'threshold', 'comparison', 'divergence_rate'}
    if any(type(b[k]) is not int or b[k] < 0 for k in counters):
        raise ValueError('Boundary counters must be nonnegative integers')
    if (b['conclusive_checks'] != b['violated_checks'] + b['satisfied_checks'] or
        b['unique_checks'] != b['conclusive_checks'] + b['skipped_checks'] + b['error_checks'] + b['unresolved_checks'] or
        b['check_evaluations'] != b['unique_checks'] + b['duplicate_evaluations'] or
        b['conflicting_checks'] > min(b['unresolved_checks'], b['duplicate_evaluations'])):
        raise ValueError('Inconsistent boundary counters')
    expected_rate = b['violated_checks'] / b['conclusive_checks'] if b['conclusive_checks'] else None
    if (b['divergence_rate'] is not None and type(b['divergence_rate']) not in (int, float)) or b['divergence_rate'] != expected_rate:
        raise ValueError('Inconsistent divergence rate')
    if checks is not None and b != compute_boundary(checks, b['threshold']):
        raise ValueError('Boundary does not match original checks')
    if payload['decision'] != boundary_decision(b):
        raise ValueError('Decision does not match boundary')
    status, attribution = payload['status'], payload['attribution_status']
    if status not in ('ok', 'partial_checker_errors', 'error') or attribution not in ('ranked', 'abstained', 'not_requested', 'error'):
        raise ValueError('Unknown handoff status')
    if (status == 'error') != (attribution == 'error'):
        raise ValueError('Pipeline and attribution error statuses disagree')
    if attribution in ('ranked', 'abstained') and payload['decision'] != 'repair_required':
        raise ValueError('Attribution requires a divergence alarm')
    if attribution == 'not_requested' and payload['decision'] == 'repair_required':
        raise ValueError('A divergence alarm requires attribution or an explicit failure')
    rows = payload['candidates']
    if not isinstance(rows, list):
        raise ValueError('Candidates must be a list')
    if attribution == 'ranked':
        if not rows:
            raise ValueError('Ranked handoff needs candidates')
        unranked = []
        for row in rows:
            if not isinstance(row, dict) or type(row.get('rank')) is not int:
                raise ValueError('Candidate needs an integer rank')
            unranked.append({k: v for k, v in row.items() if k != 'rank'})
        normalized = validate_candidates({'candidates': unranked, 'abstention_reason': payload['abstention_reason']}, trajectory)
        if normalized['candidates'] != rows:
            raise ValueError('Candidates have inconsistent ranks or ordering')
    elif rows or not isinstance(payload['abstention_reason'], str) or not payload['abstention_reason'].strip():
        raise ValueError('Unavailable attribution needs no candidates and a reason')
    return payload


def load_handoff(path, trajectory, checks=None):
    return validate_handoff(read_json(path), trajectory, checks)
