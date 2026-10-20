"""Validation of legacy predictions and grounded ranked hypotheses."""
import math
from .taxonomy import category, agent, CATEGORIES

def validate_prediction(raw, trajectory, require_category=True):
    if not isinstance(raw, dict):
        raise ValueError('Prediction must be an object')
    rationale = raw.get('rationale', raw.get('reason_for_failure', ''))
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError('Prediction needs an evidence-grounded rationale')
    label = category(raw.get('category', raw.get('failure_case'))) if (
        require_category or raw.get('category') == 'Inconclusive') else None
    index = raw.get('step_index', raw.get('index'))
    role = raw.get('agent', raw.get('failed_agent'))
    if not require_category and index is None and str(raw.get('custom_category', '')).strip():
        label = 'Inconclusive'
    if label == 'Inconclusive' and index is None:
        if not str(raw.get('custom_category', '')).strip():
            raise ValueError('Inconclusive prediction needs custom_category')
        return {'step_index': None, 'agent': None, 'category': label, 'rationale': rationale,
                'evidence_refs': [], 'custom_category': raw['custom_category']}
    step = next((s for s in trajectory['steps'] if s['index'] == index and type(index) is int), None)
    if step is None:
        raise ValueError('Prediction step is absent from trajectory')
    known_agents = {agent(e['role']) for s in trajectory['steps'] for e in s['substeps']}
    if not isinstance(role, str) or not role.strip() or agent(role) not in known_agents:
        raise ValueError('Prediction agent is absent from trajectory')
    refs = raw.get('evidence_refs', [])
    if not isinstance(refs, list):
        raise ValueError('evidence_refs must be a list')
    pairs = {(s['index'], e['sub_index']) for s in trajectory['steps'] for e in s['substeps']}
    for ref in refs:
        if not isinstance(ref, dict) or (ref.get('step_index'), ref.get('substep_index')) not in pairs:
            raise ValueError('Prediction evidence refers to absent event')
    result = {'step_index': index, 'agent': role, 'rationale': rationale, 'evidence_refs': refs}
    if label:
        if label == 'Inconclusive' and not str(raw.get('custom_category', '')).strip():
            raise ValueError('Inconclusive prediction needs custom_category')
        result['category'] = label
        if label == 'Inconclusive': result['custom_category'] = raw['custom_category']
    return result



def validate_candidates(raw, trajectory, max_candidates=3):
    if type(max_candidates) is not int or max_candidates not in (2, 3):
        raise ValueError('max_candidates must be 2 or 3')
    if not isinstance(raw, dict) or set(raw) != {'candidates', 'abstention_reason'}:
        raise ValueError('Ranking needs candidates and abstention_reason only')
    rows, reason = raw['candidates'], raw['abstention_reason']
    if not isinstance(rows, list) or len(rows) > max_candidates:
        raise ValueError('Invalid candidate count')
    if not rows:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Empty ranking needs an abstention reason')
        return {'candidates': [], 'abstention_reason': reason}
    if reason is not None:
        raise ValueError('Nonempty ranking cannot abstain')
    candidates, seen = [], set()
    fields = {'step_index','agent','category','confidence','rationale','evidence_refs'}
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError('Candidate fields do not match the contract')
        confidence = row['confidence']
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Confidence must be finite and between 0 and 1')
        if row['category'] not in CATEGORIES[:-1]:
            raise ValueError('Candidate needs a canonical failure category')
        refs = row['evidence_refs']
        if not isinstance(refs, list) or not refs:
            raise ValueError('Candidate needs evidence references')
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {'step_index','substep_index'} or any(type(v) is not int for v in ref.values()):
                raise ValueError('Evidence indices must be integers')
        selected = validate_prediction(row, trajectory)
        if selected['step_index'] in seen:
            raise ValueError('Candidates must identify distinct steps')
        seen.add(selected['step_index'])
        candidates.append({**selected, 'confidence': confidence})
    candidates.sort(key=lambda c: (-c['confidence'], c['step_index']))
    return {'candidates': [dict(c, rank=i) for i, c in enumerate(candidates, 1)], 'abstention_reason': None}
