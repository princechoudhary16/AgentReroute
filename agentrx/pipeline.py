"""Global/dynamic synthesis, guarded validation, and causal failure judging."""
from collections import Counter
from pathlib import Path
from .constraints import check_constraint, prefix, validate_constraint
from .io import write_json
from .perspectives import (PERSPECTIVES, compute_blame_matrix, encode_candidates, run_perspective_checks,
                           seed_constraints_for_perspective, synthesize_perspective_constraints)
from .rl_policy import DualLoopRLJudge, live_dynamic_recover, verify_and_score_state_change
from .taxonomy import category, agent

DEFAULTS = {'evidence': True, 'checklist': True, 'generation': 'stepwise', 'protocol': 'joint',
            'python_timeout': 3, 'recovery_loop': True}


def validate_config(config):
    result = {**DEFAULTS, **config}
    if result['generation'] not in ('stepwise', 'oneshot') or result['protocol'] not in ('joint', 'step_then_category', 'mp_rl'):
        raise ValueError('Unsupported generation/judge protocol')
    if type(result['evidence']) is not bool or type(result['checklist']) is not bool:
        raise ValueError('evidence/checklist must be booleans')
    if not isinstance(result['python_timeout'], (int,float)) or result['python_timeout'] <= 0:
        raise ValueError('python_timeout must be positive')
    if type(result['recovery_loop']) is not bool:
        raise ValueError('recovery_loop must be a boolean')
    if result.get('rl_state_path') is not None and (not isinstance(result['rl_state_path'], str) or not result['rl_state_path'].strip()):
        raise ValueError('rl_state_path must be a non-empty string')
    return result


def _constraints(answer):
    items = answer.get('constraints')
    if not isinstance(items, list):
        raise ValueError('Synthesis must return a constraints list')
    constraints = [validate_constraint(c) for c in items]
    names = [c['assertion_name'] for c in constraints]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate assertion names within synthesis output')
    return constraints


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


def diagnose(trajectory, policy, tools, config, provider, artifacts):
    config = validate_config(config)
    if config['protocol'] == 'mp_rl':
        return diagnose_mp_rl(trajectory, policy, tools, config, provider, artifacts)
    artifacts = Path(artifacts)
    checks, violations, errors = [], [], []
    result = {'trajectory_id': trajectory['trajectory_id'], 'domain': trajectory['domain'],
              'measurement_kind': provider.measurement_kind, 'config': config, 'status': 'ok',
              'prediction': None, 'checks': checks, 'violations': violations, 'errors': errors}
    write_json(artifacts/'trajectory_ir.json', trajectory)
    try:
        global_constraints = []
        if config['evidence']:
            global_constraints = _constraints(provider.complete('global',
                {'domain': trajectory['domain'], 'policy': policy, 'tools': tools}))
            write_json(artifacts/'global_constraints.json', global_constraints)
            oneshot = None
            if config['generation'] == 'oneshot':
                oneshot = _constraints(provider.complete('dynamic', {'trajectory': trajectory,
                    'policy': policy, 'tools': tools, 'global_constraints': global_constraints, 'generation': 'oneshot'}))
                write_json(artifacts/'dynamic_oneshot.json', oneshot)
            for offset, step in enumerate(trajectory['steps']):
                local = oneshot
                if local is None:
                    local = _constraints(provider.complete('dynamic', {'trajectory': prefix(trajectory, offset),
                        'policy': policy, 'tools': tools, 'global_constraints': global_constraints, 'generation': 'stepwise'}))
                    write_json(artifacts/'dynamic'/f'{step["index"]}.json', local)
                for origin, constraints in [('global',global_constraints), ('dynamic', local)]:
                    for constraint in constraints:
                        checked = check_constraint(constraint, trajectory, offset, provider,
                                                   {'policy': policy, 'tools': tools, 'python_timeout': config['python_timeout']})
                        checked['origin'] = origin
                        checks.append(checked)
                        if checked['verdict'] == 'VIOL': violations.append(checked)
                        if checked['verdict'] == 'ERROR': errors.append(checked)
        write_json(artifacts/'checks.json', checks)
        write_json(artifacts/'validation_log.json', violations)
        payload = {'trajectory': trajectory, 'policy': policy, 'tools': tools,
                   'checklist': config['checklist']}
        if config['evidence']:
            payload['violations'] = violations
            payload['checker_errors'] = [{'step_index': e['step_index'], 'assertion_name': e['assertion_name'],
                                          'error': e['error']} for e in errors]
        if config['protocol'] == 'step_then_category':
            selected = validate_prediction(provider.complete('localize', payload), trajectory, require_category=False)
            write_json(artifacts/'localization.json', selected)
            if selected.get('category') == 'Inconclusive':
                raw = selected
            else:
                raw = provider.complete('categorize', {**payload, 'selected_step': selected})
                # The second phase may categorize but cannot move the first phase's step/agent.
                if raw.get('step_index', raw.get('index')) != selected['step_index'] or agent(raw.get('agent')) != agent(selected['agent']):
                    raise ValueError('Category phase attempted to change localized step/agent')
        else:
            raw = provider.complete('judge', payload)
        result['prediction'] = validate_prediction(raw, trajectory)
        if errors:
            result['status'] = 'partial_checker_errors'
    except Exception as exc:
        result['status'] = 'error'
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['prediction'] = None
    finally:
        write_json(artifacts/'checks.json', checks)
        write_json(artifacts/'validation_log.json', violations)
        write_json(artifacts/'result.json', result)
    return result


def _compact(c):
    return {'agent': c['agent'], 'raw_agent': c['raw_agent'], 'step_index': c['step_index'],
            'category': c['dominant_category'], 'perspective': c['dominant_perspective'],
            'probability': c['probability'], 'logit': c['logit'], 'features': c['features']}


def _rca_payload(trajectory, result, ranked, violations):
    """Cross-branch contract read by recovery_orchestrator RCAPayload.from_dict()."""
    by_step = {}
    for v in violations:
        entry = by_step.setdefault(str(v['step_index']), {'count': 0, 'runs': 1, 'categories': []})
        entry['count'] += 1
        entry['categories'] = sorted(set(entry['categories']) | set(v.get('taxonomy_targets', [])))
    prediction = result['prediction']
    return {
        'trajectory_id': str(trajectory['trajectory_id']), 'domain': trajectory['domain'],
        'status': result['status'], 'confidence': float(ranked[0]['probability']) if ranked else 0.0,
        'n_runs': 1, 'total_runs': 1,
        'inconclusive_share': 0.0 if prediction and prediction['step_index'] is not None else 1.0,
        'top_k': [{'step_index': int(c['step_index']), 'agent': c['agent'], 'category': c['dominant_category'],
                   'votes': 1, 'share': float(c['probability']), 'rationale': c.get('rationale', ''),
                   'evidence_refs': c.get('evidence_refs', []), 'source': 'fa'} for c in ranked[:3]],
        'violations_by_step': by_step, 'prediction': prediction,
        'ranked_probabilities': [{'step_index': int(c['step_index']), 'agent': c['agent'],
                                  'category': c['dominant_category'], 'probability': float(c['probability'])}
                                 for c in ranked]}


def diagnose_mp_rl(trajectory, policy, tools, config, provider, artifacts, rl_judge=None):
    """Phase 2: 3-perspective checks -> blame matrix -> RL Top-1 (k*, t*) -> optional repair/verify/reward loop."""
    config = validate_config(config)
    artifacts = Path(artifacts)
    checks, violations, errors, ranked = [], [], [], []
    result = {'trajectory_id': trajectory['trajectory_id'], 'domain': trajectory['domain'],
              'measurement_kind': provider.measurement_kind, 'config': config, 'status': 'ok',
              'prediction': None, 'checks': checks, 'violations': violations, 'errors': errors,
              'ranked_candidates': [], 'recovery': {'status': 'not_run'}}
    recovery_log = {'status': 'not_run'}
    write_json(artifacts/'trajectory_ir.json', trajectory)
    try:
        if rl_judge is None:
            rl_judge = DualLoopRLJudge.load(config['rl_state_path']) if config.get('rl_state_path') else DualLoopRLJudge()
        context = {'policy': policy, 'tools': tools, 'python_timeout': config['python_timeout']}
        effective, views = {}, {}
        for pid, spec in PERSPECTIVES.items():
            if config['evidence']:
                constraints, synthesis = synthesize_perspective_constraints(
                    pid, trajectory, policy, tools, provider, rl_judge.memory)
            else:
                constraints, synthesis = seed_constraints_for_perspective(pid, tools), []
            rows, effective[pid] = run_perspective_checks(pid, constraints, trajectory, provider, context)
            checks.extend(rows)
            views[pid] = {'categories': list(spec['categories']), 'focus': spec['focus'],
                          'constraints': effective[pid], 'synthesis_log': synthesis,
                          'verdicts': dict(Counter(r['verdict'] for r in rows))}
        violations.extend(r for r in checks if r['verdict'] == 'VIOL')
        errors.extend(r for r in checks if r['verdict'] == 'ERROR')
        write_json(artifacts/'perspectives.json', views)
        write_json(artifacts/'checks.json', checks)
        write_json(artifacts/'validation_log.json', violations)
        blame = compute_blame_matrix(trajectory, checks)
        candidates = encode_candidates(trajectory, checks, blame)
        top1, ranked, probs = rl_judge.select_action(candidates, trajectory_id=trajectory['trajectory_id'])
        names = {v['assertion_name'] for v in top1['violated_checks'] if v['perspective'] == top1['dominant_perspective']}
        rl_judge.pending[str(trajectory['trajectory_id'])]['constraints_used'] = [
            c for c in effective[top1['dominant_perspective']] if c.get('origin') == 'llm' and c['assertion_name'] in names]
        write_json(artifacts/'blame_matrix.json', {'perspectives': list(PERSPECTIVES), 'matrix': blame,
                                                   'weights': list(rl_judge.weights), 'candidates': candidates})
        result['ranked_candidates'] = [_compact(c) for c in ranked]
        result['prediction'] = validate_prediction({
            'step_index': top1['step_index'], 'agent': top1['raw_agent'], 'category': top1['dominant_category'],
            'rationale': top1['rationale'], 'evidence_refs': top1['evidence_refs']}, trajectory)
        if errors:
            result['status'] = 'partial_checker_errors'
        if config['recovery_loop']:
            # Recovery is auxiliary to diagnosis: its failure is recorded but never voids the prediction.
            try:
                recovery = live_dynamic_recover(trajectory, top1, policy, tools, provider, rl_judge.memory)
                verification = verify_and_score_state_change(trajectory, checks, effective, recovery, top1,
                                                             candidates, probs, provider, context, rl_judge)
                recovery_log = {'status': 'ok', 'action': _compact(top1), 'recovery': recovery,
                                'verification': verification}
                result['recovery'] = {'status': 'ok', 'mode': recovery['mode'], 'feedback': verification['feedback'],
                                      'reward': verification['reward']}
            except Exception as exc:
                recovery_log = result['recovery'] = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
            if config.get('rl_state_path'):
                rl_judge.save(config['rl_state_path'])
    except Exception as exc:
        result['status'] = 'error'
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['prediction'] = None
    finally:
        write_json(artifacts/'checks.json', checks)
        write_json(artifacts/'validation_log.json', violations)
        write_json(artifacts/'rl_recovery.json', {**recovery_log, 'ranked_candidates': result['ranked_candidates']})
        write_json(artifacts/'rca_payload.json', _rca_payload(trajectory, result, ranked, violations))
        write_json(artifacts/'result.json', result)
    return result
