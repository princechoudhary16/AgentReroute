"""Global/dynamic synthesis, guarded validation, and causal failure judging."""
from pathlib import Path
from .constraints import check_constraint, prefix, validate_constraint
from .io import write_json
from .taxonomy import agent
from .predictions import validate_prediction

DEFAULTS = {'evidence': True, 'checklist': True, 'generation': 'stepwise', 'protocol': 'joint',
            'python_timeout': 3}


def validate_config(config):
    result = {**DEFAULTS, **config}
    if result['generation'] not in ('stepwise', 'oneshot') or result['protocol'] not in ('joint', 'step_then_category'):
        raise ValueError('Unsupported generation/judge protocol')
    if type(result['evidence']) is not bool or type(result['checklist']) is not bool:
        raise ValueError('evidence/checklist must be booleans')
    if not isinstance(result['python_timeout'], (int,float)) or result['python_timeout'] <= 0:
        raise ValueError('python_timeout must be positive')
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



def diagnose(trajectory, policy, tools, config, provider, artifacts):
    config = validate_config(config)
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
