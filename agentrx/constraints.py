"""Appendix F constraints, event guards, and auditable checker outcomes."""
import copy
import json
import re
import subprocess
import sys
from pathlib import Path
from .taxonomy import category
from .boundary import constraint_fingerprint

TYPES = {'SCHEMA', 'PROTOCOL', 'RELATIONAL_POST', 'PROVENANCE', 'TEMPORAL', 'CAPABILITY', 'ANY'}


def _selector(value, index):
    if value in (None, '*'):
        return True
    if type(value) is int:
        if value < 0:
            raise ValueError('Index selectors must be nonnegative')
        return value == index
    if isinstance(value, list):
        if any(type(item) is not int or item < 0 for item in value):
            raise ValueError('Index lists require nonnegative integers, excluding booleans')
        return index in value
    if isinstance(value, str):
        if value.isdigit():
            return int(value) == index
        match = re.fullmatch(r'(\d+)\s*(?:-|:|\.\.)\s*(\d+)', value)
        if match:
            if int(match[1]) > int(match[2]):
                raise ValueError('Index range start exceeds end')
            return int(match[1]) <= index <= int(match[2])
    raise ValueError(f'Unsupported index selector {value!r}')


def _pattern(pattern, value, search=False):
    if pattern in (None, '*'):
        return True
    return bool((re.search if search else re.fullmatch)(str(pattern), str(value)))


def validate_constraint(raw):
    if not isinstance(raw, dict):
        raise ValueError('Constraint must be an object')
    c = copy.deepcopy(raw)
    name = c.get('assertion_name', '')
    if not isinstance(name, str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]*', name):
        raise ValueError('Invalid assertion_name')
    targets = c.get('taxonomy_targets', [])
    if not isinstance(targets, list) or not targets:
        raise ValueError('Constraint needs taxonomy_targets')
    c['taxonomy_targets'] = [category(t) for t in targets]
    c['constraint_type'] = c.get('constraint_type', c.get('invariant_type', c.get('type', 'ANY')))
    if c['constraint_type'] not in TYPES:
        raise ValueError('Unsupported constraint type')
    trigger = c.get('event_trigger', c.get('trigger', {}))
    if not isinstance(trigger, dict):
        raise ValueError('event_trigger must be an object')
    trigger = copy.deepcopy(trigger)
    if 'agent_name' in trigger:
        trigger.setdefault('role_name', trigger['agent_name'])
    for key in ('step_index', 'substep_index'):
        _selector(trigger.get(key), 0)
    for key in ('role_name', 'tool_name', 'content_regex'):
        pattern = trigger.get(key)
        if pattern not in (None, '*'):
            try:
                re.compile(str(pattern))
            except re.error as exc:
                raise ValueError(f'Invalid {key} regex: {exc}') from exc
    c['event_trigger'] = trigger
    if not isinstance(c.get('check_hint'), str) or not c['check_hint'].strip():
        raise ValueError('Constraint needs a non-empty check_hint')
    if c.get('check_type') == 'python_check':
        code = c.get('python_check', {})
        if not isinstance(code, dict) or not isinstance(code.get('code_lines'), list) or not all(
                isinstance(s, str) for s in code['code_lines']):
            raise ValueError('Python check needs code_lines')
        if not re.fullmatch('[A-Za-z][A-Za-z0-9_]*', str(code.get('function_name', ''))):
            raise ValueError('Python check needs function_name')
    elif c.get('check_type') == 'nl_check':
        rubric = c.get('nl_check', {}).get('judge_rubric', [])
        if not isinstance(rubric, list) or not rubric or not all(isinstance(x, str) and x.strip() for x in rubric):
            raise ValueError('Semantic check needs a non-empty judge_rubric')
    else:
        raise ValueError('Unsupported check_type')
    return c


def guard_matches(c, step):
    trigger = c.get('event_trigger', {})
    if not _selector(trigger.get('step_index'), step['index']):
        return False
    for event in step['substeps']:
        if not _selector(trigger.get('substep_index'), event['sub_index']):
            continue
        if not _pattern(trigger.get('role_name'), event['role']):
            continue
        tools = [call['name'] for call in event.get('tool_calls', [])]
        if event.get('name'):
            tools.append(event['name'])
        if trigger.get('tool_name') not in (None, '*') and not any(
                _pattern(trigger['tool_name'], tool) for tool in tools):
            continue
        content = event['content']
        if event.get('tool_calls'):
            content += '\n' + json.dumps(event['tool_calls'], ensure_ascii=False)
        if not _pattern(trigger.get('content_regex'), content, search=True):
            continue
        return True
    return False


def prefix(trajectory, offset):
    result = copy.deepcopy(trajectory)
    result['steps'] = result['steps'][:offset+1]
    if 'index_map' in result:
        result['index_map'] = result['index_map'][:offset+1]
    return result


def _python(c, trajectory, offset, timeout):
    request = {'code': '\n'.join(c['python_check']['code_lines']),
               'function_name': c['python_check']['function_name'],
               'trajectory': trajectory, 'offset': offset}
    try:
        process = subprocess.run([sys.executable, '-I', '-S', str(Path(__file__).with_name('worker.py'))],
                                 input=json.dumps(request), text=True, capture_output=True,
                                 timeout=timeout, env={})
    except subprocess.TimeoutExpired:
        return {'error': 'Python check timed out'}
    if process.returncode or len(process.stdout) > 65_536:
        return {'error': 'Python worker failed or exceeded output limit'}
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError:
        return {'error': 'Python worker returned invalid JSON'}


def check_constraint(raw, trajectory, offset, provider, context):
    result = {'step_index': trajectory['steps'][offset]['index'],
              'assertion_name': raw.get('assertion_name', '<invalid>'), 'verdict': 'ERROR'}
    try:
        c = validate_constraint(raw)
        result['rule_fingerprint'] = constraint_fingerprint(c)
        result.update(taxonomy_targets=c['taxonomy_targets'], constraint_type=c['constraint_type'],
                      check_type=c['check_type'], check_hint=c['check_hint'])
        if not guard_matches(c, trajectory['steps'][offset]):
            result['verdict'] = 'SKIP'
            return result
        observed = prefix(trajectory, offset)
        result['evidence'] = observed['steps'][max(0, offset-3):offset+1]
        if c['check_type'] == 'python_check':
            checked = _python(c, observed, offset, context.get('python_timeout', 3))
            if 'error' in checked:
                raise ValueError(checked['error'])
            result['verdict'] = 'SAT' if checked['value'] else 'VIOL'
        else:
            answer = provider.complete('semantic', {'constraint': c, 'trajectory': observed,
                'current_step_index': result['step_index'], 'current_offset': offset,
                'policy': context.get('policy', ''), 'tools': context.get('tools', {})})
            rows = answer.get('rubric_results')
            rubric = c['nl_check']['judge_rubric']
            if not isinstance(rows, list) or len(rows) != len(rubric):
                raise ValueError('Semantic response must evaluate every rubric criterion')
            indices = []
            for row in rows:
                if not isinstance(row, dict) or type(row.get('criterion_index')) is not int:
                    raise ValueError('Missing criterion index')
                indices.append(row['criterion_index'])
                if row.get('evaluation') not in ('CLEAR_PASS', 'CLEAR_FAIL', 'UNCLEAR'):
                    raise ValueError('Unknown rubric evaluation')
                if not isinstance(row.get('reasoning'), str) or not row['reasoning'].strip():
                    raise ValueError('Rubric evaluation needs reasoning')
            if sorted(indices) != list(range(len(rubric))):
                raise ValueError('Duplicate or invalid rubric indices')
            result['rubric_results'] = rows
            result['unclear_count'] = sum(r['evaluation'] == 'UNCLEAR' for r in rows)
            result['verdict'] = 'VIOL' if any(r['evaluation'] == 'CLEAR_FAIL' for r in rows) else 'SAT'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result
