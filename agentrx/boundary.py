"""Conclusive, mechanically deduplicated rule/step divergence measurements."""
import ast
import math
from .io import digest


def validate_threshold(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('divergence_threshold must be finite and between 0 and 1')
    return value


def constraint_fingerprint(constraint):
    kind = constraint['check_type']
    if kind == 'python_check':
        code = constraint['python_check']
        source = '\n'.join(code['code_lines'])
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == code['function_name']:
                    node.name = '_checker'
                elif isinstance(node, ast.Name) and node.id == code['function_name']:
                    node.id = '_checker'
            definition = ast.dump(tree, include_attributes=False)
        except (SyntaxError, ValueError):
            definition = {'source': source, 'function_name': code['function_name']}
    elif kind == 'nl_check':
        definition = {'nl_check': constraint['nl_check'], 'check_hint': constraint['check_hint']}
    else:
        raise ValueError('Unsupported check type')
    return digest({'check_type': kind, 'event_trigger': constraint.get('event_trigger', {}),
                   'definition': definition})


def compute_boundary(checks, threshold):
    validate_threshold(threshold)
    groups = {}
    for offset, check in enumerate(checks):
        index, fingerprint = check.get('step_index'), check.get('rule_fingerprint')
        if type(index) is not int:
            raise ValueError('Check step must be an integer')
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            if check.get('verdict') in ('SAT', 'VIOL'):
                raise ValueError('Conclusive checks need a rule fingerprint')
            fingerprint = ('unfingerprinted', offset)
        groups.setdefault((index, fingerprint), []).append(check)
    counts = dict(violated_checks=0, satisfied_checks=0, skipped_checks=0,
                  error_checks=0, unresolved_checks=0, conflicting_checks=0)
    for rows in groups.values():
        conclusive = set()
        for row in rows:
            verdict = row.get('verdict')
            unclear = row.get('unclear_count', 0) or any(
                r.get('evaluation') == 'UNCLEAR' for r in row.get('rubric_results', []))
            if verdict == 'VIOL' or (verdict == 'SAT' and not unclear):
                conclusive.add(verdict)
        if len(conclusive) > 1:
            counts['conflicting_checks'] += 1
            key = 'unresolved_checks'
        elif conclusive:
            key = 'violated_checks' if 'VIOL' in conclusive else 'satisfied_checks'
        elif any(r.get('verdict') not in ('SKIP', 'ERROR') for r in rows):
            key = 'unresolved_checks'
        elif any(r.get('verdict') == 'ERROR' for r in rows):
            key = 'error_checks'
        else:
            key = 'skipped_checks'
        counts[key] += 1
    denominator = counts['violated_checks'] + counts['satisfied_checks']
    return {'threshold': threshold, 'comparison': '>', **counts, 'conclusive_checks': denominator,
            'check_evaluations': len(checks), 'unique_checks': len(groups),
            'duplicate_evaluations': len(checks) - len(groups),
            'divergence_rate': counts['violated_checks'] / denominator if denominator else None}


def boundary_decision(boundary):
    rate = boundary['divergence_rate']
    if rate is None:
        return 'inconclusive'
    return 'repair_required' if rate > boundary['threshold'] else 'within_boundary'
