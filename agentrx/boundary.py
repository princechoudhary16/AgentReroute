"""Conclusive, mechanically deduplicated rule/step divergence measurements."""
import ast
import math
import re
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
            name = code['function_name']
            nodes = list(ast.walk(tree))
            definitions = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
            # Only the unused entry-point label is harmless. References, shadowing,
            # rebinding or multiple definitions retain the original AST identity.
            ambiguous = any((isinstance(n, ast.Name) and n.id == name) or
                            (isinstance(n, ast.arg) and n.arg == name) or
                            (isinstance(n, ast.alias) and (n.asname or n.name.split('.')[0]) == name) or
                            (isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names) or
                            (isinstance(n, ast.ClassDef) and n.name == name) for n in nodes)
            if len(definitions) == 1 and not ambiguous:
                definitions[0].name = '<checker-entry-point>'  # Cannot collide with a Python identifier.
            definition = ast.dump(tree, include_attributes=False)
        except (SyntaxError, ValueError):
            definition = {'source': source, 'function_name': code['function_name']}
    elif kind == 'nl_check':
        definition = {'nl_check': constraint['nl_check'], 'check_hint': constraint['check_hint']}
    else:
        raise ValueError('Unsupported check type')
    return digest({'check_type': kind, 'event_trigger': _canonical_guard(constraint.get('event_trigger', constraint.get('trigger', {}))),
                   'definition': definition})


def _canonical_guard(trigger):
    guard = {}
    for key in ('step_index', 'substep_index'):
        value = trigger.get(key)
        if value in (None, '*'):
            continue
        if type(value) is int:
            ranges = [(value, value)]
        elif isinstance(value, list):
            ranges = [(v, v) for v in value]
        elif isinstance(value, str) and value.isdigit():
            ranges = [(int(value), int(value))]
        else:
            match = re.fullmatch(r'(\d+)\s*(?:-|:|\.\.)\s*(\d+)', str(value))
            if not match:
                raise ValueError('Unsupported index selector')
            ranges = [(int(match[1]), int(match[2]))]
        merged = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        guard[key] = merged
    for key in ('role_name', 'tool_name', 'content_regex'):
        value = trigger.get(key, trigger.get('agent_name') if key == 'role_name' else None)
        if value not in (None, '*'):
            guard[key] = str(value)
    return guard


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
