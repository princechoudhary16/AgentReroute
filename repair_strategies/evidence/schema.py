"""Validation of tool calls against their JSON schemas (the subset tool schemas use)."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from ..context import TaskSpec, ToolSpec
from ..trace import ToolCall

UNKNOWN_TOOL, MALFORMED, MISSING, UNEXPECTED, TYPE, ENUM, PATTERN, RANGE, LENGTH = (
    'unknown_tool', 'malformed_arguments', 'missing_argument', 'unexpected_argument', 'wrong_type',
    'not_in_enum', 'pattern_mismatch', 'out_of_range', 'wrong_length')

_TYPES = {'string': str, 'integer': int, 'number': (int, float), 'boolean': bool, 'array': list,
          'object': dict, 'null': type(None)}


@dataclass(frozen=True)
class Issue:
    path: str
    code: str
    message: str
    suggestion: str | None = None

    def describe(self) -> str:
        hint = f' (did you mean "{self.suggestion}"?)' if self.suggestion else ''
        return f'{self.path}: {self.message}{hint}'


def suggest(name: str, options) -> str | None:
    match = difflib.get_close_matches(name, list(options), n=1, cutoff=0.6)
    return match[0] if match else None


def _type_ok(value, expected) -> bool:
    names = expected if isinstance(expected, list) else [expected]
    for name in names:
        py = _TYPES.get(name)
        if py is None:
            return True                  # unknown type keyword: do not invent a violation
        if name in ('integer', 'number') and isinstance(value, bool):
            continue
        if isinstance(value, py):
            return True
    return False


def validate(value, schema: dict, path: str = '$', strict_properties: bool = False) -> list[Issue]:
    issues: list[Issue] = []
    if not isinstance(schema, dict):
        return issues
    expected = schema.get('type')
    if expected is not None and not _type_ok(value, expected):
        return [Issue(path, TYPE, f'expected {expected}, got {type(value).__name__}')]
    if 'enum' in schema and value not in schema['enum']:
        options = [str(v) for v in schema['enum']]
        issues.append(Issue(path, ENUM, f'must be one of {options}',
                            suggest(str(value), options) if isinstance(value, str) else None))
    if isinstance(value, str):
        if 'pattern' in schema and not re.search(schema['pattern'], value):
            issues.append(Issue(path, PATTERN, f'does not match {schema["pattern"]}'))
        if len(value) < schema.get('minLength', 0) or len(value) > schema.get('maxLength', len(value)):
            issues.append(Issue(path, LENGTH, 'string length out of bounds'))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if 'minimum' in schema and value < schema['minimum'] or 'maximum' in schema and value > schema['maximum']:
            issues.append(Issue(path, RANGE, 'number out of range'))
    if isinstance(value, list):
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', len(value)):
            issues.append(Issue(path, LENGTH, 'array length out of bounds'))
        for i, item in enumerate(value):
            issues.extend(validate(item, schema.get('items', {}), f'{path}[{i}]', strict_properties))
    if isinstance(value, dict):
        properties = schema.get('properties', {}) or {}
        for name in schema.get('required', ()) or ():
            if name not in value:
                issues.append(Issue(f'{path}.{name}', MISSING, 'required argument is missing'))
        additional = schema.get('additionalProperties', not strict_properties)
        for name, item in value.items():
            if name in properties:
                issues.extend(validate(item, properties[name], f'{path}.{name}', strict_properties))
            elif additional is False:
                issues.append(Issue(f'{path}.{name}', UNEXPECTED, 'argument is not accepted',
                                    suggest(name, properties)))
    return issues


def validate_call(call: ToolCall, task: TaskSpec, allowed: tuple[str, ...] | None = None) -> list[Issue]:
    """Issues with one call. ``allowed`` restricts the tools the calling agent may use."""
    names = allowed if allowed else tuple(task.tools)
    spec = task.tool(call.name)
    if spec is None or call.name not in names:
        return [Issue('$.name', UNKNOWN_TOOL, f'"{call.name}" is not an available tool', suggest(call.name, names))]
    if '__raw__' in call.arguments:
        return [Issue('$.arguments', MALFORMED, 'arguments are not a valid JSON object')]
    return validate(call.arguments, dict(spec.parameters), '$', strict_properties=True)


def example_arguments(spec: ToolSpec) -> dict:
    """A placeholder call body showing the required shape: {"order_id": "<string>", ...}."""
    def sample(schema):
        kind = schema.get('type') if isinstance(schema, dict) else None
        if isinstance(kind, list):
            kind = kind[0]
        if 'enum' in (schema or {}):
            return schema['enum'][0]
        if kind == 'array':
            return [sample(schema.get('items', {}))]
        if kind == 'object':
            return {k: sample(v) for k, v in (schema.get('properties') or {}).items()}
        return f'<{kind or "value"}>'
    props = spec.properties
    return {name: sample(props.get(name, {})) for name in (spec.required or tuple(props))}
