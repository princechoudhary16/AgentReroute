"""Tool metadata and compilation of small, declarative prerequisite plans."""
import copy
import hashlib
import json
import math
from dataclasses import dataclass, replace


class PlanError(ValueError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: object
    output_schema: dict = None
    effect: str = 'unknown'

    def definition(self):
        result = {'name': self.name, 'description': self.description,
                  'input_schema': copy.deepcopy(self.input_schema)}
        if self.output_schema is not None:
            result['output_schema'] = copy.deepcopy(self.output_schema)
        return result

    def snapshot(self):
        return replace(self, input_schema=copy.deepcopy(self.input_schema),
                       output_schema=copy.deepcopy(self.output_schema))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def catalog_hash(tools):
    metadata = [{**tool.definition(), 'effect': tool.effect} for tool in tools]
    return hashlib.sha256(encoded(sorted(metadata, key=lambda t: t['name'])).encode()).hexdigest()


def tokens(pointer):
    if not isinstance(pointer, str) or (pointer and not pointer.startswith('/')):
        raise PlanError('Expected an RFC 6901 JSON pointer')
    parts = pointer.split('/')[1:] if pointer else []
    for part in parts:
        for index, char in enumerate(part):
            if char == '~' and (index + 1 == len(part) or part[index + 1] not in '01'):
                raise PlanError('Invalid JSON pointer escape')
    return [part.replace('~1', '/').replace('~0', '~') for part in parts]


def at(value, pointer):
    for part in tokens(pointer):
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and (part == '0' or not part.startswith('0')):
            value = value[int(part)]
        else:
            raise KeyError(pointer)
    return value


def schema_at(schema, pointer):
    for part in tokens(pointer):
        if schema.get('type') == 'object' and part in schema.get('properties', {}):
            if part not in schema.get('required', []):
                raise PlanError('Bound/evidence paths must be required in their schemas')
            schema = schema['properties'][part]
        elif schema.get('type') == 'array' and part.isdigit():
            schema = schema['items']
        else:
            raise PlanError(f'Undeclared schema path {pointer}')
    return schema


ANNOTATIONS = {'description', 'title', 'default', 'examples', 'readOnly', 'writeOnly', 'deprecated', '$schema'}
KEYS = ANNOTATIONS | {'type', 'properties', 'required', 'additionalProperties', 'items', 'enum', 'const',
                      'minimum', 'maximum', 'minLength', 'maxLength', 'minItems', 'maxItems'}
TYPES = {'object', 'array', 'string', 'integer', 'number', 'boolean', 'null'}


def supported(schema):
    if not isinstance(schema, dict) or set(schema) - KEYS or schema.get('type') not in TYPES:
        raise PlanError('Activated validation requires the supported, explicit JSON schema subset')
    if schema['type'] == 'object':
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        if (not isinstance(properties, dict) or not isinstance(required, list)
                or any(not isinstance(name, str) for name in required) or not set(required) <= set(properties)
                or type(schema.get('additionalProperties', True)) is not bool):
            raise PlanError('Invalid object schema')
        for child in properties.values():
            supported(child)
    if schema['type'] == 'array':
        supported(schema.get('items'))
    for key in ('minLength', 'maxLength', 'minItems', 'maxItems'):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise PlanError('Invalid schema bound')
    for key in ('minimum', 'maximum'):
        if key in schema and (type(schema[key]) not in (int, float) or not math.isfinite(schema[key])):
            raise PlanError('Invalid numeric schema bound')
    if 'enum' in schema and (not isinstance(schema['enum'], list) or not schema['enum']):
        raise PlanError('Invalid schema enum')


def equal(left, right):
    # JSON numeric equality is allowed, but a boolean is never the number 1/0.
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return encoded(left) == encoded(right) or (type(left) in (int, float) and type(right) in (int, float) and left == right)


def validate(value, schema):
    kind = schema['type']
    valid = {'object': isinstance(value, dict), 'array': isinstance(value, list),
             'string': isinstance(value, str), 'integer': type(value) is int,
             'number': type(value) in (int, float), 'boolean': type(value) is bool,
             'null': value is None}[kind]
    if not valid:
        raise ValueError(f'Expected {kind}')
    if 'enum' in schema and not any(equal(value, item) for item in schema['enum']):
        raise ValueError('Value outside enum')
    if 'const' in schema and not equal(value, schema['const']):
        raise ValueError('Value differs from const')
    if kind == 'object':
        properties = schema.get('properties', {})
        if not set(schema.get('required', [])) <= set(value):
            raise ValueError('Missing required fields')
        if schema.get('additionalProperties') is False and set(value) - set(properties):
            raise ValueError('Unexpected fields')
        for key in set(value) & set(properties):
            validate(value[key], properties[key])
    elif kind == 'array':
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', float('inf')):
            raise ValueError('Array outside bounds')
        for child in value:
            validate(child, schema['items'])
    elif kind == 'string':
        if len(value) < schema.get('minLength', 0) or len(value) > schema.get('maxLength', float('inf')):
            raise ValueError('String outside bounds')
    elif kind in ('integer', 'number'):
        if not math.isfinite(value) or value < schema.get('minimum', -float('inf')) or value > schema.get('maximum', float('inf')):
            raise ValueError('Number outside bounds')


def compile_plan(tools, plan, validators):
    tools = list(tools)
    names = [tool.name for tool in tools]
    if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise PlanError('Tool names must be unique nonempty strings')
    registry = {tool.name: tool.snapshot() for tool in tools}
    for tool in registry.values():
        if not callable(tool.handler) or tool.effect not in ('read_only', 'write', 'unknown'):
            raise PlanError('Tools need callable handlers and an explicit effect classification')
    allowlist = set(validators)
    if not allowlist <= set(names):
        raise PlanError('Validator designation references unknown tools')
    if not isinstance(plan, dict) or set(plan) != {'version', 'catalog_hash', 'rules', 'abstentions'}:
        raise PlanError('Expected a versioned dependency plan')
    if type(plan['version']) is not int or plan['version'] != 1 or plan['catalog_hash'] != catalog_hash(tools):
        raise PlanError('Plan version/catalog mismatch; rediscover against the current catalog')
    if not isinstance(plan['rules'], list) or not isinstance(plan['abstentions'], list):
        raise PlanError('Rules and abstentions must be lists')
    if len(plan['rules']) > 512:
        raise PlanError('A plan may activate at most 512 rules')
    grouped = {name: [] for name in names}
    ids = set()
    for original in plan['rules']:
        rule = copy.deepcopy(original)
        keys = {'id', 'target', 'validator', 'bindings', 'success', 'confidence', 'rationale'}
        if not isinstance(rule, dict) or set(rule) != keys:
            raise PlanError('Unsupported rule structure')
        if not isinstance(rule['id'], str) or not rule['id'] or rule['id'] in ids:
            raise PlanError('Rule IDs must be unique nonempty strings')
        ids.add(rule['id'])
        if (not isinstance(rule['target'], str) or not isinstance(rule['validator'], str)
                or rule['target'] not in registry or rule['validator'] not in registry):
            raise PlanError('Rule references unknown tool')
        target, prerequisite = registry[rule['target']], registry[rule['validator']]
        if prerequisite.name not in allowlist or prerequisite.effect != 'read_only':
            raise PlanError('Only explicitly designated read-only validators may be automated')
        if rule['confidence'] != 'high' or not isinstance(rule['rationale'], str) or not rule['rationale'].strip():
            raise PlanError('Only grounded high-confidence rules may be activated')
        supported(prerequisite.input_schema)
        supported(prerequisite.output_schema)
        if prerequisite.input_schema['type'] != 'object':
            raise PlanError('Callable validator inputs must be an object')
        bindings = rule['bindings']
        properties = prerequisite.input_schema.get('properties', {})
        if (not isinstance(bindings, dict) or not set(prerequisite.input_schema.get('required', [])) <= set(bindings)
                or not set(bindings) <= set(properties)):
            raise PlanError('Bindings must cover required validator inputs and reference declared fields')
        for name, binding in bindings.items():
            supported(properties[name])
            if not isinstance(binding, dict):
                raise PlanError('Invalid binding')
            if set(binding) == {'from'}:
                source = schema_at(target.input_schema, binding['from'])
                supported(source)
                if source['type'] != properties[name]['type']:
                    raise PlanError('Binding field types differ')
            elif set(binding) == {'literal'}:
                try:
                    encoded(binding['literal'])
                    validate(binding['literal'], properties[name])
                except (ValueError, TypeError):
                    raise PlanError('Literal does not match validator schema') from None
            else:
                raise PlanError('Bindings may only copy a target field or provide a literal')
        shared = set(target.input_schema.get('required', [])) & set(prerequisite.input_schema.get('required', []))
        for name in shared:
            pointer = '/' + name.replace('~', '~0').replace('/', '~1')
            if bindings.get(name) != {'from': pointer}:
                raise PlanError('Shared required scope/identity fields must retain exact bindings')
        if not isinstance(rule['success'], list) or not rule['success']:
            raise PlanError('A success predicate is required')
        for predicate in rule['success']:
            if not isinstance(predicate, dict) or set(predicate) != {'pointer', 'equals'}:
                raise PlanError('Only explicit literal equality predicates are supported')
            field = schema_at(prerequisite.output_schema, predicate['pointer'])
            try:
                encoded(predicate['equals'])
                validate(predicate['equals'], field)
            except (ValueError, TypeError):
                raise PlanError('Success literal does not match its declared output field') from None
        grouped[target.name].append(rule)
    depths, visiting = {}, set()
    def visit(name):
        if name in visiting:
            raise PlanError('Prerequisite graph contains a cycle')
        if name in depths:
            return depths[name]
        visiting.add(name)
        depth = 1 + max((visit(rule['validator']) for rule in grouped[name]), default=0)
        visiting.remove(name)
        if depth > 8:
            raise PlanError('Prerequisite graph exceeds maximum depth 8')
        depths[name] = depth
        return depth
    for name in names:
        visit(name)
    return registry, grouped
