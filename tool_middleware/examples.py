"""Synthetic enterprise catalogs and controlled comparisons, not paper benchmarks."""
import json
from pathlib import Path
from . import Tool, Middleware, Blocked, catalog_hash


def obj(**properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def shipping():
    calls = []
    schema = obj(shipment_number={'type': 'string'})
    def validate(shipment_number):
        calls.append(['validate', shipment_number])
        return {'valid': shipment_number == 'SHIP-123'}
    def status(shipment_number):
        calls.append(['status', shipment_number])
        return {'status': 'in transit'}
    tools = [Tool('validate_shipment_number',
                  'Check whether a shipment number exists. The output valid is true only for an existing shipment.',
                  schema, validate, obj(valid={'type': 'boolean'}), 'read_only'),
             Tool('get_shipment_status',
                  'Get status for an existing shipment. Shipment number must first pass validate_shipment_number.',
                  schema, status, obj(status={'type': 'string'}), 'read_only')]
    return tools, calls, ['validate_shipment_number']


def employees():
    calls = []
    schema = obj(employee_id={'type': 'integer'})
    def verify(employee_id):
        calls.append(['verify', employee_id])
        return {'exists': employee_id == 7}
    def devices(employee_id):
        calls.append(['devices', employee_id])
        return {'devices': ['laptop']}
    tools = [Tool('verify_employee', 'Verify employee exists; exists=true means this ID belongs to an existing employee.',
                  schema, verify, obj(exists={'type': 'boolean'}), 'read_only'),
             Tool('retrieve_employee_devices', 'Retrieve devices; requires employee_id verified by verify_employee.',
                  schema, devices, obj(devices={'type': 'array', 'items': {'type': 'string'}}), 'read_only')]
    return tools, calls, ['verify_employee']


def fixture_plan(tools, target, validator, argument, success):
    return {'version': 1, 'catalog_hash': catalog_hash(tools),
            'rules': [{'id': 'validated_entity', 'target': target, 'validator': validator,
                       'bindings': {argument: {'from': '/' + argument}},
                       'success': [{'pointer': '/' + success, 'equals': True}],
                       'confidence': 'high', 'rationale': 'Explicit prerequisite in synthetic tool descriptions.'}],
            'abstentions': []}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def smoke(output, plan=None):
    tools, calls, validators = shipping()
    if plan is None:
        plan = fixture_plan(tools, 'get_shipment_status', validators[0], 'shipment_number', 'valid')
        kind = 'fixture; not a model measurement'
    else:
        kind = 'provided dependency plan; inspect its discovery provenance separately'
    layer = Middleware(tools, plan, validators=validators)
    valid = layer.dispatch('get_shipment_status', {'shipment_number': 'SHIP-123'})
    try:
        invalid = layer.dispatch('get_shipment_status', {'shipment_number': 'BAD'})
    except Blocked as exc:
        invalid = exc.to_dict()
    report = {'discovery': kind, 'valid_result': valid, 'invalid_result': invalid,
              'calls': calls, 'coverage': layer.coverage}
    root = Path(output)
    save(root / 'report.json', report)
    save(root / 'plan.json', plan)
    save(root / 'catalog.json', [{**tool.definition(), 'effect': tool.effect} for tool in tools])
    save(root / 'audit.json', layer.audit())
    return report


def evaluate(output):
    report = {'paper_benchmark': False, 'discovery': 'explicit fixture plans; no model measurement',
              'scope': 'Synthetic original-vs-wrapped tools with deliberately permissive original inputs', 'domains': {}}
    configurations = [('shipping', shipping, 'get_shipment_status', 'shipment_number', 'valid', 'status', 'SHIP-123', 'BAD', {'status': 'in transit'}),
                      ('employee_devices', employees, 'retrieve_employee_devices', 'employee_id', 'exists', 'devices', 7, 8, {'devices': ['laptop']})]
    for domain, factory, target, argument, predicate, target_label, good, bad, expected in configurations:
        results = {}
        for condition in ('original', 'wrapped'):
            tools, calls, validators = factory()
            plan = fixture_plan(tools, target, validators[0], argument, predicate)
            layer = Middleware(tools, plan, validators=validators)
            original = next(tool.handler for tool in tools if tool.name == target)
            valid_completed = false_blocks = blocked = 0
            for value in (good, bad, good, bad):
                try:
                    result = original(**{argument: value}) if condition == 'original' else layer.dispatch(target, {argument: value})
                    if value == good and result == expected:
                        valid_completed += 1
                except Blocked:
                    blocked += 1
                    false_blocks += value == good
            results[condition] = {'target_invocations': sum(name == target_label for name, _ in calls),
                                  'invalid_target_invocations': sum(name == target_label and value == bad for name, value in calls),
                                  'extra_validator_calls': sum(name != target_label for name, _ in calls),
                                  'valid_completed': valid_completed, 'false_blocks': false_blocks, 'blocked': blocked}
        report['domains'][domain] = results
    save(Path(output) / 'evaluation.json', report)
    return report
