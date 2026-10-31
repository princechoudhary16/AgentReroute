"""The structural replacements for the injector-coupled seed regexes.

Covers what the seeds now key on instead of wording: a call's arguments against the tool's declared
schema, a tool name against the known vocabulary (and the refusal to judge names when the vocabulary is
unknown), an identical call returning an identical result, and a failure that has already happened once.

Two implementations are deliberately held in step here, because neither can import the other:

* the stuck-loop clause inside the generated p2 seed runs as source text in the worker sandbox, while
  ``perspectives.repeats_earlier_tool_result`` is its Python twin for callers that classify a ranked
  candidate (``rl_cross_branch._fault_label``);
* ``repair_strategies/evidence/schema.py`` already validates calls against schemas and is tested, but
  the sandbox forbids its imports (``worker.check_source`` allows a fixed module whitelist and rejects
  class definitions and private names), so the seed cannot call it. The agreement test below pins the
  key-level verdicts instead of letting the two drift.
"""
import json
import unittest

from agentrx.data import normalize
from agentrx.perspectives import (repeats_earlier_tool_result, run_perspective_checks,
                                  seed_constraints_for_perspective, tool_catalog)

# External AgentRx release shape: inputs as name -> type description, with no optionality marker.
INPUTS_TOOLS = {'tools': {
    'cancel_pending_order': {'capability': 'cancel', 'inputs': {'order_id': 'string', 'reason': 'string'}},
    'modify_pending_order_items': {'capability': 'modify', 'inputs': {
        'order_id': 'string', 'item_ids_to_add': 'list', 'item_ids_to_remove': 'list'}},
    'get_order_details': {'capability': 'read', 'inputs': {'order_id': 'string'}}}}

# Internal benchmark shape: JSON Schema under parameters, which does state required.
PARAM_TOOLS = {'tools': {'calculate': {'name': 'calculate', 'parameters': {
    'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression']}}}}

MAGENTIC_TOOLS = {'protocol': {'event_types': ['Orchestrator']}, 'agents': {'Orchestrator': {'capability': 'plan'}}}


def trace(steps, domain='tau'):
    return normalize({'trajectory_id': 't', 'steps': [{'index': i, **s} for i, s in enumerate(steps, 1)]}, domain)


def p2_rows(trajectory, tools):
    rows, _ = run_perspective_checks('P2_TOOL_EXECUTION',
                                     seed_constraints_for_perspective('P2_TOOL_EXECUTION', tools),
                                     trajectory, None, {'policy': '', 'tools': tools, 'python_timeout': 3})
    assert not [r for r in rows if r['verdict'] == 'ERROR'], rows
    return rows


def call_verdict(tools, name, arguments):
    """P2's verdict on a single agent step making one tool call."""
    trajectory = trace([{'substeps': [{'role': 'user', 'content': 'go'}]},
                        {'substeps': [{'role': 'assistant', 'content': '',
                                       'tool_calls': [{'id': 'c', 'name': name, 'arguments': arguments}]}]}])
    return next(r['verdict'] for r in p2_rows(trajectory, tools) if r['step_index'] == 2)


def tool_step(name, arguments, result):
    return [{'substeps': [{'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'c', 'name': name, 'arguments': arguments}]}]},
            {'substeps': [{'role': 'tool', 'name': name, 'content': result}]}]


class SchemaArgumentTests(unittest.TestCase):
    def test_catches_the_trace_34_argument_mismatch(self):
        """tau trace 34 step 17, the annotated InvalidInvocation, with no model call involved."""
        self.assertEqual(call_verdict(INPUTS_TOOLS, 'modify_pending_order_items',
                                      {'order_id': '#W1', 'item_ids': ['A'], 'new_item_ids': ['B'],
                                       'payment_method_id': 'paypal_1'}), 'VIOL')
        self.assertEqual(call_verdict(INPUTS_TOOLS, 'modify_pending_order_items',
                                      {'order_id': '#W1', 'item_ids_to_add': ['B'],
                                       'item_ids_to_remove': ['A']}), 'SAT')

    def test_inputs_shape_judges_unknown_keys_but_never_requires_any(self):
        """The release schemas carry no optionality marker, so a subset of inputs cannot be a violation."""
        self.assertEqual(call_verdict(INPUTS_TOOLS, 'cancel_pending_order', {'order_id': '#W1'}), 'SAT')
        self.assertEqual(call_verdict(INPUTS_TOOLS, 'cancel_pending_order',
                                      {'order_id': '#W1', 'bogus': 1}), 'VIOL')

    def test_parameters_shape_also_enforces_required(self):
        self.assertEqual(call_verdict(PARAM_TOOLS, 'calculate', {'expression': '1+1'}), 'SAT')
        self.assertEqual(call_verdict(PARAM_TOOLS, 'calculate', {}), 'VIOL')
        self.assertEqual(call_verdict(PARAM_TOOLS, 'calculate', {'expression': '1+1', 'x': 2}), 'VIOL')

    def test_unknown_tool_name_is_judged_only_when_the_vocabulary_is_known(self):
        self.assertEqual(call_verdict(INPUTS_TOOLS, 'no_such_tool', {}), 'VIOL')
        # The Magentic payload declares agents and a protocol, not tools. Reading its keys as tool names
        # would flag every real call as unknown, so an unrecognised payload judges no names at all.
        self.assertEqual(tool_catalog(MAGENTIC_TOOLS), {})
        self.assertEqual(call_verdict(MAGENTIC_TOOLS, 'no_such_tool', {}), 'SAT')

    def test_non_object_arguments_remain_a_violation(self):
        self.assertEqual(call_verdict(INPUTS_TOOLS, 'cancel_pending_order', '{bad json'), 'VIOL')

    def test_tool_catalog_reads_every_payload_shape_in_circulation(self):
        flat = {'cancel_pending_order': {'inputs': {'order_id': 'string'}}}
        per_agent = {'order_agent': [{'name': 'cancel_pending_order',
                                      'parameters': {'properties': {'order_id': {}}, 'required': ['order_id']}}]}
        openai = [{'function': {'name': 'cancel_pending_order', 'parameters': {'properties': {'order_id': {}}}}}]
        for payload in (INPUTS_TOOLS, flat, per_agent, openai):
            self.assertIn('cancel_pending_order', tool_catalog(payload), payload)
        self.assertEqual(tool_catalog(per_agent)['cancel_pending_order']['required'], ('order_id',))
        # An explicit tools key is authoritative even when the values declare nothing.
        self.assertEqual(tool_catalog({'tools': {'x': {}}})['x'], {'declared': None, 'required': ()})


class StuckAndErrorTests(unittest.TestCase):
    def test_identical_repeat_is_a_violation_but_the_first_result_is_not(self):
        args = {'order_id': '#W1'}
        body = json.dumps({'state': 'in_flight'})
        steps = tool_step('get_order_details', args, body) * 2
        rows = p2_rows(trace(steps), INPUTS_TOOLS)
        viol = sorted(r['step_index'] for r in rows if r['verdict'] == 'VIOL')
        self.assertEqual(viol, [4])          # step 2 is the first result, step 4 repeats it identically

    def test_a_different_result_for_the_same_call_is_not_a_loop(self):
        args = {'order_id': '#W1'}
        steps = tool_step('get_order_details', args, '{"state": "in_flight"}') + \
            tool_step('get_order_details', args, '{"order_id": "#W1"}')[0:2]
        rows = p2_rows(trace(steps), INPUTS_TOOLS)
        self.assertEqual([r['step_index'] for r in rows if r['verdict'] == 'VIOL'], [])

    def test_a_single_error_is_not_evidence_but_a_repeated_failure_is(self):
        args = {'order_id': '#W1'}
        once = p2_rows(trace(tool_step('get_order_details', args, 'Error: user not found')), INPUTS_TOOLS)
        self.assertEqual([r['step_index'] for r in once if r['verdict'] == 'VIOL'], [])
        twice = p2_rows(trace(tool_step('get_order_details', args, 'Error: 503 Service Unavailable') +
                              tool_step('get_order_details', args, 'Error: 504 Gateway Timeout')), INPUTS_TOOLS)
        self.assertEqual([r['step_index'] for r in twice if r['verdict'] == 'VIOL'], [4])

    def test_a_recovered_lookup_error_no_longer_fires(self):
        """tau trace 34 steps 5-7: a lookup errors, another route succeeds. The old seed flagged step 6."""
        steps = tool_step('get_order_details', {'order_id': '#W1'}, 'Error: user not found') + \
            tool_step('get_order_details', {'order_id': '#W2'}, '{"order_id": "#W2"}')
        rows = p2_rows(trace(steps), INPUTS_TOOLS)
        self.assertEqual([r['step_index'] for r in rows if r['verdict'] == 'VIOL'], [])

    def test_the_python_twin_agrees_with_the_sandboxed_clause(self):
        args = {'order_id': '#W1'}
        body = json.dumps({'state': 'in_flight'})
        trajectory = trace(tool_step('get_order_details', args, body) * 3)
        rows = p2_rows(trajectory, INPUTS_TOOLS)
        by_index = {s['index']: s for s in trajectory['steps']}
        for row in rows:
            twin = repeats_earlier_tool_result(by_index, row['step_index'])
            if twin:
                self.assertEqual(row['verdict'], 'VIOL', row['step_index'])
            self.assertEqual(twin, row['step_index'] in (4, 6), row['step_index'])


class EvidenceLibraryAgreementTests(unittest.TestCase):
    """The seed must not be looser than repair_strategies' tested schema validator on argument names.

    Only key-level mutations are compared, so every issue the validator reports is a key issue and the
    comparison needs no knowledge of its value-type rules. One-directional on purpose: the validator
    also checks types, which the seed does not claim to do.
    """

    def test_seed_flags_every_key_issue_the_validator_reports(self):
        from repair_strategies.context import TaskSpec
        from repair_strategies.evidence import schema as ev
        task = TaskSpec.preset('retail')
        tools = {'tools': {name: {'name': name, 'parameters': spec.parameters}
                           for name, spec in task.tools.items()}}
        cases = [('cancel_pending_order', {'order_id': '#W1', 'reason': 'no longer needed'}),
                 ('cancel_pending_order', {'order_id': '#W1'}),
                 ('cancel_pending_order', {'order_id': '#W1', 'reason': 'x', 'extra': 'y'}),
                 ('cancel_pending_order', {}),
                 ('get_order_details', {'order_id': '#W1'}),
                 ('get_order_details', {'order': '#W1'})]
        compared = 0
        for name, arguments in cases:
            issues = ev.validate(arguments, task.tools[name].parameters, strict_properties=True)
            verdict = call_verdict(tools, name, arguments)
            compared += 1
            if issues:
                self.assertEqual(verdict, 'VIOL', (name, arguments, [i.describe() for i in issues]))
        self.assertEqual(compared, len(cases))


if __name__ == '__main__':
    unittest.main()
