import unittest
from agentrx.constraints import validate_constraint, guard_matches, check_constraint
from agentrx.data import normalize


def python_constraint(body, trigger=None):
    return {'assertion_name': 'check', 'taxonomy_targets': ['InvalidInvocation'],
            'constraint_type': 'SCHEMA', 'event_trigger': trigger or {}, 'check_hint': 'check args',
            'check_type': 'python_check', 'python_check': {
                'function_name': 'check', 'code_lines': ['def check(trajectory, current_step_index):'] + ['    '+line for line in body]}}


class SemanticProvider:
    def __init__(self, result): self.result = result
    def complete(self, stage, payload): return self.result


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.ir = normalize({'trajectory_id': 'x', 'steps': [
            {'index': 4, 'substeps': [{'role': 'assistant', 'content': 'hello'}]},
            {'index': 9, 'substeps': [{'role': 'assistant', 'content': 'bad call', 'tool_calls': [
                {'function': {'name': 'modify_order', 'arguments': {'id': 'x'}}}]}]},
            {'index': 11, 'substeps': [{'role': 'assistant', 'content': 'FUTURE_SECRET'}]}]}, 'tau')

    def test_guard_matches_same_event_and_sparse_step(self):
        c = validate_constraint(python_constraint(['return True'], {'step_index': '8-10',
            'role_name': 'assistant', 'tool_name': 'modify_.*', 'content_regex': 'bad'}))
        self.assertFalse(guard_matches(c, self.ir['steps'][0]))
        self.assertTrue(guard_matches(c, self.ir['steps'][1]))
        step = {'index': 9, 'substeps': [{'sub_index': 1, 'role': 'assistant', 'content': ''},
                {'sub_index': 2, 'role': 'tool', 'name': 'modify_order', 'content': 'bad'}]}
        self.assertFalse(guard_matches(c, step))

    def test_checker_gets_prefix_and_zero_based_offset(self):
        c = python_constraint(['return current_step_index == 1 and len(trajectory["steps"]) == 2'])
        result = check_constraint(c, self.ir, 1, None, {})
        self.assertEqual(result['verdict'], 'SAT')
        self.assertEqual(result['step_index'], 9)
        self.assertNotIn('FUTURE_SECRET', str(result))

    def test_false_is_violation_but_exception_and_nonbool_are_error(self):
        for body, expected in [(['return False'], 'VIOL'), (['raise ValueError("bad")'], 'ERROR'),
                               (['return "False"'], 'ERROR')]:
            with self.subTest(body=body):
                self.assertEqual(check_constraint(python_constraint(body), self.ir, 0, None, {})['verdict'], expected)

    def test_timeout_and_unsafe_code_are_errors(self):
        c = python_constraint(['while True: pass'])
        self.assertEqual(check_constraint(c, self.ir, 0, None, {'python_timeout': .15})['verdict'], 'ERROR')
        c = python_constraint(['import os', 'return bool(os.environ)'])
        self.assertEqual(check_constraint(c, self.ir, 0, None, {})['verdict'], 'ERROR')

    def test_invalid_regex_and_unknown_category_rejected(self):
        for changes in [{'event_trigger': {'content_regex': '['}}, {'taxonomy_targets': ['fake']}]:
            c = python_constraint(['return True']); c.update(changes)
            with self.assertRaises(ValueError): validate_constraint(c)

    def test_semantic_unclear_does_not_create_violation(self):
        c = {'assertion_name': 'meaning', 'taxonomy_targets': ['SystemFailure'], 'invariant_type': 'ANY',
             'trigger': {'step_index': 9}, 'check_hint': 'x', 'check_type': 'nl_check',
             'nl_check': {'judge_rubric': ['criterion A', 'criterion B']}}
        result = check_constraint(c, self.ir, 1, SemanticProvider({'rubric_results': [
            {'criterion_index': 0, 'evaluation': 'UNCLEAR', 'reasoning': 'unknown'},
            {'criterion_index': 1, 'evaluation': 'CLEAR_PASS', 'reasoning': 'clear'}]}), {})
        self.assertEqual(result['verdict'], 'SAT')
        self.assertEqual(result['unclear_count'], 1)
        result = check_constraint(c, self.ir, 1, SemanticProvider({'rubric_results': []}), {})
        self.assertEqual(result['verdict'], 'ERROR')

    def test_semantic_clear_failure_needs_complete_rubric(self):
        c = {'assertion_name': 'meaning', 'taxonomy_targets': ['SystemFailure'], 'constraint_type': 'ANY',
             'check_hint': 'x', 'check_type': 'nl_check', 'nl_check': {'judge_rubric': ['A']}}
        provider = SemanticProvider({'rubric_results': [{'criterion_index': 0,
                       'evaluation': 'CLEAR_FAIL', 'reasoning': 'explicit timeout at step 4'}]})
        result = check_constraint(c, self.ir, 0, provider, {})
        self.assertEqual(result['verdict'], 'VIOL')
        self.assertIn('explicit timeout', str(result['rubric_results']))

    def test_skip_never_runs_python(self):
        c = python_constraint(['raise ValueError("must not run")'], {'step_index': 99})
        self.assertEqual(check_constraint(c, self.ir, 0, None, {})['verdict'], 'SKIP')

    def test_module_and_dynamic_attribute_paths_cannot_read_files(self):
        attacks = [
            ['import statistics', 'return statistics.sys.modules["builtins"].open("README.md").read().startswith("# AgentRx")'],
            ['import statistics', 'import operator',
             'system = operator.attrgetter("sys")(statistics)',
             'return system.modules["builtins"].open("README.md").read().startswith("# AgentRx")'],
            ['g = (g.gi_frame.f_back.f_back.f_globals for _ in [0])',
             'real = next(g)["__builtins__"]',
             'return real.open("README.md").read().startswith("# AgentRx")']]
        for attack in attacks:
            with self.subTest(attack=attack):
                result=check_constraint(python_constraint(attack), self.ir, 0, None, {})
                self.assertEqual(result['verdict'],'ERROR')

    def test_malformed_selectors_are_errors_not_skips_or_violations(self):
        for selector in [['1'],[True],[-1],[1.0],True,'9-4','bad']:
            with self.subTest(selector=selector):
                c=python_constraint(['return False'],{'step_index':selector})
                with self.assertRaises(ValueError): validate_constraint(c)
                self.assertEqual(check_constraint(c,self.ir,0,None,{})['verdict'],'ERROR')

    def test_constrained_library_exports_still_support_normal_checks(self):
        c=python_constraint(['import statistics','import math','from collections import Counter',
                            'return statistics.mean([1,2,3]) == 2 and math.isclose(2,2) and Counter("aa")["a"] == 2'])
        self.assertEqual(check_constraint(c,self.ir,0,None,{})['verdict'],'SAT')
