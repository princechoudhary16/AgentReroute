import json
import tempfile
import unittest
from pathlib import Path
from agentrx.data import normalize, normalize_gold, validate_case, prepare_corpus
from agentrx.taxonomy import category


class DataTests(unittest.TestCase):
    def test_retail_strips_gold_but_preserves_instruction_and_raw_positions(self):
        raw = {'task_id': 7, 'reward': 0, 'info': {'task': {
            'instruction': 'Return my item', 'actions': ['SECRET_ACTION'], 'outputs': ['SECRET_ANSWER']}},
            'traj': [{'role': 'system', 'content': 'policy'}, {'role': 'user', 'content': 'return'},
                     {'role': 'assistant', 'content': None, 'tool_calls': [
                         {'id': 'c1', 'function': {'name': 'return_item', 'arguments': '{"id": "1"}'}}]}]}
        ir = normalize(raw, 'tau')
        self.assertEqual(ir['trajectory_id'], '7')
        self.assertEqual(ir['instruction'], 'Return my item')
        self.assertEqual([s['index'] for s in ir['steps']], [1, 2, 3])
        self.assertEqual(ir['steps'][2]['substeps'][0]['tool_calls'][0]['name'], 'return_item')
        self.assertNotIn('SECRET', json.dumps(ir))
        self.assertNotIn('reward', ir)

    def test_explicit_sparse_ir_indices_are_preserved(self):
        ir = normalize({'trajectory_id': 'a', 'instruction': 'x', 'steps': [
            {'index': 4, 'substeps': [{'sub_index': 2, 'role': 'Agent', 'content': 'x'}]},
            {'index': 9, 'substeps': [{'sub_index': 1, 'role': 'Agent', 'content': 'y'}]}]}, 'flash')
        self.assertEqual([s['index'] for s in ir['steps']], [4, 9])
        self.assertEqual(ir['index_map'][1], {'offset': 1, 'step_index': 9, 'raw_position': 1})

    def test_root_is_joined_by_id_and_categories_normalized(self):
        gold = normalize_gold({'trajectory_id': 1, 'failures': [
            {'failure_id': 9, 'step_number': 1, 'failure_category': 'System Failure', 'failed_agent': 'Assistant'},
            {'failure_id': '2', 'step_number': 3, 'failure_category': 'Invention of new information', 'failed_agent': 'FileSurfer'}],
            'root_cause': {'failure_id': 2}})
        self.assertEqual(gold['critical']['step_index'], 3)
        self.assertEqual(gold['critical']['category'], 'InventionOfNewInformation')
        self.assertEqual(category('Instruction Adherence Failure'), 'Instruction/PlanAdherenceFailure')
        with self.assertRaises(ValueError):
            category('Unknown category')

    def test_gold_step_outside_trace_is_rejected(self):
        case = {'trajectory': normalize([{'role': 'human', 'content': 'x'}], 'magentic', 'a'),
                'gold': normalize_gold({'trajectory_id': 'a', 'failures': [
                    {'failure_id': 1, 'step_number': 8, 'failure_category': 'System Failure', 'failed_agent': 'A'}],
                    'root_cause': {'failure_id': 1}})}
        with self.assertRaisesRegex(ValueError, 'step'):
            validate_case(case)

    def test_duplicate_steps_and_unknown_root_fail(self):
        with self.assertRaises(ValueError):
            normalize({'trajectory_id': 'a', 'steps': [{'index': 2, 'substeps': []}, {'index': 2, 'substeps': []}]}, 'flash')
        with self.assertRaises(ValueError):
            normalize_gold({'trajectory_id': 'a', 'failures': [], 'root_cause': {'failure_id': 8}})

    def test_prepare_imports_only_matching_gold_and_marks_missing_domains(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'source/data/ground_truth').mkdir(parents=True)
            (root/'source/data/tau_retail').mkdir()
            (root/'source/data/magentic_dataset').mkdir()
            (root/'source/data/ground_truth/tau_ground_truth.json').write_text(json.dumps([
                {'trajectory_id': 1, 'failures': [{'failure_id': 1, 'step_number': 1,
                  'failure_category': 'System Failure', 'failed_agent': 'assistant'}], 'root_cause': {'failure_id': 1}}]))
            (root/'source/data/tau_retail/tau_dataset_failed.json').write_text(json.dumps([
                {'task_id': 1, 'traj': [{'role': 'assistant', 'content': 'failed'}]}]))
            manifest = prepare_corpus(root/'source', root/'out')
            self.assertEqual(manifest['domains']['tau']['available'], 1)
            self.assertEqual(manifest['domains']['flash']['status'], 'blocked_missing_data')
            self.assertTrue((root/'out/cases.json').is_file())
            self.assertTrue(manifest['corpus_hash'])

    def test_prepare_never_uses_tool_examples_as_schemas(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'data/tool_schemas').mkdir(parents=True)
            (root/'data/tool_schemas/tools_structure.json').write_text('[{"role":"tool","content":"UNRELATED_EXAMPLE"}]')
            (root/'data/ground_truth').mkdir()
            (root/'data/tau_retail').mkdir()
            (root/'data/ground_truth/tau_ground_truth.json').write_text(json.dumps([{'trajectory_id':1,'failures':[{'failure_id':1,'step_number':1,'failure_category':'System Failure'}],'root_cause':{'failure_id':1}}]))
            (root/'data/tau_retail/tau_dataset_failed.json').write_text('[{"task_id":1,"traj":[{"role":"assistant","content":"timeout"}]}]')
            prepare_corpus(root,root/'out')
            manifest=json.loads((root/'out/manifest.json').read_text())
            self.assertEqual(manifest['tool_schema_source'],'packaged_reference_domain_registry')
            imported=json.loads((root/'out/cases.json').read_text())
            self.assertIsInstance(imported[0]['tools'],dict)
            self.assertIn('tools',imported[0]['tools'])
            self.assertNotIn('UNRELATED_EXAMPLE', (root/'out/cases.json').read_text())
