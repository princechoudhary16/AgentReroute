import json
import tempfile
import unittest
from pathlib import Path
from agentrx.data import normalize
from agentrx.pipeline import diagnose
from agentrx.providers import CachedProvider, ReplayProvider
from tests.test_constraints import python_constraint


class TraceProvider:
    identity = {'provider': 'fixture', 'model': 'test'}
    measurement_kind = 'offline_fixture'
    def __init__(self, invalid=False): self.requests = []; self.invalid = invalid
    def complete(self, stage, payload):
        self.requests.append((stage, payload))
        if stage == 'global': return {'constraints': []}
        if stage == 'dynamic':
            k = payload['trajectory']['steps'][-1]['index']
            return {'constraints': [python_constraint(['return False'], {'step_index': k})]}
        if stage == 'localize': return {'step_index': 9, 'agent': 'assistant', 'rationale': 'step 4 recovered; step 9 unrecovered'}
        if stage in ('judge', 'categorize'):
            return {'step_index': 999 if self.invalid else 9, 'agent': 'assistant',
                    'category': 'InvalidInvocation', 'rationale': 'step 4 recovered; step 9 unrecovered',
                    'evidence_refs': [{'step_index': 9, 'substep_index': 1}]}
        raise AssertionError(stage)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.ir = normalize({'trajectory_id': 'x', 'steps': [
            {'index': 4, 'substeps': [{'role': 'assistant', 'content': 'recovered error'}]},
            {'index': 9, 'substeps': [{'role': 'assistant', 'content': 'fatal error'}]}]}, 'tau')

    def test_stepwise_prefix_and_judge_can_skip_recovered_first_violation(self):
        provider = TraceProvider()
        with tempfile.TemporaryDirectory() as d:
            result = diagnose(self.ir, '', {}, {'evidence': True}, provider, Path(d))
            self.assertEqual(result['prediction']['step_index'], 9)
            self.assertEqual([v['step_index'] for v in result['violations']], [4, 9])
            prefixes = [p['trajectory']['steps'] for s,p in provider.requests if s == 'dynamic']
            self.assertEqual([len(p) for p in prefixes], [1,2])
            self.assertNotIn('fatal', json.dumps(prefixes[0]))
            self.assertTrue((Path(d)/'validation_log.json').exists())

    def test_baseline_does_not_generate_evidence_and_two_stage_fixes_step(self):
        provider = TraceProvider()
        with tempfile.TemporaryDirectory() as d:
            result = diagnose(self.ir, '', {}, {'evidence': False, 'protocol': 'step_then_category'}, provider, d)
            self.assertEqual([s for s,p in provider.requests], ['localize', 'categorize'])
            self.assertEqual(result['prediction']['step_index'], 9)
            self.assertEqual(provider.requests[-1][1]['selected_step']['step_index'], 9)

    def test_invalid_prediction_is_error_not_scored(self):
        with tempfile.TemporaryDirectory() as d:
            result = diagnose(self.ir, '', {}, {'evidence': False}, TraceProvider(True), d)
            self.assertEqual(result['status'], 'error')
            self.assertIsNone(result['prediction'])

    def test_oneshot_only_one_dynamic_call_and_checks_each_step(self):
        provider = TraceProvider()
        with tempfile.TemporaryDirectory() as d:
            result = diagnose(self.ir, '', {}, {'evidence': True, 'generation': 'oneshot'}, provider, d)
            self.assertEqual(sum(s == 'dynamic' for s,p in provider.requests), 1)
            self.assertEqual([c['verdict'] for c in result['checks']], ['SKIP','VIOL'])

    def test_cached_calls_isolate_repetition_and_prompt_settings(self):
        provider = TraceProvider()
        payload = {'trajectory': self.ir}
        with tempfile.TemporaryDirectory() as d:
            CachedProvider(provider, d, {'rep': 0}).complete('dynamic', payload)
            CachedProvider(provider, d, {'rep': 0}).complete('dynamic', payload)
            self.assertEqual(len(provider.requests), 1)
            CachedProvider(provider, d, {'rep': 1}).complete('dynamic', payload)
            CachedProvider(provider, d, {'rep': 1, 'settings': 'changed'}).complete('dynamic', payload)
            self.assertEqual(len(provider.requests), 3)

    def test_fixture_replay_marks_outputs_as_non_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'replay.json'; path.write_text('{"global": {"constraints": []}}')
            provider = ReplayProvider(path)
            self.assertEqual(provider.measurement_kind, 'offline_fixture')
            self.assertEqual(provider.complete('global', {}), {'constraints': []})

    def test_two_stage_can_abstain_without_made_up_localization(self):
        class Abstaining(TraceProvider):
            def complete(self, stage, payload):
                return {'step_index': None, 'agent': None, 'category': 'Inconclusive',
                        'custom_category': 'Insufficient trace', 'rationale': 'No evidence of decisive failure'}
        with tempfile.TemporaryDirectory() as d:
            result = diagnose(self.ir, '', {}, {'evidence': False, 'protocol': 'step_then_category'}, Abstaining(), d)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['prediction']['category'], 'Inconclusive')

    def test_responsible_agent_can_differ_from_selected_event_speaker(self):
        from agentrx.pipeline import validate_prediction
        ir=normalize({'trajectory_id':'handoff','steps':[
            {'index':1,'substeps':[{'role':'Assistant','content':'proposal'}]},
            {'index':2,'substeps':[{'role':'user','content':'confirmation'}]}]},'tau')
        p=validate_prediction({'step_index':2,'agent':'Assistant','category':'InvalidInvocation',
                               'rationale':'assistant responsible for earlier proposal'},ir)
        self.assertEqual(p['agent'],'Assistant')
        with self.assertRaises(ValueError):
            validate_prediction({'step_index':2,'agent':'Nonexistent','category':'InvalidInvocation',
                                 'rationale':'invented agent'},ir)

    def test_localization_can_abstain_while_deferring_category(self):
        class NoCategory(TraceProvider):
            def complete(self,stage,payload):
                if stage!='localize': raise AssertionError('must not categorize an unlocalized step')
                return {'step_index':None,'agent':None,'custom_category':'Insufficient trace',
                        'rationale':'no decisive event can be grounded'}
        with tempfile.TemporaryDirectory() as d:
            result=diagnose(self.ir,'',{}, {'evidence':False,'protocol':'step_then_category'},NoCategory(),d)
            self.assertEqual(result['status'],'ok')
            self.assertEqual(result['prediction']['category'],'Inconclusive')
