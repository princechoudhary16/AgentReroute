"""Offline integration tests against tiny normalized published-format traces."""
import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agentrx.data import normalize
from agentrx.io import write_json
from agentrx.providers import ProviderError
from agentrx.rl_policy import DualLoopRLJudge, THETA0
from demo.comparison import ComparisonService, MeasuredProvider, PublishedInput, run_system


def fixture_record():
    trace = normalize({'trajectory_id': 'fixture-1', 'instruction': 'Look up this order.', 'steps': [
        {'index': 1, 'substeps': [{'role': 'user', 'content': 'Please look up my order.'}]},
        {'index': 2, 'substeps': [{'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call1', 'function': {'name': 'unknown_tool', 'arguments': {}}}]}]},
        {'index': 3, 'substeps': [{'role': 'tool', 'name': 'unknown_tool', 'tool_call_id': 'call1', 'content': 'Unknown tool'}]},
        {'index': 4, 'substeps': [{'role': 'assistant', 'content': 'I cannot complete the lookup.'}]},
    ]}, 'tau')
    critical = {'failure_id': 'annotation-only-sentinel', 'step_index': 4,
                'agent': 'assistant', 'category': 'InvalidInvocation'}
    return {'trajectory': trace, 'gold': {'trajectory_id': 'fixture-1', 'critical': critical, 'failures': [critical]},
            'policy': 'Use the listed tools.', 'tools': {'get_order_details': {'inputs': {}}}}


def write_fixture_corpus(path):
    write_json(Path(path)/'cases.json', [fixture_record()])
    return Path(path)


class TransportFixture:
    """Only the external model is replaced; both diagnosis pipelines execute normally."""
    measurement_kind = 'model_measurement'
    identity = {'provider': 'test-transport', 'model': 'fixture-model',
                'settings': {'temperature': 0, 'max_tokens': 16000}}
    def __init__(self, fail=False):
        self.fail, self.payloads, self.last_telemetry = fail, [], {}
    def complete(self, stage, payload):
        self.payloads.append((stage, copy.deepcopy(payload)))
        if self.fail:
            raise ProviderError('Fixture transport unavailable')
        self.last_telemetry = {'usage': {'prompt_tokens': 10, 'completion_tokens': 4}, 'model': 'fixture-model'}
        if stage in ('global', 'dynamic', 'perspective_global', 'perspective_dynamic'):
            return {'constraints': []}
        if stage in ('judge', 'mp_rl_select'):
            return {'step_index': 4, 'agent': 'assistant', 'category': 'InvalidInvocation',
                    'rationale': 'The final response abandoned the lookup.',
                    'evidence_refs': [{'step_index': 4, 'substep_index': 1}]}
        raise AssertionError(stage)
    def complete_messages(self, messages):
        raise AssertionError('Benchmark attribution must not replay or repair.')


def wait_done(service, job_id):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = service.snapshot(job_id)
        if state['status'] in ('completed', 'partial', 'error', 'canceled'):
            return state
        time.sleep(.01)
    raise AssertionError('comparison did not terminate')


class ComparisonExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus = write_fixture_corpus(self.root/'corpus')

    def service(self, factory=None, **kwargs):
        service = ComparisonService(self.root/'runs', corpus=self.corpus,
            provider_factory=factory or (lambda _: TransportFixture()), model_catalog=lambda: ['fixture-model'], **kwargs)
        self.addCleanup(service.close)
        return service

    def start(self, service, variant='seeds_select'):
        return wait_done(service, service.start('tau:fixture-1', 'fixture-model', variant)['id'])

    def test_shared_blind_input_selected_out_of_pool_and_post_decision_scoring(self):
        transports = []
        def factory(model):
            provider = TransportFixture(); transports.append(provider); return provider
        service = self.service(factory)
        self.assertEqual(service.catalog()[0]['id'], 'tau:fixture-1')
        with patch.object(DualLoopRLJudge, 'save', side_effect=AssertionError('No theta writes')), \
             patch.object(DualLoopRLJudge, 'record_recovery_outcome', side_effect=AssertionError('No reward updates')):
            state = self.start(service)
        self.assertEqual(state['status'], 'completed', state['results'])
        self.assertEqual(len(transports), 1)  # pacing budget shared across systems
        a, b = [state['results'][s] for s in ('agentrx', 'agentreroute')]
        self.assertEqual(a['input_hash'], b['input_hash'])
        self.assertEqual(a['provider']['settings'], b['provider']['settings'])
        self.assertEqual(b['prediction']['step_index'], 4)
        self.assertNotIn(4, [c['step_index'] for c in b['ranked_candidates']])
        self.assertFalse(b['selection']['in_pool'])
        self.assertTrue(b['score']['step_correct'])
        self.assertEqual(b['synthesis']['state'], 'fixed_checks')
        stages = [stage for stage, payload in transports[0].payloads]
        self.assertEqual(stages, ['global', 'dynamic', 'judge', 'mp_rl_select'])
        for stage, payload in transports[0].payloads:
            self.assertNotIn('"gold"', json.dumps(payload))
            self.assertNotIn('annotation-only-sentinel', json.dumps(payload))
        events = state['events']
        ready = next(e for e in events if e['kind'] == 'input_ready')
        self.assertNotIn('"gold"', json.dumps(ready))
        scored = [e['seq'] for e in events if e['kind'] == 'attribution_scored']
        decisions = [e['seq'] for e in events if e['kind'] == 'attribution_completed']
        self.assertGreater(min(scored), max(decisions))
        self.assertFalse(any(e['kind'] in ('repair_started', 'verification_completed') for e in events))
        self.assertFalse(list((self.root/'runs').rglob('policy_after.json')))
        blame = json.loads(next((self.root/'runs').rglob('blame_matrix.json')).read_text())
        self.assertEqual(blame['weights'], list(THETA0))
        self.assertEqual(len(service.snapshot(state['id'], events[-2]['seq'])['events']), 1)

    def test_allowlisted_input_discards_annotation_fields(self):
        row = fixture_record()
        row['trajectory']['gold'] = {'step': 4}
        row['trajectory']['steps'][0]['fault'] = 'secret annotation'
        row['trajectory']['steps'][0]['substeps'][0]['injected_fault'] = True
        case = PublishedInput(row)
        self.assertNotIn('gold', case.__dict__)
        self.assertNotIn('fault', json.dumps(case.trace))

    def test_selector_fallback_and_failed_arm_mark_job_partial(self):
        class FailedSelector(TransportFixture):
            def complete(self, stage, payload):
                if stage == 'mp_rl_select':
                    raise ProviderError('Fixture selector unavailable')
                return super().complete(stage, payload)
        state = self.start(self.service(lambda _: FailedSelector()))
        self.assertEqual(state['status'], 'partial')
        self.assertEqual(state['results']['agentrx']['status'], 'completed')
        reroute = state['results']['agentreroute']
        self.assertEqual(reroute['status'], 'partial')
        self.assertTrue(reroute['selection']['fell_back'])
        self.assertTrue(reroute['degradation_reasons'])
        self.assertTrue(any(e['kind'] == 'request_failed' for e in state['events']))
        class FailedBaseline(TransportFixture):
            def complete(self, stage, payload):
                if stage == 'judge':
                    raise ProviderError('Fixture baseline unavailable')
                return super().complete(stage, payload)
        state = self.start(self.service(lambda _: FailedBaseline()))
        self.assertEqual(state['status'], 'partial')
        self.assertEqual(state['results']['agentrx']['status'], 'error')
        self.assertEqual(state['results']['agentreroute']['status'], 'completed')

    def test_invalid_and_empty_written_synthesis_are_distinct(self):
        class Invalid(TransportFixture):
            def complete(self, stage, payload):
                answer = super().complete(stage, payload)
                return {'constraints': [{'invalid': 'constraint'}]} if stage.startswith('perspective_') else answer
        for invalid in (False, True):
            with self.subTest(invalid=invalid):
                state = self.start(self.service(lambda _: Invalid() if invalid else TransportFixture()), 'synth_select_p10')
                result = state['results']['agentreroute']
                self.assertEqual(result['status'], 'partial' if invalid else 'completed')
                self.assertEqual(state['status'], 'partial' if invalid else 'completed')
                self.assertEqual(result['synthesis']['state'], 'invalid_synthesis_fallback' if invalid else 'valid_empty_synthesis')
                self.assertEqual(result['config']['selector_pool'], 10)

    def test_failed_written_check_is_partial_even_with_successful_selector(self):
        from agentrx.perspectives import seed_constraints_for_perspective
        # A valid generated check whose execution returns ERROR must not count as a clean arm.
        check = copy.deepcopy(seed_constraints_for_perspective('P1_INTENT_PLAN', {})[0])
        check['assertion_name'] = 'model_check'
        check['python_check']['function_name'] = 'model_check'
        check['python_check']['code_lines'] = ['def model_check(trajectory, current_step_index):', '    if current_step_index >= 2:', '        return 1 / 0', '    return True']
        class BadCheck(TransportFixture):
            def complete(self, stage, payload):
                answer = super().complete(stage, payload)
                return {'constraints': [check]} if stage == 'perspective_global' and payload['perspective'] == 'P1_INTENT_PLAN' else answer
        state = self.start(self.service(lambda _: BadCheck()), 'synth_select_p10')
        result = state['results']['agentreroute']
        self.assertEqual(result['status'], 'partial')
        self.assertGreater(result['checks']['ERROR'], 0)
        self.assertFalse(result['selection']['fell_back'])

    def test_missing_corpus_is_actionable_and_never_substitutes_scenarios(self):
        service = ComparisonService(self.root/'none', corpus=self.root/'missing',
            provider_factory=lambda _: self.fail('No model calls'), model_catalog=lambda: ['fixture-model'])
        self.addCleanup(service.close)
        self.assertEqual(service.catalog(), [])
        self.assertIn('data/benchmark/cases.json', service.catalog_error)
        with self.assertRaisesRegex(ValueError, 'Prepare'):
            service.start('handoff_drop_context', 'fixture-model')
        self.assertEqual(service.jobs, {})

    def test_cancellation_and_duplicate_guard(self):
        entered, released = threading.Event(), threading.Event()
        class Slow(TransportFixture):
            def complete(self, stage, payload):
                entered.set(); released.wait(5)
                return super().complete(stage, payload)
        service = self.service(lambda _: Slow())
        job = service.start('tau:fixture-1', 'fixture-model')
        self.assertTrue(entered.wait(3))
        try:
            with self.assertRaises(RuntimeError):
                service.start('tau:fixture-1', 'fixture-model')
            service.cancel(job['id'])
        finally:
            released.set()
        state = wait_done(service, job['id'])
        self.assertEqual(state['status'], 'canceled')
        self.assertNotIn('agentreroute', state['results'])
        self.assertEqual(sum(e['kind'] == 'request_started' for e in state['events']), 1)

    def test_saved_new_and_legacy_runs_keep_original_metadata_without_inference(self):
        service = self.service()
        original = self.start(service)
        restored = self.service(lambda _: self.fail('No replay inference'))
        saved = restored.snapshot(original['id'])
        self.assertEqual(saved['results'], original['results'])
        self.assertEqual(saved['events'], original['events'])
        self.assertFalse(saved['legacy'])
        jid = 'a'*32
        write_json(self.root/'runs'/jid/'comparison.json', {'id': jid, 'scenario': 'old-scenario',
            'model': 'old-model', 'status': 'completed', 'events': [{'kind': 'repair_started'}],
            'results': {'agentrx': {'provider': {'provider': 'ollama', 'model': 'old-model'}}}})
        saved = restored.snapshot(jid)
        self.assertTrue(saved['legacy'])
        self.assertEqual(saved['mode'], 'legacy_recovery')
        self.assertEqual(saved['provider'], 'ollama')
        self.assertEqual(saved['events'], [{'kind': 'repair_started'}])
        with self.assertRaises(KeyError):
            restored.snapshot('../escape')

    def test_finished_state_waits_for_save_and_partial_jobs_can_be_restarted(self):
        real_write, entered, released = write_json, threading.Event(), threading.Event()
        def slow_write(path, value):
            if Path(path).name == 'comparison.json':
                entered.set(); released.wait(5)
            real_write(path, value)
        service = self.service()
        with patch('demo.comparison.write_json', side_effect=slow_write):
            job = service.start('tau:fixture-1', 'fixture-model')
            self.assertTrue(entered.wait(5))
            try:
                self.assertEqual(service.jobs[job['id']]['status'], 'running')
            finally:
                released.set()
            self.assertEqual(wait_done(service, job['id'])['status'], 'completed')
        service = self.service(lambda _: TransportFixture(fail=True))
        self.assertEqual(self.start(service)['status'], 'partial')
        self.assertEqual(self.start(service)['status'], 'partial')

    def test_unknown_inputs_cursors_and_budget_are_rejected(self):
        service = self.service()
        for sid, model, variant in [('../escape', 'fixture-model', 'seeds_select'),
                ('tau:fixture-1', 'missing', 'seeds_select'), ('tau:fixture-1', 'fixture-model', 'trained')]:
            with self.assertRaises(ValueError):
                service.start(sid, model, variant)
        state = self.start(service)
        for cursor in (-1, True, '0'):
            with self.assertRaises(ValueError):
                service.snapshot(state['id'], cursor)
        provider = MeasuredProvider(TransportFixture(), lambda _: None, self.root/'calls', threading.Event(), max_calls=1)
        provider.complete('global', {})
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            provider.complete('dynamic', {})
        self.assertEqual(provider.metrics()['model_calls'], 1)


class ComparisonHTTPTests(unittest.TestCase):
    def setUp(self):
        from demo.visualizer import make_server
        self.temp=tempfile.TemporaryDirectory()
        self.service=ComparisonService(Path(self.temp.name)/'runs',corpus=write_fixture_corpus(Path(self.temp.name)/'corpus'),provider_factory=lambda _:TransportFixture(),model_catalog=lambda:['fixture-model'])
        self.server=make_server(port=0,delay=0,comparison_service=self.service)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_address[1]}'

    def tearDown(self):
        if hasattr(self,'server'):
            self.server.shutdown();self.server.server_close();self.thread.join()
        self.service.close();self.temp.cleanup()

    def fetch(self,path,data=None,origin=None,raw=None):
        from urllib.request import Request,urlopen
        headers={'Content-Type':'application/json'} if data is not None or raw is not None else {}
        if origin:headers['Origin']=origin
        body=raw if raw is not None else json.dumps(data).encode() if data is not None else None
        req=Request(self.base+path,data=body,headers=headers)
        with urlopen(req,timeout=10) as response:return response.status,response.read().decode()

    def test_navigation_models_and_full_comparison_round_trip(self):
        status,body=self.fetch('/compare')
        self.assertEqual(status,200)
        self.assertIn('id="comparison-app"',body)
        self.assertIn('href="/compare"',self.fetch('/')[1])
        self.assertEqual(json.loads(self.fetch('/api/models')[1])['models'],['fixture-model'])
        status,body=self.fetch('/api/comparisons',{'scenario':'tau:fixture-1','model':'fixture-model'},origin=self.base)
        self.assertEqual(status,202)
        jid=json.loads(body)['id'];wait_done(self.service,jid)
        state=json.loads(self.fetch('/api/comparisons/'+jid+'?after=0')[1])
        self.assertEqual(set(state['results']),{'agentrx','agentreroute'})
        self.assertEqual(state['status'],'completed')
        status,body=self.fetch('/api/comparisons/'+jid+'/cancel',{})
        self.assertEqual(status,200)

    def test_cross_origin_invalid_bodies_and_cursors_are_rejected(self):
        from urllib.error import HTTPError
        payload={'scenario':'tau:fixture-1','model':'fixture-model'}
        with self.assertRaises(HTTPError) as raised:self.fetch('/api/comparisons',payload,origin='https://untrusted.example')
        self.assertEqual(raised.exception.code,403);raised.exception.close()
        for value in ([],{},dict(payload,endpoint='http://elsewhere'),dict(payload,scenario='../escape'),dict(payload,model='missing')):
            with self.subTest(value=value),self.assertRaises(HTTPError) as raised:self.fetch('/api/comparisons',value)
            self.assertEqual(raised.exception.code,400);raised.exception.close()
        with self.assertRaises(HTTPError) as raised:self.fetch('/api/comparisons',raw=b'{broken')
        self.assertEqual(raised.exception.code,400);raised.exception.close()
        job=self.service.start(**payload);wait_done(self.service,job['id'])
        for cursor in ('-1','abc','1.5'):
            with self.assertRaises(HTTPError) as raised:self.fetch('/api/comparisons/'+job['id']+'?after='+cursor)
            self.assertEqual(raised.exception.code,400);raised.exception.close()
        with self.assertRaises(HTTPError) as raised:self.fetch('/api/comparisons/unknown')
        self.assertEqual(raised.exception.code,404);raised.exception.close()



class MeasurementTests(unittest.TestCase):
    def test_failed_call_usage_is_included_or_marked_unknown_without_stale_telemetry(self):
        class FailSecond(TransportFixture):
            def __init__(self,known):super().__init__();self.known=known;self.calls=0
            def complete(self,stage,payload):
                self.calls+=1
                if self.calls==1:return super().complete(stage,payload)
                if self.known:self.last_telemetry={'usage':{'prompt_tokens':30,'completion_tokens':20}}
                raise RuntimeError('Incomplete or timed-out response')
        with tempfile.TemporaryDirectory() as d:
            for known in (True,False):
                events=[]
                p=MeasuredProvider(FailSecond(known),events.append,Path(d)/str(known),threading.Event())
                p.complete('global',{})
                with self.assertRaises(RuntimeError):p.complete('dynamic',{})
                stats=p.metrics()
                self.assertEqual(stats['model_calls'],2)
                self.assertEqual(stats['prompt_tokens'],40 if known else None)
                self.assertEqual(stats['completion_tokens'],24 if known else None)
                error=next(e for e in events if e['kind']=='request_failed')
                self.assertEqual(error['telemetry'].get('usage',{}).get('prompt_tokens'),30 if known else None)
