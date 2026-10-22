import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from demo.comparison import ComparisonService, MeasuredProvider, ComparisonCancelled


class TransportFixture:
    """Only replaces the external model; both diagnosis pipelines and retail replay remain real."""
    measurement_kind = 'model_measurement'
    identity = {'provider': 'test-transport', 'model': 'fixture-model'}
    def __init__(self, fail=False):
        self.fail = fail
        self.payloads = []
        self.last_telemetry = {}
    def prepare_for_run(self): pass
    def complete(self, stage, payload):
        self.payloads.append((stage, copy.deepcopy(payload)))
        if self.fail: raise RuntimeError('Ollama transport unavailable')
        self.last_telemetry = {'usage': {'prompt_tokens': 10, 'completion_tokens': 4}, 'model': 'fixture-model'}
        if stage in ('global','dynamic','perspective_global','perspective_dynamic'): return {'constraints': []}
        if stage == 'judge':
            return {'step_index':6,'agent':'planner','category':'MisinterpretationOfToolOutput',
                    'rationale':'The handoff omitted the order identifier; the specialist did not recover.',
                    'evidence_refs':[{'step_index':6,'substep_index':1}]}
        raise AssertionError(stage)
    def complete_messages(self, messages):
        self.last_telemetry = {'usage': {'prompt_tokens': 6, 'completion_tokens': 3}, 'model': 'fixture-model'}
        return {'action':'reply','content':'I cannot complete this request.'}


def wait_done(service, job_id):
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        state=service.snapshot(job_id)
        if state['status'] in ('completed','error','canceled'): return state
        time.sleep(.01)
    raise AssertionError('comparison did not terminate')


class ComparisonExecutionTests(unittest.TestCase):
    def test_both_real_pipelines_share_input_and_emit_measured_model_events(self):
        transports=[]
        def factory(model):
            p=TransportFixture();transports.append(p);return p
        with tempfile.TemporaryDirectory() as d:
            service=ComparisonService(d,provider_factory=factory,model_catalog=lambda:['fixture-model'])
            job=service.start('handoff_drop_context','fixture-model')
            state=wait_done(service,job['id'])
            self.assertEqual(state['status'],'completed')
            self.assertEqual(set(state['results']),{'agentrx','agentreroute'})
            self.assertEqual(len(transports),2)
            a,b=(state['results'][s] for s in ('agentrx','agentreroute'))
            self.assertEqual(a['input_hash'],b['input_hash'])
            self.assertGreater(a['metrics']['model_calls'],0)
            self.assertGreater(b['metrics']['model_calls'],0)
            self.assertGreater(a['metrics']['prompt_tokens'],0)
            self.assertTrue(any(e['kind']=='request_started' for e in state['events']))
            self.assertTrue(any(e['kind']=='request_completed' for e in state['events']))
            self.assertEqual(a['prediction']['step_index'],6)
            for transport in transports:
                for stage,payload in transport.payloads:
                    self.assertNotIn('"faults"',json.dumps(payload))
                    self.assertNotIn('"gold"',json.dumps(payload))
            self.assertTrue((Path(d)/job['id']/'comparison.json').exists())
            after=state['events'][-2]['seq']
            self.assertEqual(len(service.snapshot(job['id'],after)['events']),1)
            service.close()

    def test_one_failed_system_does_not_inherit_other_success_or_stop_it(self):
        count=0
        def factory(model):
            nonlocal count
            count+=1
            return TransportFixture(fail=count==1)
        with tempfile.TemporaryDirectory() as d:
            service=ComparisonService(d,provider_factory=factory,model_catalog=lambda:['fixture-model'])
            state=wait_done(service,service.start('handoff_drop_context','fixture-model')['id'])
            self.assertEqual(state['results']['agentrx']['status'],'error')
            self.assertIsNone(state['results']['agentrx'].get('task_outcome'))
            self.assertNotEqual(state['results']['agentreroute']['status'],'error')
            self.assertTrue(any(e['kind']=='request_failed' for e in state['events']))
            service.close()

    def test_cancellation_stops_before_next_model_call_and_rejects_duplicate_jobs(self):
        entered,released=threading.Event(),threading.Event()
        class Slow(TransportFixture):
            def complete(self,stage,payload):
                entered.set();released.wait(5);return super().complete(stage,payload)
        with tempfile.TemporaryDirectory() as d:
            service=ComparisonService(d,provider_factory=lambda _:Slow(),model_catalog=lambda:['fixture-model'])
            job=service.start('handoff_drop_context','fixture-model')
            self.assertTrue(entered.wait(3))
            with self.assertRaises(RuntimeError):service.start('skip_write_claim','fixture-model')
            service.cancel(job['id']);released.set()
            state=wait_done(service,job['id'])
            self.assertEqual(state['status'],'canceled')
            self.assertNotIn('agentreroute',state['results'])
            self.assertEqual(sum(e['kind']=='request_started' for e in state['events']),1)
            service.close()

    def test_unknown_inputs_and_invalid_cursors_never_start_work(self):
        with tempfile.TemporaryDirectory() as d:
            service=ComparisonService(d,provider_factory=lambda _:TransportFixture(),model_catalog=lambda:['fixture-model'])
            for scenario,model in [('../escape','fixture-model'),('handoff_drop_context','not-installed')]:
                with self.assertRaises(ValueError):service.start(scenario,model)
            job=service.start('clean_control_flow','fixture-model')
            for cursor in (-1,True,'0'):
                with self.assertRaises(ValueError):service.snapshot(job['id'],cursor)
            wait_done(service,job['id']);service.close()

    def test_finished_state_is_published_only_after_artifacts_are_saved(self):
        from unittest.mock import patch
        from agentrx.io import write_json as real_write
        entered,released=threading.Event(),threading.Event()
        def slow_write(path,value):
            if Path(path).name=='comparison.json':
                entered.set();released.wait(5)
            real_write(path,value)
        with tempfile.TemporaryDirectory() as d:
            service=ComparisonService(d,provider_factory=lambda _:TransportFixture(),model_catalog=lambda:['fixture-model'])
            with patch('demo.comparison.write_json',side_effect=slow_write):
                job=service.start('handoff_drop_context','fixture-model')
                self.assertTrue(entered.wait(3))
                try:
                    # A nonblocking status flag is not public yet; snapshot must wait for artifact commit.
                    self.assertEqual(service.jobs[job['id']]['status'],'running')
                finally: released.set()
                state=wait_done(service,job['id'])
                self.assertTrue((Path(d)/job['id']/'comparison.json').exists())
            service.close()

    def test_completed_jobs_are_evicted_at_the_configured_retention_limit(self):
        with tempfile.TemporaryDirectory() as d:
            service=ComparisonService(d,provider_factory=lambda _:TransportFixture(),model_catalog=lambda:['fixture-model'],max_jobs=2)
            ids=[]
            for _ in range(3):
                job=service.start('clean_control_flow','fixture-model');ids.append(job['id']);wait_done(service,job['id'])
            self.assertNotIn(ids[0],service.jobs)
            self.assertEqual(len(service.jobs),2)
            self.assertEqual(service.snapshot(ids[-1])['status'],'completed')
            self.assertEqual(service.snapshot(ids[0])['status'],'completed')
            self.assertEqual(len(service.jobs),2)
            service.close()

    def test_finished_comparison_survives_server_restart_without_new_inference(self):
        with tempfile.TemporaryDirectory() as d:
            service=ComparisonService(d,provider_factory=lambda _:TransportFixture(),model_catalog=lambda:['fixture-model'])
            job=service.start('handoff_drop_context','fixture-model');original=wait_done(service,job['id']);service.close()
            def forbidden(_): raise AssertionError('Restoring a run must not invoke a model')
            restored=ComparisonService(d,provider_factory=forbidden,model_catalog=lambda:[])
            snapshot=restored.snapshot(job['id'])
            self.assertEqual(snapshot['results'],original['results'])
            self.assertEqual(snapshot['events'],original['events'])
            with self.assertRaises(KeyError):restored.snapshot('../escape')
            restored.close()

    def test_model_budget_is_enforced_before_sending_an_extra_request(self):
        events=[]
        with tempfile.TemporaryDirectory() as d:
            p=MeasuredProvider(TransportFixture(),events.append,Path(d),threading.Event(),max_calls=1)
            p.complete('global',{})
            with self.assertRaisesRegex(RuntimeError,'budget'):p.complete('dynamic',{})
            self.assertEqual(p.metrics()['model_calls'],1)


class ComparisonHTTPTests(unittest.TestCase):
    def setUp(self):
        from demo.visualizer import make_server
        self.temp=tempfile.TemporaryDirectory()
        self.service=ComparisonService(self.temp.name,provider_factory=lambda _:TransportFixture(),model_catalog=lambda:['fixture-model'])
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
        status,body=self.fetch('/api/comparisons',{'scenario':'handoff_drop_context','model':'fixture-model'},origin=self.base)
        self.assertEqual(status,202)
        jid=json.loads(body)['id'];wait_done(self.service,jid)
        state=json.loads(self.fetch('/api/comparisons/'+jid+'?after=0')[1])
        self.assertEqual(set(state['results']),{'agentrx','agentreroute'})
        self.assertEqual(state['status'],'completed')
        status,body=self.fetch('/api/comparisons/'+jid+'/cancel',{})
        self.assertEqual(status,200)

    def test_cross_origin_invalid_bodies_and_cursors_are_rejected(self):
        from urllib.error import HTTPError
        payload={'scenario':'handoff_drop_context','model':'fixture-model'}
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
