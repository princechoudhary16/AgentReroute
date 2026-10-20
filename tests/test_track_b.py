import copy
import math
import tempfile
import unittest
from pathlib import Path
from agentrx.boundary import compute_boundary, boundary_decision, constraint_fingerprint
from agentrx.predictions import validate_candidates
from agentrx.handoff import build_handoff, validate_handoff, load_handoff
from agentrx.data import normalize


def trace():
    return normalize({'trajectory_id':'ranked-demo','steps':[
        {'index':1,'substeps':[{'role':'assistant','content':'initial action'}]},
        {'index':2,'substeps':[{'role':'tool','content':'observed output'}]},
        {'index':3,'substeps':[{'role':'assistant','content':'misinterpreted output'}]},
        {'index':4,'substeps':[{'role':'assistant','content':'downstream error'}]}]},'tau')


def check(verdict,step=3,identity='a',**extra):
    return {'verdict':verdict,'step_index':step,'rule_fingerprint':identity*64,**extra}


def candidate(step=3,confidence=.7):
    return {'step_index':step,'agent':'assistant','category':'MisinterpretationOfToolOutput',
            'confidence':confidence,'rationale':'The agent misread the recorded output and did not recover.',
            'evidence_refs':[{'step_index':2,'substep_index':1},{'step_index':step,'substep_index':1}]}


class BoundaryTests(unittest.TestCase):
    def test_only_applicable_conclusive_checks_enter_the_boundary(self):
        checks=[check('VIOL'),check('SAT',identity='b'),check('SKIP',identity='c'),
                check('ERROR',identity='d'),check('SAT',identity='e',unclear_count=1)]
        boundary=compute_boundary(checks,.3)
        self.assertEqual(boundary['conclusive_checks'],2)
        self.assertEqual(boundary['divergence_rate'],.5)
        self.assertEqual(boundary_decision(boundary),'repair_required')
        self.assertEqual(boundary['unresolved_checks'],1)
        self.assertEqual(boundary['error_checks'],1)
        self.assertEqual(boundary['skipped_checks'],1)

    def test_strict_threshold_equality_zero_one_and_no_evidence(self):
        rows=[check('VIOL'),check('SAT',identity='b')]
        self.assertEqual(boundary_decision(compute_boundary(rows,.5)),'within_boundary')
        self.assertEqual(boundary_decision(compute_boundary(rows,0)),'repair_required')
        self.assertEqual(boundary_decision(compute_boundary(rows,1)),'within_boundary')
        for rows in ([],[check('SKIP')],[check('ERROR')],[check('SAT',unclear_count=1)]):
            boundary=compute_boundary(rows,.3)
            self.assertIsNone(boundary['divergence_rate'])
            self.assertEqual(boundary_decision(boundary),'inconclusive')
        for threshold in (True,float('nan'),float('inf'),-.1,1.1):
            with self.assertRaises(ValueError):compute_boundary([],threshold)

    def test_deduplication_is_per_rule_and_step_and_conflicts_are_unresolved(self):
        rows=[check('VIOL'),check('VIOL'),check('SAT',step=4),check('SAT',identity='b')]
        b=compute_boundary(rows,.3)
        self.assertEqual(b['conclusive_checks'],3)
        self.assertEqual(b['duplicate_evaluations'],1)
        conflict=compute_boundary([check('SAT'),check('VIOL')],.3)
        self.assertEqual(conflict['conflicting_checks'],1)
        self.assertEqual(boundary_decision(conflict),'inconclusive')

    def test_python_fingerprints_ignore_label_formatting_and_function_name(self):
        first={'check_type':'python_check','event_trigger':{'step_index':3},
               'python_check':{'function_name':'one','code_lines':['def one(trajectory, offset):','    return False']}}
        second=copy.deepcopy(first);second['assertion_name']='different'
        second['python_check']={'function_name':'two','code_lines':['# comment','def two(trajectory,offset):',' return False']}
        self.assertEqual(constraint_fingerprint(first),constraint_fingerprint(second))
        second['python_check']['code_lines'][-1]=' return True'
        self.assertNotEqual(constraint_fingerprint(first),constraint_fingerprint(second))


class CandidateAndHandoffTests(unittest.TestCase):
    def test_ranked_candidates_are_sorted_and_scores_are_not_normalized(self):
        result=validate_candidates({'candidates':[candidate(4,.6),candidate(3,.8)],'abstention_reason':None},trace(),3)
        self.assertEqual([c['step_index'] for c in result['candidates']],[3,4])
        self.assertEqual([c['rank'] for c in result['candidates']],[1,2])
        self.assertEqual(sum(c['confidence'] for c in result['candidates']),1.4)

    def test_invalid_candidates_and_unsupported_evidence_are_rejected(self):
        for field,value in [('confidence',True),('confidence',float('nan')),('confidence',1.2),
                            ('step_index',999),('agent','invented'),('category','invented'),
                            ('category','Inconclusive'),('rationale',''),('evidence_refs',[]),
                            ('evidence_refs',[{'step_index':True,'substep_index':1}])]:
            c=candidate();c[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                validate_candidates({'candidates':[c],'abstention_reason':None},trace(),3)
        with self.assertRaises(ValueError):validate_candidates({'candidates':[candidate(),candidate()],'abstention_reason':None},trace(),3)
        with self.assertRaises(ValueError):validate_candidates({'candidates':[],'abstention_reason':''},trace(),3)
        self.assertEqual(validate_candidates({'candidates':[],'abstention_reason':'Insufficient evidence'},trace(),3)['candidates'],[])

    def test_handoff_alarm_survives_ranker_abstention_and_is_versioned(self):
        boundary=compute_boundary([check('VIOL')],.3)
        payload=build_handoff(trace(),boundary,abstention_reason='No critical step can be grounded')
        self.assertEqual(payload['decision'],'repair_required')
        self.assertEqual(payload['attribution_status'],'abstained')
        self.assertEqual(payload['candidates'],[])
        self.assertEqual(payload['confidence_kind'],'uncalibrated_model_estimate')
        self.assertEqual(payload['taxonomy_version'],'agentrx-9-v1')
        self.assertEqual(validate_handoff(payload,trace()),payload)

    def test_consumer_validates_counts_ranks_identity_and_original_checks(self):
        checks=[check('VIOL')];b=compute_boundary(checks,.3)
        candidates=validate_candidates({'candidates':[candidate()],'abstention_reason':None},trace(),3)['candidates']
        payload=build_handoff(trace(),b,candidates=candidates)
        for mode in ('rate','counts','decision','trajectory','rank','confidence_kind'):
            invalid=copy.deepcopy(payload)
            if mode=='rate':invalid['boundary']['divergence_rate']=0
            if mode=='counts':invalid['boundary']['conclusive_checks']=2
            if mode=='decision':invalid['decision']='within_boundary'
            if mode=='trajectory':invalid['trajectory_id']='another'
            if mode=='rank':invalid['candidates'][0]['rank']=2
            if mode=='confidence_kind':invalid['confidence_kind']='calibrated_probability'
            with self.subTest(mode=mode),self.assertRaises(ValueError):validate_handoff(invalid,trace())
        with tempfile.TemporaryDirectory() as tmp:
            from agentrx.io import write_json
            path=Path(tmp)/'recovery_handoff.json';write_json(path,payload)
            self.assertEqual(load_handoff(path,trace(),checks=checks),payload)
            with self.assertRaises(ValueError):load_handoff(path,trace(),checks=[check('SAT')])


class RankedPipelineTests(unittest.TestCase):
    def provider(self, *, body='return False', ranking=None, fail_stage=None):
        from tests.test_constraints import python_constraint
        class Fixture:
            measurement_kind='offline_fixture'
            identity={'provider':'track-b-fixture'}
            def __init__(self): self.requests=[]
            def complete(self,stage,payload):
                self.requests.append((stage,copy.deepcopy(payload)))
                if stage == fail_stage: raise RuntimeError('deliberate transport failure')
                if stage == 'global': return {'constraints':[] if body is None else [python_constraint([body],{'step_index':3})]}
                if stage == 'dynamic': return {'constraints':[]}
                if stage == 'ranked_judge': return ranking or {'candidates':[candidate(4,.6),candidate(3,.8)],'abstention_reason':None}
                raise AssertionError(stage)
        return Fixture()

    def run_case(self,provider,**extra):
        from agentrx.pipeline import diagnose
        from agentrx.io import read_json
        with tempfile.TemporaryDirectory() as tmp:
            result=diagnose(trace(),'Policy',{}, {'protocol':'ranked','divergence_threshold':.3,**extra},provider,tmp)
            self.assertEqual(read_json(Path(tmp)/'recovery_handoff.json'),result['recovery_handoff'])
            self.assertFalse((Path(tmp)/'rca_payload.json').exists())
            validate_handoff(result['recovery_handoff'],trace(),result['checks'])
            return result

    def test_ranked_config_requires_evidence_explicit_threshold_and_2_or_3_candidates(self):
        from agentrx.pipeline import validate_config
        for cfg in ({'protocol':'ranked'}, {'protocol':'ranked','evidence':False,'divergence_threshold':.3},
                    {'protocol':'ranked','divergence_threshold':True}, {'protocol':'ranked','divergence_threshold':.3,'max_candidates':1},
                    {'protocol':'joint','divergence_threshold':.3}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError): validate_config(cfg)
        self.assertEqual(validate_config({'protocol':'ranked','divergence_threshold':.3})['max_candidates'],3)

    def test_alarm_calls_ranker_once_and_preserves_compatible_top_one(self):
        provider=self.provider();result=self.run_case(provider)
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['prediction']['step_index'],3)
        self.assertNotIn('confidence',result['prediction'])
        handoff=result['recovery_handoff']
        self.assertEqual(handoff['attribution_status'],'ranked')
        self.assertEqual([c['step_index'] for c in handoff['candidates']],[3,4])
        payloads=[p for s,p in provider.requests if s=='ranked_judge']
        self.assertEqual(len(payloads),1)
        self.assertEqual(payloads[0]['boundary']['divergence_rate'],1)
        self.assertNotIn('gold',str(payloads))

    def test_within_boundary_and_no_conclusive_checks_skip_judging(self):
        for body,decision in [('return True','within_boundary'),(None,'inconclusive')]:
            provider=self.provider(body=body);result=self.run_case(provider)
            self.assertEqual(result['recovery_handoff']['decision'],decision)
            self.assertEqual(result['recovery_handoff']['attribution_status'],'not_requested')
            self.assertEqual(result['prediction']['category'],'Inconclusive')
            self.assertNotIn('ranked_judge',[s for s,p in provider.requests])

    def test_ranker_abstention_error_and_generation_failure_preserve_handoff(self):
        abstained=self.run_case(self.provider(ranking={'candidates':[],'abstention_reason':'No causal step is supported'}))
        self.assertEqual(abstained['recovery_handoff']['decision'],'repair_required')
        self.assertEqual(abstained['recovery_handoff']['attribution_status'],'abstained')
        for stage,decision in [('ranked_judge','repair_required'),('global','inconclusive')]:
            result=self.run_case(self.provider(fail_stage=stage))
            self.assertEqual(result['status'],'error')
            self.assertIsNone(result['prediction'])
            self.assertEqual(result['recovery_handoff']['attribution_status'],'error')
            self.assertEqual(result['recovery_handoff']['decision'],decision)
        partial=self.run_case(self.provider(body='return 3'))
        self.assertEqual(partial['status'],'partial_checker_errors')
        self.assertEqual(partial['recovery_handoff']['decision'],'inconclusive')

    def test_ranked_prompt_and_schema_bind_observed_locations_without_labels(self):
        from agentrx.prompts import messages, ranked_response_schema
        payload={'trajectory':trace(),'max_candidates':2,'checklist':True,'boundary':compute_boundary([check('VIOL')],.3)}
        prompt=messages('ranked_judge',payload)
        self.assertIn('uncalibrated',prompt[0]['content'])
        schema=ranked_response_schema(payload)
        rows=schema['properties']['candidates']
        self.assertEqual(rows['maxItems'],2)
        self.assertEqual(rows['items']['properties']['step_index']['enum'],[1,2,3,4])
        self.assertNotIn('Inconclusive',rows['items']['properties']['category']['enum'])


class RankedReportTests(unittest.TestCase):
    def test_failed_abstained_and_missing_jobs_stay_in_top_k_denominator(self):
        from agentrx.experiments import run_experiment,evaluate_run,job_path
        from agentrx.io import write_json,read_json,digest
        trajectory=trace()
        from agentrx.data import normalize_gold
        gold=normalize_gold({'trajectory_id':'ranked-demo','failures':[{'failure_id':1,'step_number':3,'failed_agent':'assistant','failure_category':'MisinterpretationOfToolOutput'}],'root_cause':{'failure_id':1}})
        rows=[{'trajectory':trajectory,'policy':'','tools':{},'gold':gold}]
        with tempfile.TemporaryDirectory() as tmp:
            corpus=Path(tmp)/'corpus';out=Path(tmp)/'out'
            write_json(corpus/'cases.json',rows)
            write_json(corpus/'manifest.json',{'corpus_hash':digest(rows),'domains':{'tau':{'available':1,'expected':29,'status':'fixture_only'}}})
            provider=RankedPipelineTests().provider()
            config={'repetitions':3,'domains':['tau'],'variants':[{'name':'ranked','protocol':'ranked','divergence_threshold':.3}]}
            run_experiment(corpus,config,out,lambda _:provider)
            report=read_json(out/'handoff_summary.json')['results'][0]
            self.assertEqual(report['denominator'],3)
            self.assertEqual(report['top_k_step_accuracy'],{'1':1.0,'2':1.0,'3':1.0})
            path=job_path(out,'ranked',0,trajectory)/'result.json'
            path.unlink()
            evaluate_run(corpus,out)
            report=read_json(out/'handoff_summary.json')['results'][0]
            self.assertEqual(report['denominator'],3)
            self.assertEqual(report['missing_jobs'],1)
            self.assertEqual(report['top_k_step_accuracy']['1'],2/3)
            bins=report['confidence_bins']
            self.assertEqual(sum(b['count'] for b in bins),4)

            for rep in (1,2):
                path=job_path(out,'ranked',rep,trajectory)/'result.json'
                result=read_json(path)
                if rep==1:
                    result.update(status='error',prediction=None)
                    result['recovery_handoff']=build_handoff(trajectory,compute_boundary(result['checks'],.3),status='error',measurement_kind='offline_fixture')
                else:
                    result['prediction']={'step_index':None,'agent':None,'category':'Inconclusive','rationale':'Insufficient causal evidence','custom_category':'abstained'}
                    result['recovery_handoff']=build_handoff(trajectory,compute_boundary(result['checks'],.3),measurement_kind='offline_fixture')
                write_json(path,result)
            evaluate_run(corpus,out)
            report=read_json(out/'handoff_summary.json')['results'][0]
            self.assertEqual(report['top_k_step_accuracy'],{'1':0.0,'2':0.0,'3':0.0})
            self.assertEqual(report['failed_jobs'],1)
            self.assertEqual(report['attribution_counts'],{'error':1,'abstained':1})
