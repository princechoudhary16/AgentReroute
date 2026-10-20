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
