import unittest
from agentrx.data import normalize, normalize_gold
from agentrx.evaluation import score, aggregate


def case(tid='a'):
    return {'trajectory': normalize({'trajectory_id':tid,'steps':[
        {'index':2,'substeps':[{'role':'Orchestrator (thought)','content':'early'}]},
        {'index':7,'substeps':[{'role':'Orchestrator (-> Assistant)','content':'fatal'}]},
        {'index':9,'substeps':[{'role':'Assistant','content':'last'}]}]},'magentic'),
        'gold': normalize_gold({'trajectory_id':tid,'failures':[
            {'failure_id':1,'step_number':9,'failure_category':'System Failure','failed_agent':'Assistant'},
            {'failure_id':9,'step_number':2,'failure_category':'Invalid Invocation','failed_agent':'Orchestrator'},
            {'failure_id':3,'step_number':7,'failure_category':'Invention of new information','failed_agent':'Orchestrator'}],
            'root_cause':{'failure_id':3}}), 'policy':'','tools':{}}


def prediction(tid='a', step=7, category='InventionOfNewInformation'):
    return {'domain':'magentic','trajectory_id':tid,'status':'ok','prediction': {
        'step_index':step,'agent':'Orchestrator','category':category,'rationale':'unrecovered',
        'evidence_refs':[{'step_index':step,'substep_index':1}]}}


class EvaluationTests(unittest.TestCase):
    def test_missing_job_keeps_denominator_and_original_index_distance(self):
        result=score([case('a'),case('b')],[prediction('a',step=2,category='InvalidInvocation')])
        self.assertEqual(result['denominator'],2)
        self.assertEqual(result['missing_jobs'],1)
        self.assertEqual(result['metrics']['step_accuracy'],0)
        self.assertEqual(result['metrics']['acc_at_5'],.5)
        self.assertEqual(result['metrics']['average_step_distance'],5)
        self.assertEqual(result['valid_step_predictions'],1)
        self.assertEqual(result['metrics']['earliest_category_accuracy'],.5)
        self.assertEqual(result['metrics']['terminal_category_accuracy'],0)
        self.assertEqual(result['metrics']['agent_accuracy'],.5)

    def test_agent_aliases_match_orchestrator_decorated_roles(self):
        result=score([case()],[prediction()])
        self.assertEqual(result['metrics']['step_accuracy'],1)
        self.assertEqual(result['metrics']['critical_category_accuracy'],1)
        self.assertEqual(result['metrics']['agent_accuracy'],1)

    def test_abstention_invalid_and_error_never_count_as_correct(self):
        abstain=prediction('a'); abstain['prediction'].update(step_index=None,agent=None,
            category='Inconclusive',custom_category='insufficient evidence')
        invalid=prediction('b',999)
        error=prediction('c'); error.update(status='error',prediction=None)
        result=score([case('a'),case('b'),case('c')],[abstain,invalid,error])
        self.assertEqual(result['abstentions'],1)
        self.assertEqual(result['invalid_predictions'],1)
        self.assertEqual(result['failed_jobs'],1)
        self.assertEqual(result['metrics']['acc_at_5'],0)
        self.assertIsNone(result['metrics']['average_step_distance'])

    def test_unknown_ids_and_duplicates_are_rejected(self):
        with self.assertRaises(ValueError): score([case()],[prediction('unknown')])
        with self.assertRaises(ValueError): score([case()],[prediction(),prediction()])

    def test_aggregate_is_mean_of_repetitions_and_sample_sd(self):
        result=aggregate([{'metrics':{'step_accuracy':0}}, {'metrics':{'step_accuracy':1}},
                          {'metrics':{'step_accuracy':1}}])
        self.assertAlmostEqual(result['step_accuracy']['mean'],2/3)
        self.assertAlmostEqual(result['step_accuracy']['std'],.57735026919)
        self.assertEqual(result['step_accuracy']['n'],3)

    def test_tied_earliest_failures_use_any_label_at_that_step(self):
        c=case()
        c['gold']['failures'].append({'failure_id':'10','step_index':2,
            'category':'InventionOfNewInformation','agent':'Orchestrator'})
        measured=score([c],[prediction(step=2)])
        self.assertEqual(measured['metrics']['earliest_category_accuracy'],1)
