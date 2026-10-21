import tempfile
import unittest
from pathlib import Path
from agentrx.io import digest,read_json,write_json
from agentrx.experiments import run_experiment, evaluate_run, experiment_variants
from tests.test_pipeline import TraceProvider
from tests.test_evaluation import case


class ExperimentTests(unittest.TestCase):
    def test_matrix_covers_six_columns_and_generation_variants(self):
        variants=experiment_variants({'matrix':'paper'})
        self.assertEqual(len(variants),9)
        self.assertEqual(len({v['name'] for v in variants}),9)
        self.assertEqual(sum(not v['evidence'] for v in variants),3)

    def test_repetitions_and_resume_are_independent_and_labels_never_sent(self):
        class Counter(TraceProvider):
            def complete(self, stage, payload):
                # A real protocol spy; only model transport is replaced.
                if 'gold' in str(payload) or 'failures' in payload:
                    raise AssertionError('Label leakage')
                if stage in ('judge','categorize'):
                    self.requests.append((stage,payload))
                    return {'step_index':7,'agent':'Orchestrator','category':'InventionOfNewInformation',
                            'rationale':'explicit fatal event','evidence_refs':[{'step_index':7,'substep_index':1}]}
                return super().complete(stage,payload)
        with tempfile.TemporaryDirectory() as d:
            corpus=Path(d)/'corpus'; rows=[case()]
            write_json(corpus/'cases.json',rows)
            write_json(corpus/'manifest.json',{'corpus_hash':digest(rows),'domains':{
                'magentic':{'available':1,'expected':44,'status':'available_subset'},
                'flash':{'available':0,'expected':42,'status':'blocked_missing_data'}}})
            provider=Counter()
            config={'name':'test','repetitions':3,'variants':[{'name':'baseline','evidence':False,'checklist':False}]}
            out=Path(d)/'run'
            summary=run_experiment(corpus,config,out,lambda _:provider)
            self.assertEqual(len(provider.requests),3)
            self.assertEqual(summary['measurement_kind'],'offline_fixture')
            run_experiment(corpus,config,out,lambda _:provider)
            self.assertEqual(len(provider.requests),3)
            self.assertEqual(summary['results'][0]['aggregate']['step_accuracy']['mean'],1)
            self.assertTrue((out/'table4.md').exists())
            self.assertTrue((out/'table7.csv').exists())
            changed={**config,'variants':[{'name':'baseline','evidence':False,'checklist':True}]}
            with self.assertRaisesRegex(ValueError,'identity'):
                run_experiment(corpus,changed,out,lambda _:provider)
            # Missing on-disk jobs retain the intended denominator on re-evaluation.
            (out/'jobs/baseline/rep-0/magentic'/digest('a')/'result.json').unlink()
            report=evaluate_run(corpus,out)
            self.assertEqual(report['results'][0]['repetitions'][0]['missing_jobs'],1)
            self.assertEqual(report['results'][0]['aggregate']['step_accuracy']['mean'],2/3)

    def test_secret_keys_cannot_be_persisted_in_experiment_config(self):
        from agentrx.experiments import validate_experiment
        for config in ({'client':{'provider':'gemini','api_key':'secret'}},
                       {'models':{'gemini':{'provider':'gemini','api_key':'secret'}}}):
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError,'secret|credential'):
                    validate_experiment(config)
