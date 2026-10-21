import json
import shutil
import tempfile
import unittest
from pathlib import Path

from recovery_orchestrator.rca import (RCAPayload, aggregate, list_trajectories, load_payload,
                                       trace_dir_name)
from recovery_orchestrator.taxonomy import normalize_agent, normalize_category

from .helpers import SMOKE_ID, SMOKE_RUN, SYN_ID, SYN_RUN


class TaxonomyTest(unittest.TestCase):
    def test_spelling_variants(self):
        self.assertEqual(normalize_category('Instruction/Plan Adherence Failure'), 'Instruction/PlanAdherenceFailure')
        self.assertEqual(normalize_category('Misinterpretation of Tool Output / Handoff Failure'),
                         'MisinterpretationOfToolOutput')
        self.assertEqual(normalize_category(10), 'Inconclusive')
        self.assertEqual(normalize_category('cycle'), 'Cycle')
        with self.assertRaises(ValueError):
            normalize_category('Drift')

    def test_agent(self):
        self.assertEqual(normalize_agent('Orchestrator (-> WebSurfer)'), 'orchestrator')
        self.assertEqual(normalize_agent('Assistant'), 'assistant')
        self.assertEqual(normalize_agent('human'), 'user')


class AdapterTest(unittest.TestCase):
    def test_dir_name_matches_fa(self):
        self.assertTrue((SMOKE_RUN / 'jobs/agentrx/rep-0/tau' / trace_dir_name(SMOKE_ID)).is_dir())

    def test_smoke_payload_pools_full_agentrx_only(self):
        p = load_payload(SMOKE_RUN, SMOKE_ID, 'tau')
        self.assertEqual(p.variants, ('agentrx',))
        self.assertEqual((p.n_runs, p.total_runs, p.confidence, p.inconclusive_share), (3, 3, 1.0, 0.0))
        self.assertEqual(p.top.step_index, 3)
        self.assertEqual(p.top.category, 'MisinterpretationOfToolOutput')
        self.assertEqual(p.violations_by_step[3]['runs'], 3)
        self.assertEqual(p.prediction['step_index'], 3)

    def test_synthetic_disagreement(self):
        p = load_payload(SYN_RUN, SYN_ID, 'tau')
        self.assertEqual(p.total_runs, 4)          # baseline variant excluded, error run included
        self.assertEqual(p.n_runs, 3)
        self.assertEqual(p.inconclusive_share, 0.25)
        self.assertEqual([(c.step_index, c.votes, c.share) for c in p.candidates], [(7, 2, 0.5), (9, 1, 0.25)])
        self.assertEqual(p.candidates[0].category, 'MisinterpretationOfToolOutput')  # aliases merged
        self.assertEqual(p.candidates[1].category, 'Instruction/PlanAdherenceFailure')
        self.assertEqual(p.candidates[0].agent, 'assistant')
        self.assertEqual(p.status, 'partial_checker_errors')
        self.assertEqual(p.violations_by_step[9], {'count': 2, 'runs': 2,
                                                   'categories': ['Instruction/PlanAdherenceFailure']})

    def test_explicit_variants(self):
        p = load_payload(SYN_RUN, SYN_ID, 'tau', variants=['baseline'])
        self.assertEqual((p.top.step_index, p.top.category, p.confidence), (3, 'InvalidInvocation', 1.0))

    def test_list_and_missing(self):
        self.assertEqual(list_trajectories(SYN_RUN), [('tau', SYN_ID)])
        with self.assertRaises(FileNotFoundError):
            load_payload(SYN_RUN, 'nope', 'tau')

    def test_inconclusive_and_error(self):
        results = [{'trajectory_id': 't', 'domain': 'tau', 'status': 'ok',
                    'prediction': {'step_index': None, 'agent': None, 'category': 'Inconclusive',
                                   'rationale': 'x', 'custom_category': 'unclear'}},
                   {'trajectory_id': 't', 'domain': 'tau', 'status': 'error', 'prediction': None}]
        p = aggregate(results)
        self.assertTrue(p.is_inconclusive)
        self.assertEqual((p.confidence, p.inconclusive_share, p.status), (0.0, 1.0, 'partial_checker_errors'))

    def test_tie_prefers_earlier_step(self):
        mk = lambda s: {'trajectory_id': 't', 'domain': 'tau', 'status': 'ok',
                        'prediction': {'step_index': s, 'agent': 'a', 'category': 'InvalidInvocation', 'rationale': 'x'}}
        p = aggregate([mk(9), mk(4)])
        self.assertEqual([c.step_index for c in p.candidates], [4, 9])

    def test_round_trip_and_official_payload(self):
        p = load_payload(SYN_RUN, SYN_ID, 'tau')
        q = RCAPayload.from_dict(json.loads(json.dumps(p.to_dict(), default=list)))
        self.assertEqual([(c.step_index, c.category, c.share) for c in q.candidates],
                         [(c.step_index, c.category, c.share) for c in p.candidates])
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            shutil.copytree(SYN_RUN, run)
            d = run / 'jobs/agentrx/rep-0/tau' / trace_dir_name(SYN_ID)
            (d / 'rca_payload.json').write_text(json.dumps({
                'trajectory_id': SYN_ID, 'domain': 'tau', 'status': 'ok', 'n_runs': 3, 'total_runs': 3,
                'inconclusive_share': 0.0, 'confidence': 0.67,
                'top_k': [{'step_index': 9, 'agent': 'Assistant', 'category': 'Plan Adherence Failure',
                           'votes': 2, 'share': 0.67}],
                'causal_chain': [{'step_index': 7}, {'step_index': 9}]}))
            official = load_payload(run, SYN_ID, 'tau')
            self.assertEqual(official.top.step_index, 9)
            self.assertEqual(official.causal_chain, [{'step_index': 7}, {'step_index': 9}])


if __name__ == '__main__':
    unittest.main()
