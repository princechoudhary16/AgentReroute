import unittest
from dataclasses import dataclass

from recovery_orchestrator.adapters.benchmark import OrchestratorPolicy, payload_from_hypotheses
from recovery_orchestrator.policy import EVConfig, EVPolicy
from recovery_orchestrator.rca import aggregate_handoffs
from recovery_orchestrator.registry import mock_registry
from recovery_orchestrator.taxonomy import category_from_mode
from recovery_orchestrator.tau_retail import WRITE_TOOLS
from recovery_orchestrator.verifier import VerifierResult

from .helpers import GOOD_REPLY, SYN_ID, syn_trace


def handoff(decision='repair_required', status='ok', attribution='ranked', candidates=()):
    return {'schema_version': 'agentrx-recovery-v1', 'trajectory_id': 't', 'domain': 'tau', 'status': status,
            'decision': decision, 'attribution_status': attribution, 'candidates': list(candidates)}


def cand(step, category, conf, rank=1):
    return {'rank': rank, 'step_index': step, 'agent': 'Assistant', 'category': category, 'confidence': conf,
            'rationale': f'r{step}', 'evidence_refs': [{'step_index': step, 'substep_index': 1}]}


class TrackBHandoffTest(unittest.TestCase):
    def test_pooling_and_gate(self):
        p = aggregate_handoffs([
            handoff(candidates=[cand(7, 'MisinterpretationOfToolOutput', 0.8), cand(9, 'Plan Adherence Failure', 0.4, 2)]),
            handoff(candidates=[cand(7, 'MisinterpretationOfToolOutput', 0.6)]),
            handoff(decision='inconclusive', attribution='not_requested'),
            handoff(status='error', candidates=[cand(3, 'InvalidInvocation', 0.9)]),   # FA gate: don't act on it
        ])
        self.assertEqual([(c.step_index, round(c.share, 3), c.votes) for c in p.candidates], [(7, 0.35, 2), (9, 0.1, 1)])
        self.assertEqual(p.candidates[1].category, 'Instruction/PlanAdherenceFailure')
        self.assertEqual((p.n_runs, p.total_runs, p.inconclusive_share), (2, 4, 0.5))
        self.assertEqual((p.fa_decision, p.confidence_kind), ('repair_required', 'uncalibrated_model_estimate'))

    def test_within_boundary_abstains(self):
        p = aggregate_handoffs([handoff(decision='within_boundary', attribution='not_requested')] * 2)
        policy = EVPolicy(mock_registry(), EVConfig(respect_irreversible=False))
        d = policy.decide(p, syn_trace(), policy.make_guard())
        self.assertEqual(d.action, 'abstain')
        self.assertIn('within boundary', d.reason)

    def test_rejects_unknown_schema(self):
        with self.assertRaises(ValueError):
            aggregate_handoffs([{**handoff(), 'schema_version': 'v2'}])


class ModeMappingTest(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(category_from_mode('context_propagation'), 'MisinterpretationOfToolOutput')
        self.assertEqual(category_from_mode('cycles'), 'Cycle')
        self.assertEqual(category_from_mode('Plan Adherence Failure'), 'Instruction/PlanAdherenceFailure')

    def test_payload_from_hypotheses(self):
        p = payload_from_hypotheses('t', 'tau', [
            {'step_index': 9, 'mode': 'tool_failure', 'confidence': 0.6},
            {'step_index': 7, 'mode': 'drift', 'confidence': 0.6},
            {'step_index': 7, 'mode': 'cycles', 'confidence': 0.9},          # duplicate step: first wins
            {'step_index': None, 'mode': 'drift'}])
        self.assertEqual([(c.step_index, c.category) for c in p.candidates],
                         [(7, 'IntentPlanMisalignment'), (9, 'SystemFailure')])


# ---- a fake of the benchmark harness, enough for the adapter -----------------------------------
@dataclass
class FakeDirective:
    rollback_step: int
    kind: str = 'guidance'
    text: str = ''


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class FakeRollout:
    rollout_id: str
    trace: dict
    tokens: int
    termination: str


class FakeEnv:
    def __init__(self, fix_on_guidance=True, budget=10):
        base = syn_trace()
        self.trajectory_id = SYN_ID
        self.trace = {'trajectory_id': SYN_ID, 'domain': 'tau', 'instruction': base.instruction, 'steps': [
            {'index': s.index, 'substeps': [{'sub_index': 1, 'role': s.role, 'content': s.content,
                                             **({'name': s.name} if s.name else {}),
                                             'tool_calls': [{'id': c.id, 'name': c.name, 'arguments': c.arguments}
                                                            for c in s.tool_calls]}]} for s in base.steps]}
        self.task_view = {'first_user_message': base.instruction, 'policy': 'p'}
        self.hypotheses = [{'step_index': 7, 'agent': 'assistant', 'mode': 'misinterpretation_of_tool_output',
                            'confidence': 0.8, 'rationale': 'misread items'}]
        self.fix, self.budget, self.calls = fix_on_guidance, budget, []

    def remaining(self):
        return {'tokens': 100_000, 'attempts': self.budget - len(self.calls), 'seconds': 60}

    def replay(self, directive):
        if len(self.calls) >= self.budget:
            raise BudgetExceeded('attempts')
        self.calls.append(directive)
        good = self.fix and directive.kind == 'guidance'
        steps = [s for s in self.trace['steps'] if s['index'] < directive.rollback_step]
        steps.append({'index': directive.rollback_step, 'substeps': [
            {'sub_index': 1, 'role': 'assistant', 'content': GOOD_REPLY if good else 'only a Watch'}]})
        return FakeRollout(f'r{len(self.calls)}', {**self.trace, 'steps': steps}, 1000,
                           'user_stop' if good else 'max_steps')


class CheckVerifier:
    def __init__(self, environment):
        self.environment = environment

    def verify(self, task, rollout):
        ok = self.environment.raw[rollout.rollout_id].termination == 'user_stop'
        return VerifierResult(ok, float(ok), 'PASS' if ok else 'FAIL')


def make_policy(**kw):
    return OrchestratorPolicy(FakeDirective,
                              lambda: EVPolicy(mock_registry(), EVConfig(respect_irreversible=False),
                                               write_tools=WRITE_TOOLS),
                              CheckVerifier, **kw)


class BenchmarkAdapterTest(unittest.TestCase):
    def test_recovers_and_delivers_verified_rollout(self):
        env = FakeEnv()
        decision = make_policy().recover(env)
        self.assertEqual(decision['delivered'], 'r1')
        self.assertEqual(decision['status'], 'intervened')
        self.assertEqual((env.calls[0].rollback_step, env.calls[0].kind), (7, 'guidance'))
        self.assertIn('misread items', env.calls[0].text)

    def test_unverified_rollouts_are_not_delivered(self):
        env = FakeEnv(fix_on_guidance=False)
        decision = make_policy().recover(env)
        self.assertIsNone(decision['delivered'])
        self.assertTrue(decision['notes'].startswith(('escalated', 'aborted', 'partial')))

    def test_harness_budget_is_respected(self):
        env = FakeEnv(fix_on_guidance=False, budget=1)
        decision = make_policy().recover(env)
        self.assertEqual(len(env.calls), 1)
        self.assertIn('environment refused', decision['notes'])

    def test_no_hypotheses_abstains_without_replay(self):
        env = FakeEnv()
        env.hypotheses = []
        self.assertEqual(make_policy().recover(env)['status'], 'abstained')
        self.assertEqual(env.calls, [])


if __name__ == '__main__':
    unittest.main()
