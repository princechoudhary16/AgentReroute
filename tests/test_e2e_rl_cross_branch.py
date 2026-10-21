import json
import tempfile
import unittest
from pathlib import Path

from agentrx.io import read_json
from agentrx.pipeline import diagnose_mp_rl
from agentrx.rl_cross_branch import (MPRLOrchestratorSession, MPRLRecoveryPolicy, candidates_to_benchmark_hypotheses,
                                     train_and_evaluate_benchmark, trace_tools_and_policy)
from agentrx.rl_policy import THETA0, DualLoopRLJudge, verifier_reward
from benchmark.env.scripted import scripted_llm
from benchmark.scoring.baselines import from_agentrx_run
from benchmark.scoring.recovery import POLICIES, run_recovery
from benchmark.tests.helpers import corpus
from recovery_orchestrator.orchestrator import RECOVERED
from recovery_orchestrator.policy import INTERVENE, Decision, EVConfig, EVPolicy, Option
from recovery_orchestrator.rca import RCAPayload, is_full_agentrx, load_payload, trace_dir_name
from recovery_orchestrator.registry import StrategyRegistry
from recovery_orchestrator.tau_retail import WRITE_TOOLS
from recovery_orchestrator.tests import helpers
from recovery_orchestrator.trace import Trace
from repair_strategies.adapters.orchestrator import export_registry
from tests.test_constraints import python_constraint
from tests.test_mp_rl import FakeProvider

P3 = 'P3_HANDOFF_GROUNDING'
MP_RL = {'protocol': 'mp_rl', 'evidence': True, 'checklist': True, 'recovery_loop': False}


def syn_ir():
    return read_json(next(helpers.SYN_RUN.glob('jobs/agentrx/rep-0/*/*/trajectory_ir.json')))


def items_named():
    """P3 constraint grounded in the synthetic trace: a reply must name every item of the latest order lookup."""
    c = python_constraint([
        "import json",
        "sub = trajectory['steps'][current_step_index]['substeps'][0]",
        "if sub['role'] != 'assistant' or not sub.get('content'):",
        "    return True",
        "outs = [s['substeps'][0] for s in trajectory['steps'][:current_step_index] "
        "if s['substeps'][0].get('name') == 'get_order_details']",
        "if not outs:",
        "    return True",
        "return all(i['name'] in sub['content'] for i in json.loads(outs[-1]['content']).get('items', []))"])
    c.update(assertion_name='reply_names_every_order_item', taxonomy_targets=['MisinterpretationOfToolOutput'],
             constraint_type='RELATIONAL_POST', check_hint='compare the reply with the latest order lookup')
    c['python_check']['function_name'] = c['assertion_name']
    c['python_check']['code_lines'][0] = f"def {c['assertion_name']}(trajectory, current_step_index):"
    return c


def grounded_provider():
    return FakeProvider(perspective_global=lambda p: {'constraints': [items_named()] if p['perspective'] == P3 else []},
                        perspective_dynamic={'constraints': []})


def session(fixes, judge):
    def script(trace, prefix, directive):
        start = prefix.rollback_step
        if fixes and directive.kind != 'none':
            return helpers.good_continuation(start), 1.0, 2000
        return helpers.bad_continuation(start), 0.0, 2000
    verifier, _ = helpers.make_verifier()
    return MPRLOrchestratorSession(helpers.make_env(script=script), verifier, grounded_provider(), judge,
                                   use_llm_synthesis=True, ev_config=EVConfig(respect_irreversible=False))


class _Tripwire(dict):
    """The harness ledger during recover(): rollouts may write rewards, nothing may read them."""
    def _read(self, *args, **kwargs):
        raise AssertionError('recovery policy read the reward ledger')
    __getitem__ = get = values = items = _read


class LedgerGuardedPolicy(MPRLRecoveryPolicy):
    def recover(self, env):
        env._runner.ledger = _Tripwire(env._runner.ledger)
        try:
            return super().recover(env)
        finally:
            env._runner.ledger = dict.copy(env._runner.ledger)


class CrossBranchRLTests(unittest.TestCase):
    def test_rca_payload_contract_into_orchestrator_and_ev_policy(self):
        ir = syn_ir()
        with tempfile.TemporaryDirectory() as run:
            artifacts = Path(run)/'jobs'/'mp_rl'/'rep-0'/ir['domain']/trace_dir_name(ir['trajectory_id'])
            result = diagnose_mp_rl(ir, 'Confirm before writes.', {}, MP_RL, grounded_provider(), artifacts)
            payload = RCAPayload.from_dict(read_json(artifacts/'rca_payload.json'))
            discovered = load_payload(run, ir['trajectory_id'], ir['domain'])     # variants=None pools mp_rl now
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(is_full_agentrx(result['config']))
        self.assertEqual(discovered.to_dict(), payload.to_dict())
        self.assertEqual((payload.top.step_index, payload.top.category), (7, 'MisinterpretationOfToolOutput'))
        self.assertEqual([c.step_index for c in payload.candidates], [7, 11])
        self.assertAlmostEqual(payload.confidence, result['ranked_candidates'][0]['probability'])
        self.assertEqual(payload.violations_by_step[7]['categories'], ['MisinterpretationOfToolOutput'])
        policy = EVPolicy(StrategyRegistry.from_dict(export_registry()), EVConfig(respect_irreversible=False),
                          write_tools=WRITE_TOOLS)
        decision = policy.decide(payload, Trace.from_ir(ir), policy.make_guard())
        self.assertIsInstance(decision, Decision)
        self.assertEqual(decision.action, INTERVENE)
        self.assertIsInstance(decision.option, Option)
        self.assertEqual(decision.option.primary.step_index, 7)
        self.assertLessEqual(decision.option.rollback_step, 7)
        self.assertGreater(decision.option.ev, 0)

    def test_mprl_orchestrator_session_closed_loop_weight_update(self):
        ir, trace = syn_ir(), Trace.from_ir(syn_ir())
        good, bad = DualLoopRLJudge(), DualLoopRLJudge()
        result, outcome, reward = session(True, good).run_episode(ir, trace, 'Confirm before writes.')
        self.assertEqual((outcome.status, reward), (RECOVERED, 1.0))
        self.assertTrue(outcome.attempts[-1].verifier['passed'])
        candidates = scored_candidates(result)
        top = result['ranked_candidates'][0]
        self.assertEqual((top['step_index'], top['perspective']), (7, P3))
        expected = [sum(c['probability'] * c['features'][j] for c in result['ranked_candidates']) for j in range(10)]
        step = [w - t for w, t in zip(good.weights, THETA0)]
        for j in range(10):     # theta += lr * (R - 0) * (h* - E_pi[h])
            self.assertAlmostEqual(step[j], 0.05 * 1.0 * (top['features'][j] - expected[j]), places=9)
        self.assertGreater(probability_of(good, candidates, 7), top['probability'])
        promoted = good.memory.retrieve_exemplars(P3)
        self.assertEqual([c['assertion_name'] for c in promoted], ['reply_names_every_order_item'])

        result, outcome, reward = session(False, bad).run_episode(ir, trace, 'Confirm before writes.')
        self.assertNotEqual(outcome.status, RECOVERED)
        self.assertLess(reward, 0)
        self.assertLess(probability_of(bad, candidates, 7), result['ranked_candidates'][0]['probability'])
        self.assertEqual(bad.memory.retrieve_exemplars(P3), [])
        self.assertEqual(next(iter(bad.memory.exemplars.values()))['failed_count'], 1)
        self.assertFalse(good.pending or bad.pending)
        # Scaled reward: early stop is the floor, and a worsened verifier score is below an unchanged one.
        self.assertEqual(verifier_reward(False, 0.2, 0.9, stopped_early=True), -1.0)
        self.assertLess(verifier_reward(False, 0.6, 0.2), verifier_reward(False, 0.2, 0.2))
        self.assertLess(verifier_reward(False, 0.2, 0.6), 0)

    def test_mprl_recovery_policy_on_benchmark_smoke_corpus(self):
        self.assertIsInstance(POLICIES['mp_rl_recovery'](), MPRLRecoveryPolicy)
        policy = LedgerGuardedPolicy()
        report = run_recovery(corpus(), 'train', policy, scripted_llm, include_controls=False, limit=2)
        self.assertEqual(report['policy'], 'mp_rl_recovery')
        self.assertEqual(set(policy.hypotheses), {r['trajectory_id'] for r in report['per_episode']})
        for tid, hypotheses in policy.hypotheses.items():
            self.assertTrue(hypotheses)
            self.assertEqual(len({h['step_index'] for h in hypotheses}), len(hypotheses))
            self.assertTrue(all({'category', 'mode', 'agent', 'confidence'} <= set(h) for h in hypotheses))
            self.assertTrue(any(a.get('rollout') for a in policy.log[tid].attempts))   # Directive -> env.replay
            self.assertIn('accepted', policy.log[tid].attempts[0])                    # acceptance.evaluate ran
        self.assertEqual(set(policy.rl_log), set(policy.hypotheses))
        self.assertTrue(all(-1.0 <= r['reward'] <= 1.0 for r in policy.rl_log.values()))
        self.assertNotEqual(policy.rl_judge.weights, list(THETA0))
        self.assertEqual(policy.rl_judge.episodes, 2)
        self.assertEqual(policy.rl_judge.pending, {})
        view = {'policy': 'p', 'tools': {'a': [{'name': 'x'}], 'b': [{'name': 'x'}, {'name': 'y'}]}}
        self.assertEqual(trace_tools_and_policy(view), ('p', {'x': {'name': 'x'}, 'y': {'name': 'y'}}))

    def test_from_agentrx_run_exports_mp_rl_ranked_candidates(self):
        ranked = [{'step_index': 4, 'agent': 'planner', 'raw_agent': 'planner', 'category': 'InvalidInvocation',
                   'probability': 0.612345},
                  {'step_index': 2, 'agent': 'planner', 'category': 'IntentPlanMisalignment', 'probability': 0.3},
                  {'step_index': 6, 'agent': 'planner', 'category': 'GuardrailsTriggered', 'probability': 0.05},
                  {'step_index': 8, 'agent': 'planner', 'category': 'InvalidInvocation', 'probability': 0.03}]
        joint = {'step_index': 3, 'agent': 'planner', 'category': 'InvalidInvocation'}
        with tempfile.TemporaryDirectory() as run:
            for variant, rep, tid, data in [('mp_rl', 0, 't-mp', {'prediction': None, 'ranked_candidates': ranked}),
                                            ('agentrx', 0, 't-joint', {'prediction': joint}),
                                            ('agentrx', 1, 't-joint', {'prediction': {**joint, 'step_index': 5}})]:
                path = Path(run)/'jobs'/variant/f'rep-{rep}'/'retail'/tid/'result.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'trajectory_id': tid, **data}), encoding='utf-8')
            rows = {r['trajectory_id']: r['hypotheses'] for r in from_agentrx_run(run)['predictions']}
        self.assertEqual([(h['step_index'], h['mode'], h['confidence']) for h in rows['t-mp']],
                         [(4, 'tool_failure', 0.6123), (2, 'drift', 0.3), (6, 'out_of_scope', 0.05)])
        self.assertEqual(rows['t-mp'], [{k: v for k, v in h.items() if k in rows['t-mp'][0]}
                                        for h in candidates_to_benchmark_hypotheses(ranked)[:3]])
        # No ranked list: the original repetition vote pooling is unchanged.
        self.assertEqual([(h['step_index'], h['confidence']) for h in rows['t-joint']], [(3, 0.5), (5, 0.5)])

    def test_train_and_evaluate_benchmark_freezes_policy_at_test_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)/'rl_state.json'
            out = train_and_evaluate_benchmark(corpus(), epochs=2, rl_state_path=state, limit=2)
            saved = DualLoopRLJudge.load(state)
        history = out['train_history']
        self.assertEqual([h['episodes'] for h in history], [2, 4])
        self.assertNotEqual(history[0]['weights'], list(THETA0))
        self.assertNotEqual(history[0]['weights'], history[1]['weights'])
        self.assertEqual(out['learned_weights'], history[-1]['weights'])     # test split never moved theta
        self.assertEqual((saved.weights, saved.episodes), (out['learned_weights'], 4))
        self.assertEqual(out['test_report']['split'], 'test')
        self.assertEqual(len(out['test_report']['per_episode']), 2)


def scored_candidates(result):
    return [{'step_index': c['step_index'], 'agent': c['agent'], 'features': c['features']}
            for c in result['ranked_candidates']]


def probability_of(judge, candidates, step):
    ranked = judge.select_action([{**c, 'offset': c['step_index']} for c in candidates])[1]
    return next(c['probability'] for c in ranked if c['step_index'] == step)


if __name__ == '__main__':
    unittest.main()
