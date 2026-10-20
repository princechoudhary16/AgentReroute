import json
import tempfile
import unittest
from pathlib import Path

from recovery_orchestrator.budget import BudgetCaps
from recovery_orchestrator.llm import ScriptedLLM
from recovery_orchestrator.orchestrator import Orchestrator
from recovery_orchestrator.policy import BaselinePolicy, EVConfig, EVPolicy, UnmodifiedReplayPolicy
from recovery_orchestrator.rca import load_payload
from recovery_orchestrator.registry import mock_registry
from recovery_orchestrator.tau_retail import WRITE_TOOLS
from recovery_orchestrator.trace import Step, ToolCall
from recovery_orchestrator.verifier import FreshContextVerifier, render_transcript

from .helpers import GOOD_REPLY, SYN_ID, SYN_RUN, make_env, make_verifier, syn_trace

SIM = EVConfig(respect_irreversible=False)


def ev_policy():
    return EVPolicy(mock_registry(), SIM, write_tools=WRITE_TOOLS)


def scores(original, good, other):
    """Verifier that never passes; scores the original/good/other transcripts."""
    def reply(messages, purpose):
        text = messages[-1]['content']
        score = good if GOOD_REPLY in text else original if 'Done, the watch' in text else other
        return json.dumps({'verdict': 'FAIL', 'score': score, 'gaps': ['g']})
    return ScriptedLLM(reply)


class OrchestratorTest(unittest.TestCase):
    def setUp(self):
        self.trace, self.payload = syn_trace(), load_payload(SYN_RUN, SYN_ID, 'tau')

    def test_recovers_with_verification(self):
        env = make_env()
        verifier, llm = make_verifier()
        out = Orchestrator(env, verifier, ev_policy()).run(self.trace, self.payload)
        self.assertEqual(out.status, 'recovered')
        self.assertEqual(len(out.attempts), 1)
        self.assertEqual(env.ledger.reward(out.final_rollout_id), 1.0)
        self.assertEqual([r['purpose'] for r in llm.requests], ['verify', 'verify'])  # original + repair
        self.assertEqual(out.budget['verifier_tokens'], 1400)

    def test_no_success_without_verifier_pass(self):
        env = make_env(fix_with=('reground_tool_output', 'policy_reminder', 'generic_reflect'))
        verifier = FreshContextVerifier(scores(0.2, 0.2, 0.2))
        out = Orchestrator(env, verifier, ev_policy()).run(self.trace, self.payload)
        self.assertNotEqual(out.status, 'recovered')
        self.assertTrue(all(env.ledger.reward(a.rollout_id) == 1.0 for a in out.attempts))
        self.assertEqual(out.status, 'escalated')

    def test_partial_result_with_gaps(self):
        out = Orchestrator(make_env(), FreshContextVerifier(scores(0.2, 0.4, 0.2)), ev_policy()).run(
            self.trace, self.payload)
        self.assertEqual(out.status, 'partial')
        self.assertEqual(out.gaps, ('g',))

    def test_guard_stops_when_repair_worsens(self):
        out = Orchestrator(make_env(), FreshContextVerifier(scores(0.5, 0.1, 0.1)), ev_policy()).run(
            self.trace, self.payload)
        self.assertEqual(out.status, 'aborted')
        self.assertEqual(len(out.attempts), 1)
        self.assertIn('worsened', out.reason)

    def test_budget_cap_before_first_attempt(self):
        verifier, _ = make_verifier()
        out = Orchestrator(make_env(), verifier, ev_policy(), caps=BudgetCaps(max_tokens=3000)).run(
            self.trace, self.payload)
        self.assertEqual((out.status, out.intervened), ('aborted', False))
        self.assertIn('budget', out.reason)

    def test_attempt_cap(self):
        verifier, _ = make_verifier()
        out = Orchestrator(make_env(fix_with=()), verifier, ev_policy(), caps=BudgetCaps(max_attempts=2)).run(
            self.trace, self.payload)
        self.assertEqual(len(out.attempts), 2)
        self.assertEqual(out.status, 'aborted')

    def test_runtime_loop_stops_rollout_without_verifier(self):
        def looping(trace, prefix, directive):
            i = prefix.rollback_step
            steps = []
            for _ in range(5):
                steps += [Step(i, 'assistant', '', (ToolCall('get_order_details', {'order_id': '#W1'}),)),
                          Step(i + 1, 'tool', 'same', name='get_order_details')]
                i += 2
            return steps, 0.0, 1000
        verifier, llm = make_verifier()
        out = Orchestrator(make_env(script=looping), verifier, ev_policy(),
                           caps=BudgetCaps(max_attempts=1)).run(self.trace, self.payload)
        self.assertEqual(out.attempts[0].stopped_early, 'loop')
        self.assertEqual(len(llm.requests), 1)    # only the original was verified

    def test_baseline_retries_then_escalates(self):
        verifier, _ = make_verifier()
        policy = BaselinePolicy(mock_registry(), retries=3, config=SIM, write_tools=WRITE_TOOLS)
        out = Orchestrator(make_env(fix_with=()), verifier, policy, caps=BudgetCaps(max_attempts=5)).run(
            self.trace, self.payload)
        self.assertEqual([a.option['strategy'] for a in out.attempts], ['reground_tool_output'] * 3)
        self.assertEqual(out.status, 'escalated')

    def test_unmodified_control_injects_nothing(self):
        env = make_env(fix_with=())
        verifier, llm = make_verifier()
        out = Orchestrator(env, verifier, UnmodifiedReplayPolicy(ev_policy())).run(self.trace, self.payload)
        self.assertEqual(out.attempts[0].directive_kind, 'none')
        self.assertEqual(env.calls[0]['directive'].message, '')

    def test_learning_repairer_gets_verdicts(self):
        from recovery_orchestrator.replay import TemplateRepairer

        class Learning(TemplateRepairer):
            def __init__(self):
                self.outcomes = []

            def record_outcome(self, trajectory_id, verified, rollout):
                self.outcomes.append((trajectory_id, verified, rollout.rollout_id))

        repairer = Learning()
        verifier, _ = make_verifier()
        env = make_env(fix_with=('policy_reminder',))
        out = Orchestrator(env, verifier, ev_policy(), repairer=repairer).run(self.trace, self.payload)
        self.assertEqual(out.status, 'recovered')
        self.assertEqual([(t, v) for t, v, _ in repairer.outcomes], [(SYN_ID, False), (SYN_ID, True)])
        self.assertEqual([a.applied_strategy for a in out.attempts], ['reground_tool_output', 'policy_reminder'])

        control = Learning()
        Orchestrator(make_env(fix_with=()), verifier, UnmodifiedReplayPolicy(ev_policy()),
                     repairer=control).run(self.trace, self.payload)
        self.assertEqual(control.outcomes, [])      # no repair applied, nothing to learn

    def test_repair_not_applicable_is_skipped_not_replayed(self):
        from recovery_orchestrator.replay import RepairDirective

        class Terminal:
            def build(self, strategy, candidate, trace):
                return RepairDirective('none', strategy.id)

        verifier, llm = make_verifier()
        env = make_env()
        for policy in (ev_policy(), BaselinePolicy(mock_registry(), retries=3, config=SIM, write_tools=WRITE_TOOLS)):
            out = Orchestrator(env, verifier, policy, repairer=Terminal()).run(self.trace, self.payload)
            self.assertEqual((out.status, out.intervened, out.attempts), ('escalated', False, []))
            self.assertIn('no executable plan', out.reason)
        self.assertEqual(env.calls, [])                 # nothing was replayed

    def test_substituted_strategy_is_not_offered_again(self):
        from recovery_orchestrator.replay import RepairDirective

        class Substitute:
            def build(self, strategy, candidate, trace):
                return RepairDirective('inject_guidance', 'generic_reflect', 'think again')

        verifier, _ = make_verifier()
        out = Orchestrator(make_env(fix_with=()), verifier, ev_policy(), repairer=Substitute()).run(
            self.trace, self.payload)
        offered = [(a.option['rollback_step'], a.option['strategy']) for a in out.attempts]
        self.assertNotIn((7, 'generic_reflect'), offered)

    def test_rollout_step_indices_are_unique(self):
        env = make_env()
        verifier, _ = make_verifier()
        Orchestrator(env, verifier, ev_policy(), verify_original=False).run(self.trace, self.payload)
        from recovery_orchestrator.trace import build_prefix
        prefix = build_prefix(self.trace, 7, WRITE_TOOLS)
        rollout = env.rollout(self.trace, prefix, env.calls[0]['directive'], None, 10_000)
        indices = [s.index for s in rollout.steps]
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(rollout.steps[6].injected and rollout.steps[6].index == 7)

    def test_event_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'log.jsonl'
            verifier, _ = make_verifier()
            Orchestrator(make_env(), verifier, ev_policy(), log_path=path).run(self.trace, self.payload)
            record = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(record['status'], 'recovered')
            self.assertEqual([e['event'] for e in record['events']],
                             ['start', 'verify_original', 'decision', 'attempt', 'end'])

    def test_mismatched_payload(self):
        verifier, _ = make_verifier()
        other = load_payload(SYN_RUN, SYN_ID, 'tau')
        other.trajectory_id = 'x'
        with self.assertRaises(ValueError):
            Orchestrator(make_env(), verifier, ev_policy()).run(self.trace, other)


class VerifierTest(unittest.TestCase):
    def test_transcript_hides_injected_guidance_and_system(self):
        steps = syn_trace().steps[:6] + [Step(7, 'system', 'SECRET GUIDANCE', injected=True),
                                         Step(7, 'assistant', 'ok')]
        text = render_transcript(steps)
        self.assertNotIn('SECRET GUIDANCE', text)
        self.assertNotIn('Retail agent policy', text)
        self.assertIn('get_order_details({"order_id": "#W1"})', text)

    def test_parse(self):
        from recovery_orchestrator.verifier import parse_verdict
        self.assertEqual(parse_verdict('```json\n{"verdict":"pass","score":0.8}\n```')[:2], ('PASS', 0.8))
        self.assertEqual(parse_verdict('nonsense')[:2], ('FAIL', 0.0))
        self.assertEqual(parse_verdict('{"verdict":"PASS","score":"x"}')[:2], ('PASS', 0.0))


if __name__ == '__main__':
    unittest.main()
