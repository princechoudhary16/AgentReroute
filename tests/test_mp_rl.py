import copy
import json
import tempfile
import unittest
from pathlib import Path
from agentrx.constraints import check_constraint, guard_matches, validate_constraint
from agentrx.data import normalize
from agentrx.experiments import experiment_variants, validate_experiment
from agentrx.io import digest, read_json
from agentrx.perspectives import (ANY_EVENT, ORDER, PERSPECTIVES, _check_seed_inprocess, _reject_reason, compute_blame_matrix, encode_candidates,
                                  run_perspective_checks, seed_constraints_for_perspective,
                                  synthesize_perspective_constraints)
from agentrx.pipeline import diagnose, diagnose_mp_rl, validate_prediction
from agentrx.prompts import messages
from agentrx.providers import CachedProvider, ProviderError, ReplayProvider
from agentrx.rl_policy import (THETA0, DualLoopRLJudge, PromptMemoryBank, live_dynamic_recover,
                               verify_and_score_state_change)
from agentrx.taxonomy import CATEGORIES
from tests.test_constraints import python_constraint

TOOLS = {'tools': {'cancel_pending_order': {'inputs': {'order_id': 'string', 'reason': 'string'}},
                   'get_order_details': {'inputs': {'order_id': 'string'}}}}
CONTEXT = {'policy': '', 'tools': TOOLS, 'python_timeout': 3}
DELETED = {'repaired_substep': {'role': 'assistant', 'content': 'Order cancelled.', 'tool_calls': []}}
GOOD = {'repaired_substep': {'role': 'assistant', 'content': '', 'tool_calls': [
    {'name': 'cancel_pending_order', 'arguments': {'order_id': '#W1234567', 'reason': 'no longer needed'}}]},
        'repair_rationale': 'pass a JSON object as arguments'}
FRESH = {'next_step': {'index': 999, 'substeps': [
    {'role': 'tool', 'name': 'cancel_pending_order', 'content': '{"status": "cancelled"}'}]}}
STALE = 'Error: invalid arguments'


class FakeProvider:
    identity = {'provider': 'fixture', 'model': 'mp-rl-test'}
    measurement_kind = 'offline_fixture'

    def __init__(self, **stages):
        self.stages, self.requests = stages, []

    def calls(self, stage):
        return [p for s, p in self.requests if s == stage]

    def complete(self, stage, payload):
        self.requests.append((stage, copy.deepcopy(payload)))
        if stage not in self.stages:
            raise ProviderError(f'No fixture response for stage {stage}')
        value = self.stages[stage]
        if callable(value):
            value = value(payload)
        elif isinstance(value, list):
            value = value[min(len(self.calls(stage)), len(value)) - 1]
        return copy.deepcopy(value)


def tau_trace():
    return normalize({'trajectory_id': 'tau-1', 'steps': [
        {'index': 1, 'substeps': [{'role': 'user', 'content': 'Please cancel order #W1234567.'}]},
        {'index': 2, 'substeps': [{'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'c1', 'function': {'name': 'cancel_pending_order', 'arguments': '{bad json'}}]}]},
        {'index': 3, 'substeps': [{'role': 'tool', 'name': 'cancel_pending_order', 'content': STALE}]},
        {'index': 4, 'substeps': [{'role': 'assistant', 'content': 'Your order has been cancelled successfully.'}]},
        {'index': 5, 'substeps': [{'role': 'user', 'content': 'Thanks!'}]}]}, 'tau')


def args_constraint():
    c = python_constraint(["calls = trajectory['steps'][current_step_index]['substeps'][0].get('tool_calls', [])",
                           "return all(isinstance(c.get('arguments'), dict) for c in calls)"],
                          {'tool_name': 'cancel_pending_order'})
    c['assertion_name'] = 'args_are_objects'
    return {**validate_constraint(c), 'origin': 'llm'}


def checked(trajectory, extra=None, provider=None):
    checks, effective = [], {}
    for pid in ORDER:
        rows, effective[pid] = run_perspective_checks(
            pid, (extra or {}).get(pid, []) + seed_constraints_for_perspective(pid, TOOLS), trajectory, provider, CONTEXT)
        checks.extend(rows)
    return checks, effective


def decide(trajectory, extra=None, judge=None):
    checks, effective = checked(trajectory, extra)
    candidates = encode_candidates(trajectory, checks, compute_blame_matrix(trajectory, checks))
    judge = judge or DualLoopRLJudge()
    top1, ranked, probs = judge.select_action(candidates)
    return checks, effective, candidates, top1, ranked, probs, judge


class PerspectiveTests(unittest.TestCase):
    def test_perspective_taxonomy_partition_exhaustive(self):
        groups = [set(p['categories']) for p in PERSPECTIVES.values()]
        self.assertEqual(sum(len(g) for g in groups), 9)
        self.assertEqual(set().union(*groups), set(CATEGORIES[:9]))
        self.assertNotIn('Inconclusive', set().union(*groups))
        self.assertEqual(list(PERSPECTIVES), ['P1_INTENT_PLAN', 'P2_TOOL_EXECUTION', 'P3_HANDOFF_GROUNDING'])

    def test_seed_constraints_prevent_all_skip_collapse(self):
        trajectory = tau_trace()
        for provider in (FakeProvider(perspective_global={'constraints': []}, perspective_dynamic={'constraints': []}),
                         ReplayProvider_empty()):
            checks, floor, seeds = [], set(), 0
            for pid in ORDER:
                constraints, _ = synthesize_perspective_constraints(pid, trajectory, '', TOOLS, provider)
                self.assertEqual({c['origin'] for c in constraints}, {'seed'})
                floor |= {c['assertion_name'] for c in constraints if c['event_trigger'] == ANY_EVENT}
                seeds += len(constraints)
                rows, _ = run_perspective_checks(pid, constraints, trajectory, provider, CONTEXT)
                checks.extend(rows)
            self.assertEqual(len(floor), 3)  # one wildcard floor seed per perspective; specialists may SKIP
            self.assertEqual(len(checks), seeds * len(trajectory['steps']))
            self.assertFalse([r for r in checks if r['verdict'] == 'ERROR' or
                              (r['verdict'] == 'SKIP' and r['assertion_name'] in floor)])
            # Step 2 is the invalid invocation (arguments are not an object) and step 4 claims success after
            # a failed tool. Step 3, the tool error itself, is NOT a violation: a first failure is not yet
            # evidence of anything, and a seed cannot look ahead to see whether the agent recovers from it.
            self.assertEqual(sorted((r['perspective'], r['step_index']) for r in checks if r['verdict'] == 'VIOL'),
                             [('P2_TOOL_EXECUTION', 2), ('P3_HANDOFF_GROUNDING', 4)])
            matrix = compute_blame_matrix(trajectory, checks)
            self.assertGreater(matrix['P2_TOOL_EXECUTION']['assistant'], 0)
            self.assertGreater(matrix['P3_HANDOFF_GROUNDING']['assistant'], 0)

    def test_auto_guard_relaxation_on_100_percent_skip(self):
        trajectory = tau_trace()
        narrow = {**python_constraint(['return False'], {'step_index': 99, 'content_regex': 'NEVER_MATCHES_ANYTHING'}),
                  'origin': 'llm'}
        self.assertFalse(any(guard_matches(validate_constraint(narrow), s) for s in trajectory['steps']))
        rows, effective = run_perspective_checks('P2_TOOL_EXECUTION', [narrow], trajectory, None, CONTEXT)
        self.assertEqual([r['verdict'] for r in rows], ['VIOL'] * 5)
        self.assertTrue(all(r['relaxed'] for r in rows))
        self.assertEqual(effective[0]['event_trigger']['step_index'], '*')
        self.assertEqual(effective[0]['event_trigger']['content_regex'], '*')
        self.assertEqual(narrow['event_trigger']['step_index'], 99)  # Caller's constraint is not mutated.

    def test_joint_agent_step_candidates_on_single_agent_tau_trace(self):
        *_, candidates, top1, ranked, probs, judge = decide(tau_trace())
        # The tool-output VIOL at step 3 is folded into its caller's step 2, never a candidate of its own.
        self.assertEqual([(c['agent'], c['step_index']) for c in candidates], [('assistant', 2), ('assistant', 4)])
        self.assertEqual([v['step_index'] for v in candidates[0]['violated_checks']], [2])
        self.assertEqual([r['step_index'] for r in candidates[0]['evidence_refs']], [2])
        self.assertTrue(all(len(c['features']) == 10 and c['offset'] == c['actor_offset'] for c in candidates))
        self.assertEqual((top1['agent'], top1['step_index'], top1['dominant_category']),
                         ('assistant', 2, 'InvalidInvocation'))
        # P2 blame for 'assistant' = V/(V+S+eps), a pure violation rate with no earliness factor. The
        # agent owns the P2 rows at steps 2, 3 and 4 (the tool output at 3 folds into its caller), of
        # which one is VIOL and two SAT. Position is expressed once, in feature 7, not here as well.
        self.assertAlmostEqual(candidates[0]['features'][1], 1 / (3 + 1e-6), places=6)
        # density = min(1, violations_here / 2) per perspective, then python-check share, 1 - offset/T,
        # perspective breadth, max density. One P2 violation here, and it is a python check, so the two
        # halved terms are 0.5 rather than the 1.0 that two violations produced.
        self.assertEqual(candidates[0]['features'][3:], [0.0, 0.5, 0.0, 0.5, 0.8, 1 / 3, 0.5])

    def test_blame_matrix_3xN_multi_agent_magentic_trace(self):
        trajectory = normalize({'trajectory_id': 'mag-1', 'steps': [
            {'index': 1, 'substeps': [{'role': 'human', 'content': 'Find the paper.'}]},
            {'index': 2, 'substeps': [{'role': 'Orchestrator (thought)', 'content': 'Ask WebSurfer to search arXiv.'}]},
            {'index': 3, 'substeps': [{'role': 'WebSurfer', 'content': 'The paper ID is 2103.07786.'}]},
            {'index': 4, 'substeps': [{'role': 'Coder', 'content': ''}]},
            {'index': 5, 'substeps': [{'role': 'Orchestrator (thought)', 'content': 'Ask WebSurfer to search arXiv.'}]}]},
            'magentic')
        invented = {**python_constraint(['return False'], {'role_name': 'WebSurfer'}), 'origin': 'llm',
                    'taxonomy_targets': ['InventionOfNewInformation']}
        checks, _ = checked(trajectory, {'P3_HANDOFF_GROUNDING': [invented]})
        matrix = compute_blame_matrix(trajectory, checks)
        self.assertEqual(list(matrix), list(ORDER))
        for row in matrix.values():
            self.assertEqual(set(row), {'orchestrator', 'websurfer', 'coder'})
        # Pure rates: coder is 1 VIOL and 0 SAT, orchestrator and websurfer are 1 VIOL and 1 SAT each.
        self.assertAlmostEqual(matrix['P1_INTENT_PLAN']['coder'], 1 / (1 + 1e-6), places=6)
        self.assertAlmostEqual(matrix['P1_INTENT_PLAN']['orchestrator'], 1 / (2 + 1e-6), places=6)
        self.assertAlmostEqual(matrix['P3_HANDOFF_GROUNDING']['websurfer'], 1 / (2 + 1e-6), places=6)
        # Orchestrator violates at step 5 and websurfer at step 3, and they now score identically:
        # blame answers 'how often is this agent in violation', never 'how early'. Position used to
        # separate these two by a factor of 3 (0.2 against 0.6) before anything looked at the steps.
        self.assertEqual(set(matrix['P2_TOOL_EXECUTION'].values()), {0.0})
        candidates = encode_candidates(trajectory, checks, matrix)
        self.assertEqual({(c['agent'], c['step_index'], c['dominant_category']) for c in candidates}, {
            ('websurfer', 3, 'InventionOfNewInformation'), ('coder', 4, 'Instruction/PlanAdherenceFailure'),
            ('orchestrator', 5, 'Instruction/PlanAdherenceFailure')})


MAS_TOOLS = {'tools': {n: {} for n in ('find_user_id_by_email', 'get_order_details', 'exchange_delivered_order_items')}}
LOOKUP = {'id': 'c', 'name': 'get_order_details', 'arguments': {'order_id': '#W1000003'}}


def retail_mas_trace():
    """4-agent retail shapes from benchmark/corpus: dropped handoff ids, refusal, stuck tool loop, unbacked write/amount."""
    def say(role, content, **extra):
        return {'substeps': [{'role': role, 'content': content, **extra}]}
    steps = [say('user', 'My email is a@b.c. Exchange the T-shirt in order #W1000003.'),
             say('planner', '', tool_calls=[{'id': 'a', 'name': 'find_user_id_by_email', 'arguments': {'email': 'a@b.c'}}]),
             say('tool', 'noah_brown_6181', name='find_user_id_by_email'),
             {'substeps': [{'role': 'planner', 'content': '{"action": "handoff", "to": "returns_agent"}'},
                           {'role': 'planner', 'name': 'handoff', 'content': json.dumps(
                               {'handoff_to': 'returns_agent', 'message': 'Exchange it.', 'context': {'action': 'exchange'}})}]},
             say('returns_agent', 'The handoff did not include an order id, so I cannot proceed.'),
             say('returns_agent', '', tool_calls=[LOOKUP]),
             say('tool', '{"status": "processing"}', name='get_order_details'),
             say('returns_agent', '', tool_calls=[LOOKUP]),
             say('tool', '{"status": "processing"}', name='get_order_details'),
             say('returns_agent', '', tool_calls=[LOOKUP]),
             say('tool', '{"order_id": "#W1000003", "items": [{"price": 50.0}], "total": 50.0}', name='get_order_details'),
             say('returns_agent', 'Your T-shirt has been exchanged; you owe $45.00.')]
    return normalize({'trajectory_id': 'retail-mas', 'steps': [{'index': i, **s} for i, s in enumerate(steps, 1)]},
                     'tau')


class RetailSeedTests(unittest.TestCase):
    def test_seeds_fire_on_retail_mas_fault_shapes(self):
        trajectory = retail_mas_trace()
        context = {**CONTEXT, 'tools': MAS_TOOLS}
        checks = []
        for pid in ORDER:
            checks += run_perspective_checks(pid, seed_constraints_for_perspective(pid, MAS_TOOLS), trajectory,
                                             None, context)[0]
        self.assertFalse([r for r in checks if r['verdict'] == 'ERROR'])
        self.assertEqual(sorted((r['assertion_name'], r['step_index']) for r in checks if r['verdict'] == 'VIOL'), [
            ('p1_seed_nonempty_progress', 12),           # concludes an exchange handoff with no exchange_ call
            ('p1_seed_specialist_routing_alignment', 5),            # "cannot proceed"
            ('p1_seed_specialist_routing_alignment', 10),           # 3rd identical get_order_details call
            # Step 9 repeats step 7's identical (tool, arguments) call for an identical result, which is
            # what a stuck tool looks like from the trace. Step 7 is the first such result and carries no
            # prefix evidence at all: flagging it required knowing the injector's own payload string.
            ('p2_seed_valid_tool_invocation', 9),
            ('p3_seed_handoff_and_fact_grounding', 4),              # handoff context drops #W1000003
            ('p3_seed_handoff_and_fact_grounding', 12)])            # $45.00 never observed
        matrix = compute_blame_matrix(trajectory, checks)
        self.assertTrue(all(matrix[pid]['returnsagent'] > 0 for pid in ORDER))   # all 3 lenses blame the specialist

    def test_inprocess_seed_rows_match_the_sandboxed_worker(self):
        trajectory = retail_mas_trace()
        context = {**CONTEXT, 'tools': MAS_TOOLS}
        broken = [{**validate_constraint({**python_constraint(body), 'assertion_name': name,
                                          'python_check': {'function_name': name, 'code_lines': [
                                              f'def {name}(trajectory, current_step_index):', f'    {body[0]}']}}),
                   'origin': 'seed'} for name, body in (('raises', ['return missing_name']), ('not_bool', ['return 1']))]
        seeds = [c for pid in ORDER for c in seed_constraints_for_perspective(pid, MAS_TOOLS)] + broken
        for c in seeds:
            for offset in range(len(trajectory['steps'])):
                self.assertEqual(_check_seed_inprocess(c, trajectory, offset),
                                 check_constraint(c, trajectory, offset, None, context), (c['assertion_name'], offset))


def handoff_trace(target, message, context, after):
    """user -> planner handoff (step 2) -> specialist/planner replies (steps 3..)."""
    steps = [{'substeps': [{'role': 'user', 'content': 'Return the headphones from delivered order #W1000002.'}]},
             {'substeps': [{'role': 'planner', 'content': '{"action": "handoff"}'},
                           {'role': 'planner', 'name': 'handoff', 'content': json.dumps(
                               {'handoff_to': target, 'message': message, 'context': context})}]}]
    steps += [{'substeps': [{'role': role, 'content': text}]} for role, text in after]
    return normalize({'trajectory_id': 'handoff', 'steps': [{'index': i, **x} for i, x in enumerate(steps, 1)]}, 'tau')


def viol(trajectory, tools=MAS_TOOLS):
    checks = []
    for pid in ORDER:
        checks += run_perspective_checks(pid, seed_constraints_for_perspective(pid, tools), trajectory, None,
                                         {**CONTEXT, 'tools': tools})[0]
    assert not [r for r in checks if r['verdict'] == 'ERROR'], checks
    return sorted((r['assertion_name'], r['step_index']) for r in checks if r['verdict'] == 'VIOL')


class HandoffSeedTests(unittest.TestCase):
    def test_misroute_is_flagged_at_the_planner_handoff_not_its_relay(self):
        ids = {'order_id': '#W1000002', 'user_id': 'mia_garcia_4516'}
        refusal = 'I cannot handle this request.'
        found = viol(handoff_trace('product_agent', 'Return items for order #W1000002.', ids,
                                   [('product_agent', refusal), ('planner', refusal)]))
        self.assertEqual(found, [('p1_seed_specialist_routing_alignment', 2),      # order work sent to product_agent
                                 ('p1_seed_specialist_routing_alignment', 3)])     # the specialist's refusal; relay at 4 is not
        question = viol(handoff_trace('order_agent', 'Ask how many T-shirt variants are available.',
                                      {'user_id': 'ava_lee_2203'}, []))
        self.assertEqual(question, [('p1_seed_specialist_routing_alignment', 2)])
        routed = viol(handoff_trace('returns_agent', 'Return items for order #W1000002.', ids, []))
        self.assertEqual(routed, [])                                               # correct routing stays clean

    def test_dropped_handoff_context_is_flagged(self):
        for message, context in (('Return items for order [omitted].', {'action': 'return'}),
                                 ('Return items.', {})):
            self.assertEqual(viol(handoff_trace('returns_agent', message, context, [])),
                             [('p3_seed_handoff_and_fact_grounding', 2)])

    def test_write_claims_need_a_write_since_the_latest_handoff(self):
        write = {'id': 'w', 'name': 'exchange_delivered_order_items', 'arguments': {'order_id': '#W1000002'}}
        steps = [{'substeps': [{'role': 'user', 'content': 'Exchange then return, order #W1000002.'}]},
                 {'substeps': [{'role': 'returns_agent', 'content': '', 'tool_calls': [write]}]},
                 {'substeps': [{'role': 'planner', 'name': 'handoff', 'content': json.dumps(
                     {'handoff_to': 'returns_agent', 'message': 'Now the return for #W1000002.',
                      'context': {'order_id': '#W1000002'}})}]},
                 {'substeps': [{'role': 'returns_agent', 'content': 'Return submitted for order #W1000002.'}]},
                 {'substeps': [{'role': 'planner', 'content': 'The return for order #W1000002 has been requested.'}]}]
        trajectory = normalize({'trajectory_id': 'w', 'steps': [{'index': i, **x} for i, x in enumerate(steps, 1)]},
                               'tau')
        tools = {'tools': {'exchange_delivered_order_items': {}}}
        # The earlier exchange does not excuse the specialist's later claim; the planner relay is excused.
        self.assertEqual([v for v in viol(trajectory, tools) if v[0] == 'p1_seed_nonempty_progress'],
                         [('p1_seed_nonempty_progress', 4)])


class LiveLLMRegressionTests(unittest.TestCase):
    """Failures seen live with qwen2.5-coder:7b on tau trace 2."""

    def test_never_blames_system_or_user_steps(self):
        trajectory = normalize({'trajectory_id': 'sys-1', 'steps': [
            {'index': 1, 'substeps': [{'role': 'system', 'content': '# Retail agent policy'}]},
            {'index': 2, 'substeps': [{'role': 'user', 'content': 'Cancel my order.'}]},
            {'index': 3, 'substeps': [{'role': 'assistant', 'content': 'Which order?'}]}]}, 'tau')
        always = {**python_constraint(['return False']), 'origin': 'llm', 'taxonomy_targets': ['IntentNotSupported']}
        checks, _ = checked(trajectory, {'P1_INTENT_PLAN': [always]})
        self.assertEqual({r['step_index'] for r in checks if r['verdict'] == 'VIOL'}, {1, 2, 3})
        matrix = compute_blame_matrix(trajectory, checks)
        self.assertTrue(all(set(row) == {'assistant'} for row in matrix.values()))
        candidates = encode_candidates(trajectory, checks, matrix)
        self.assertEqual([(c['agent'], c['step_index']) for c in candidates], [('assistant', 3)])
        self.assertEqual({v['step_index'] for v in candidates[0]['violated_checks']}, {3})
        # Violations only on passive steps: fall back to the agent's own actions, still never system/user.
        system_only = [r for r in checks if r['step_index'] != 3 or r['verdict'] != 'VIOL']
        fallback = encode_candidates(trajectory, system_only, compute_blame_matrix(trajectory, system_only))
        self.assertEqual([(c['agent'], c['step_index'], c['violated_checks']) for c in fallback], [('assistant', 3, [])])

    def test_relaxes_whole_guard_when_role_name_never_matches(self):
        trajectory = tau_trace()
        wrong_role = {**python_constraint(['return False'], {'role_name': 'retail_agent', 'tool_name': 'cancel_.*'}),
                      'origin': 'llm'}
        rows, effective = run_perspective_checks('P1_INTENT_PLAN', [wrong_role], trajectory, None, CONTEXT)
        self.assertEqual(effective[0]['event_trigger'], ANY_EVENT)
        self.assertEqual([r['verdict'] for r in rows], ['VIOL'] * 5)
        provider = FakeProvider(perspective_global={'constraints': []}, perspective_dynamic={'constraints': []})
        synthesize_perspective_constraints('P1_INTENT_PLAN', trajectory, '', TOOLS, provider)
        self.assertEqual(provider.calls('perspective_global')[0]['observed_agent_roles'], ['assistant'])
        prompt =messages('perspective_dynamic', provider.calls('perspective_dynamic')[0])[0]['content']
        self.assertIn('Observed autonomous agent roles in trajectory: ["assistant"]', prompt)
        self.assertIn('never trajectory["tools"]', prompt)

    def test_reject_reason_catches_placeholders_and_runtime_dry_run_errors(self):
        trajectory = tau_trace()
        placeholder = {**python_constraint(['return True']), 'assertion_name': 'unique_snake_case'}
        self.assertIn('unique_snake_case', _reject_reason(placeholder, None, trajectory)[1])
        rubric = {'assertion_name': 'grounded_reply', 'taxonomy_targets': ['InventionOfNewInformation'],
                  'check_hint': 'reply is grounded', 'check_type': 'nl_check',
                  'nl_check': {'judge_rubric': ['Objective Criterion']}}
        self.assertIn('objective criterion', _reject_reason(rubric, None, trajectory)[1])
        wrong_keys = python_constraint(["step = trajectory['steps'][current_step_index]",
                                        "return step['tool_calls'][0]['tool_name'] != ''"])
        constraint, reason = _reject_reason(wrong_keys, None, trajectory)
        self.assertIsNone(constraint)
        self.assertIn('runtime dry-run', reason)
        self.assertIn('KeyError', reason)
        self.assertIn("step['substeps'][i]['tool_calls'][j]['name']", reason)
        self.assertIsNone(_reject_reason(wrong_keys, None)[1])  # no trajectory: compile-only, as before
        # Passes on step offset 0 (no tool calls) but must also survive the first tool-call step.
        tool_only = python_constraint(["event = trajectory['steps'][current_step_index]['substeps'][0]",
                                       "return not event.get('tool_calls') or event['tool_calls'][0]['args'] is not None"])
        self.assertIn("KeyError: 'args'", _reject_reason(tool_only, None, trajectory)[1])
        self.assertEqual(_reject_reason(args_constraint(), None, trajectory)[1], None)


def ReplayProvider_empty():
    directory = tempfile.mkdtemp()
    path = Path(directory)/'replay.json'
    path.write_text('{}', encoding='utf-8')
    return ReplayProvider(path)


class RLJudgeTests(unittest.TestCase):
    def test_contextual_rl_judge_top1_and_ranked_list_extrapolation(self):
        trajectory, judge = tau_trace(), DualLoopRLJudge()
        with tempfile.TemporaryDirectory() as d:
            result = diagnose_mp_rl(trajectory, '', TOOLS, {'protocol': 'mp_rl', 'evidence': False, 'recovery_loop': False},
                                    FakeProvider(), d, rl_judge=judge)
            rca = read_json(Path(d)/'rca_payload.json')
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(validate_prediction(result['prediction'], trajectory), result['prediction'])
        ranked = result['ranked_candidates']
        probabilities = [c['probability'] for c in ranked]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertEqual((ranked[0]['step_index'], ranked[0]['category']),
                         (result['prediction']['step_index'], result['prediction']['category']))
        self.assertEqual([c['step_index'] for c in rca['top_k']], [c['step_index'] for c in ranked[:3]])
        self.assertEqual(rca['confidence'], ranked[0]['probability'])
        self.assertEqual(rca['ranked_probabilities'][-1]['step_index'], ranked[-1]['step_index'])
        # Without the local loop, the decision waits for a cross-branch verifier outcome (Stage B hook).
        before = list(judge.weights)
        telemetry = judge.record_recovery_outcome('tau-1', 2, 'assistant', accepted=True, verifier_score=0.9)
        self.assertAlmostEqual(telemetry['reward'], 0.8)
        self.assertNotEqual(judge.weights, before)
        with self.assertRaises(KeyError):
            judge.record_recovery_outcome('tau-1', 2, 'assistant', accepted=True)

    def test_layer1_anti_repeat_fingerprint_and_compiler_retry(self):
        trajectory, memory = tau_trace(), PromptMemoryBank()
        *_, top1, _, _, _ = decide(trajectory)
        inner = FakeProvider(perspective_repair=[DELETED, DELETED, GOOD], tool_sandbox=FRESH)
        with tempfile.TemporaryDirectory() as d:
            recovery = live_dynamic_recover(trajectory, top1, '', TOOLS, CachedProvider(inner, d, {'t': 1}), memory)
            self.assertEqual(len(list(Path(d).glob('*.json'))), 4)  # each retry is a distinct cached request
        repairs = inner.calls('perspective_repair')
        self.assertEqual([p['rung'] for p in repairs], [1, 1, 2])  # duplicate on attempt 1 escalates to Rung 2
        self.assertNotIn('retry_feedback', repairs[0])
        self.assertIn('Action-Structure Invariant', repairs[1]['retry_feedback'][0]['reason'])
        self.assertIn('FORBIDDEN OUTPUT (DO NOT REPEAT)', repairs[1]['retry_feedback'][0]['note'])
        self.assertIn('Duplicate', repairs[2]['retry_feedback'][1]['reason'])
        self.assertEqual([a['rejected'] is None for a in recovery['attempts']], [False, False, True])
        self.assertTrue(recovery['layer1_passed'])
        self.assertIn(digest(DELETED['repaired_substep']), memory.forbidden)
        self.assertEqual(recovery['seen_hashes'], [digest(DELETED['repaired_substep'])])
        self.assertNotIn('repair_rationale', recovery['repaired_substep'])
        # Synthesis: a constraint that fails to compile is banned and its error is fed back on retry.
        broken = python_constraint(['return ('])
        valid = args_constraint()
        valid.pop('origin')
        provider = FakeProvider(perspective_global=[{'constraints': [broken]}, {'constraints': [valid]}],
                                perspective_dynamic={'constraints': []})
        constraints, log = synthesize_perspective_constraints('P2_TOOL_EXECUTION', trajectory, '', TOOLS, provider, memory)
        retry = provider.calls('perspective_global')[1]['retry_feedback'][0]
        self.assertIn('SyntaxError', retry['reason'])
        self.assertIn('FORBIDDEN OUTPUT (DO NOT REPEAT)', retry['note'])
        self.assertEqual([c['assertion_name'] for c in constraints], ['args_are_objects', 'p2_seed_valid_tool_invocation'])
        self.assertIn(digest(broken), memory.forbidden)
        # Persistent negative memory: the next episode pre-injects the pattern and rejects it without compiling.
        again = FakeProvider(perspective_global={'constraints': [broken]}, perspective_dynamic={'constraints': []})
        synthesize_perspective_constraints('P2_TOOL_EXECUTION', trajectory, '', TOOLS, again, memory, max_retries=0)
        self.assertTrue(again.calls('perspective_global')[0]['forbidden_patterns'])
        self.assertEqual(len(again.calls('perspective_global')), 1)

    def test_live_dynamic_recovery_no_stale_response_reuse(self):
        trajectory = tau_trace()
        *_, top1, _, _, _ = decide(trajectory)
        provider = FakeProvider(perspective_repair=GOOD, tool_sandbox=FRESH)
        recovery = live_dynamic_recover(trajectory, top1, '', TOOLS, provider)
        self.assertEqual(recovery['next_step_source'], 'tool_sandbox')
        self.assertEqual(recovery['next_step']['index'], 3)
        self.assertEqual(recovery['next_step']['substeps'][0]['content'], '{"status": "cancelled"}')
        self.assertEqual([s['index'] for s in recovery['repaired_trajectory']['steps']], [1, 2, 3])
        self.assertNotIn(STALE, json.dumps(recovery['repaired_trajectory']))
        sandbox = provider.calls('tool_sandbox')[0]
        self.assertEqual(sandbox['trajectory']['steps'][-1], recovery['repaired_step'])
        self.assertNotIn(STALE, json.dumps(sandbox))
        # Offline replay without repair/sandbox fixtures degrades gracefully and still never reuses the stale step.
        offline = live_dynamic_recover(trajectory, top1, '', TOOLS, ReplayProvider_empty())
        self.assertEqual((offline['mode'], offline['next_step_source']), ('fallback', 'unavailable'))
        self.assertEqual(offline['repaired_substep']['tool_calls'][0]['arguments'], {})
        self.assertNotIn(STALE, json.dumps(offline['repaired_trajectory']))

    def test_split_role_verification_rejects_reward_hacking(self):
        trajectory = tau_trace()
        checks, effective, candidates, top1, _, probs, judge = decide(trajectory)
        recovery = live_dynamic_recover(trajectory, top1, '', TOOLS, FakeProvider(perspective_repair=DELETED,
                                                                                  tool_sandbox=FRESH), judge.memory)
        self.assertFalse(recovery['layer1_passed'])
        self.assertEqual(len(recovery['attempts']), 3)
        before = judge.select_action(candidates)[2][top1['candidate_index']]
        score = verify_and_score_state_change(trajectory, checks, effective, recovery, top1, candidates, probs,
                                              FakeProvider(), CONTEXT, judge)
        self.assertEqual((score['feedback'], score['reward']), ('-', -1.0))
        self.assertLess(judge.select_action(candidates)[2][top1['candidate_index']], before)

    def test_dual_loop_rl_updates_weights_and_prompt_memory(self):
        trajectory = tau_trace()
        llm = args_constraint()
        checks, effective, candidates, top1, _, probs, judge = decide(trajectory, {'P2_TOOL_EXECUTION': [llm]})
        self.assertEqual(top1['step_index'], 2)
        recovery = live_dynamic_recover(trajectory, top1, '', TOOLS, FakeProvider(perspective_repair=GOOD,
                                                                                  tool_sandbox=FRESH), judge.memory)
        score = verify_and_score_state_change(trajectory, checks, effective, recovery, top1, candidates, probs,
                                              FakeProvider(), CONTEXT, judge)
        self.assertEqual((score['v_before'], score['v_after'], score['feedback'], score['reward']), (2, 0, '+', 1.0))
        self.assertTrue(score['layer3']['passed'])
        delta = [w - w0 for w, w0 in zip(judge.weights, THETA0)]
        gradient = score['policy_update']['gradient']
        self.assertGreater(sum(d * g for d, g in zip(delta, gradient)), 0)
        self.assertGreater(judge.select_action(candidates)[2][top1['candidate_index']], probs[top1['candidate_index']])
        exemplar = {k: v for k, v in llm.items() if k != 'origin'}
        self.assertEqual(judge.memory.retrieve_exemplars('P2_TOOL_EXECUTION'), [exemplar])
        with tempfile.TemporaryDirectory() as d:
            judge.save(Path(d)/'rl_state.json')
            self.assertEqual(DualLoopRLJudge.load(Path(d)/'rl_state.json').to_dict(), judge.to_dict())

    def test_constraints_used_is_populated_only_when_synthesis_runs(self):
        """What actually left the Verbal Loop with nothing to learn from.

        The handoff plan reported PromptMemoryBank as silently dead because nothing ever populated
        ``decision['constraints_used']``. It is populated, in pipeline.diagnose_mp_rl immediately after
        select_action, filtered to ``origin == 'llm'``. The list was empty throughout Path B and Path C
        training for a different reason: those runs passed ``evidence=False``, so only seed constraints
        existed and the origin filter matched none of them. Turning synthesis on is the fix; more wiring
        would have been dead code. This pins both halves so the diagnosis cannot be re-misread.
        """
        trajectory = tau_trace()
        dynamic = lambda p: {'constraints': [args_constraint()] if p['perspective'] == 'P2_TOOL_EXECUTION' else []}
        base = {'protocol': 'mp_rl', 'checklist': True, 'recovery_loop': False}
        for evidence in (True, False):
            judge = DualLoopRLJudge()
            provider = (FakeProvider(perspective_global={'constraints': []}, perspective_dynamic=dynamic)
                        if evidence else FakeProvider())
            with tempfile.TemporaryDirectory() as d:
                diagnose_mp_rl(trajectory, '', TOOLS, {**base, 'evidence': evidence}, provider, d, rl_judge=judge)
            used = judge.pending[str(trajectory['trajectory_id'])]['constraints_used']
            self.assertEqual(bool(used), evidence, (evidence, used))
            self.assertTrue(all(c['origin'] == 'llm' for c in used))

    def test_end_to_end_diagnose_mp_rl_pipeline(self):
        trajectory = tau_trace()
        dynamic = lambda p: {'constraints': [args_constraint()] if p['perspective'] == 'P2_TOOL_EXECUTION' else []}
        provider = FakeProvider(perspective_global={'constraints': []}, perspective_dynamic=dynamic,
                                perspective_repair=GOOD, tool_sandbox=FRESH)
        with tempfile.TemporaryDirectory() as d:
            state = Path(d)/'rl'/'state.json'
            result = diagnose(trajectory, '', TOOLS, {'protocol': 'mp_rl', 'rl_state_path': str(state)}, provider, Path(d)/'job')
            job = Path(d)/'job'
            for name in ('trajectory_ir', 'checks', 'validation_log', 'perspectives', 'blame_matrix',
                         'rl_recovery', 'rca_payload', 'result'):
                self.assertTrue((job/f'{name}.json').exists(), name)
            self.assertEqual(read_json(job/'result.json')['prediction'], result['prediction'])
            rca, recovery = read_json(job/'rca_payload.json'), read_json(job/'rl_recovery.json')
            saved = read_json(state)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(validate_prediction(result['prediction'], trajectory), result['prediction'])
        self.assertEqual((result['prediction']['step_index'], result['prediction']['category']), (2, 'InvalidInvocation'))
        self.assertEqual(result['recovery'], {'status': 'ok', 'mode': 'live', 'feedback': '+', 'reward': 1.0})
        self.assertEqual(recovery['verification']['reward'], 1.0)
        self.assertEqual(rca['violations_by_step']['2']['count'], 2)
        self.assertEqual((rca['inconclusive_share'], rca['n_runs'], rca['top_k'][0]['source']), (0.0, 1, 'fa'))
        self.assertNotEqual(saved['weights'], list(THETA0))
        self.assertTrue(saved['memory']['exemplars'])
        # Offline fixture replay never crashes: seeds + structural fallback.
        with tempfile.TemporaryDirectory() as d:
            offline = diagnose(trajectory, '', TOOLS, {'protocol': 'mp_rl'}, ReplayProvider_empty(), d)
        self.assertEqual((offline['status'], offline['recovery']['mode']), ('ok', 'fallback'))
        # Variant plumbing: the new fields pass experiment validation, including the shipped Ollama config.
        variant = experiment_variants({'variants': [{'name': 'mp_rl', 'protocol': 'mp_rl', 'recovery_loop': False,
                                                     'rl_state_path': 'x.json'}]})[0]
        self.assertEqual((variant['protocol'], variant['recovery_loop']), ('mp_rl', False))
        shipped = validate_experiment(read_json('configs/dev_ollama.json'))
        self.assertIn('mp_rl', [v['protocol'] for v in shipped['variants']])
