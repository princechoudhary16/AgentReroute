"""The mp_rl seed checks must not recognise faults by the fault injector's own wording.

History. ``WRITE_CLAIM`` and ``STUCK_TOOL`` in ``agentrx/perspectives.py`` were regex copies of the
``CLAIMS`` templates and the ``PROCESSING`` payload that ``benchmark/faults/operators.py`` injects: 14
claim branches plus the stuck-tool payload, 15 couplings in all. On our own benchmark those seeds fired
with near-perfect precision; on 73 external traces they produced no evidence at all for 7 of 29 tau and
29 of 44 Magentic trajectories. The original version of this test asserted the coupling was gone and was
marked ``expectedFailure`` to keep the debt visible.

Both regexes are now deleted. A skipped write is detected structurally (a request for a state change
concluded with no state-changing call) and a stuck tool structurally (an identical call returning an
identical result), so neither depends on the injector's phrasing. This test holds that line.

It is deliberately FAIL-CLOSED. Rather than naming the two old constants, it enumerates every regex
constant the module defines and checks it against the injector's rendered content, with an explicit
allowlist for the patterns that are generic by design. A new constant is therefore checked by default:
adding one that matches the injector's wording fails this test until someone allowlists it on purpose
and writes down why.
"""
import re
import unittest

from agentrx import perspectives
from agentrx.perspectives import ORDER, domain_packs, seed_constraints_for_perspective
from benchmark.faults.operators import CLAIMS, PROCESSING, _Fields

# Any concrete argument values: the coupling would be in the templates' fixed phrasing, not the fields.
_FIELDS = _Fields({'order_id': '#W1234567', 'reason': 'no longer needed', 'item_ids': 'A1',
                   'new_item_ids': 'B2', 'payment_method_id': 'paypal_1', 'address1': '1 Main St',
                   'city': 'Springfield', 'user_id': 'user_1'})

# Patterns that are generic by construction, each with the reason it may legitimately match injected text.
GENERIC = {
    'NUMBER': 'bare number shape, matched against the observed prefix to ground an amount',
    'DOLLARS': 'currency shape, same grounding rule',
    'ID_TOKEN': 'identifier shape; a claim citing an order id is SUPPOSED to match, that is the rule',
    'MONEY_FIELD': 'JSON money field in a tool output, never agent prose',
    'ERROR_MARKERS': 'generic error vocabulary applied to tool output (checked against PROCESSING below)',
    'DRY_RUN_HINT': 'instruction prose shown to the model, not a matcher',
    'HANDOFF_ID': 'retail pack; applied to the ids seen in the prefix, not to a completion claim',
    'PRODUCT_MISROUTE': 'retail pack; applied to a handoff target and message',
    'ORDER_ID_FIELD': 'retail pack; applied to the observed prefix',
    'RETURNS_WORK': 'retail pack; applied to a handoff message',
    'ORDER_WORK': 'retail pack; applied to a handoff message',
    'PRODUCT_QUESTION': 'retail pack; applied to a handoff message',
}

RETAIL_IDENTIFIERS = perspectives.RETAIL_MARKERS + ('order_agent', 'returns_agent', 'product_agent', '#W')
NON_RETAIL_TOOLS = {'tools': {'update_record': {'inputs': {'record_id': 'string'}},
                              'get_record': {'inputs': {'record_id': 'string'}}}}


def claim_texts():
    return [t.format_map(_FIELDS) for templates in CLAIMS.values() for t in templates]


def regex_constants():
    return {n: v for n, v in vars(perspectives).items()
            if n.isupper() and isinstance(v, str) and not n.startswith('_')}


def seed_source(tools):
    return '\n'.join(line for pid in ORDER for c in seed_constraints_for_perspective(pid, tools)
                     for line in c['python_check']['code_lines'])


class SeedInjectorLeakageTest(unittest.TestCase):
    def test_allowlist_only_names_constants_that_exist(self):
        """A renamed constant must not silently fall out of coverage into an allowlist entry."""
        self.assertEqual(sorted(set(GENERIC) - set(regex_constants())), [])

    def test_no_verdict_pattern_matches_the_injectors_claim_templates(self):
        texts = claim_texts()
        self.assertEqual(len(texts), 14)                       # the 14 branches the old WRITE_CLAIM copied
        shared = []
        for name, pattern in sorted(regex_constants().items()):
            if name in GENERIC:
                continue
            for text in texts:
                if re.search(pattern, text, re.I):
                    shared.append((name, text))
        self.assertEqual(shared, [], f'seed patterns match injected claims: {shared}')

    def test_no_pattern_matches_the_stuck_tool_payload(self):
        shared = [n for n, p in sorted(regex_constants().items())
                  if n not in GENERIC and re.search(p, PROCESSING, re.I)]
        self.assertEqual(shared, [], f'seed patterns match the injected stuck-tool payload: {shared}')
        # ERROR_MARKERS is allowlisted against claims but must not match HERE: a stuck tool is not an
        # error, and p2's persistent-error clause would otherwise key on the injected payload's wording.
        self.assertIsNone(re.search(perspectives.ERROR_MARKERS, PROCESSING, re.I))

    def test_a_refusal_phrase_never_appears_in_an_injected_claim(self):
        """REFUSAL excuses a missing write, so a claim matching it would hide the very fault injected."""
        for text in claim_texts():
            self.assertIsNone(re.search(perspectives.REFUSAL, text, re.I), text)

    def test_core_seeds_name_no_retail_vocabulary(self):
        """The general/domain split has to be real: with no retail tools, no retail identifier is emitted."""
        self.assertEqual(domain_packs(NON_RETAIL_TOOLS), ())
        source = seed_source(NON_RETAIL_TOOLS)
        found = sorted({word for word in RETAIL_IDENTIFIERS if word in source})
        self.assertEqual(found, [], f'core seed source carries retail vocabulary: {found}')

    def test_retail_pack_activates_only_on_retail_tool_vocabulary(self):
        self.assertEqual(domain_packs({'tools': {'cancel_pending_order': {}}}), ('retail',))
        # The AgentRx Magentic payload shape: dict values that are not tool schemas.
        self.assertEqual(domain_packs({'protocol': {'event_types': []}, 'agents': {'Orchestrator': {}}}), ())
        self.assertIn('order_agent', seed_source({'tools': {'cancel_pending_order': {}}}))

    def test_a_skipped_write_is_still_detected_after_the_rewording(self):
        """The point of the exercise: the fault stays visible without the phrase that used to reveal it."""
        from agentrx.data import normalize
        from agentrx.perspectives import run_perspective_checks
        import json as _json
        tools = {'tools': {'cancel_pending_order': {'inputs': {'order_id': 'string', 'reason': 'string'}},
                           'get_order_details': {'inputs': {'order_id': 'string'}}}}
        claim = CLAIMS['cancel_pending_order'][0].format_map(_FIELDS)
        steps = [{'substeps': [{'role': 'user', 'content': 'Please cancel order #W1234567.'}]},
                 {'substeps': [{'role': 'planner', 'content': '{"action": "handoff"}'},
                               {'role': 'planner', 'name': 'handoff', 'content': _json.dumps(
                                   {'handoff_to': 'order_agent', 'message': 'Cancel order #W1234567.',
                                    'context': {'order_id': '#W1234567'}})}]},
                 {'substeps': [{'role': 'order_agent', 'content': claim}]}]
        trajectory = normalize({'trajectory_id': 'skip-write',
                                'steps': [{'index': i, **s} for i, s in enumerate(steps, 1)]}, 'tau')
        rows, _ = run_perspective_checks('P1_INTENT_PLAN', seed_constraints_for_perspective('P1_INTENT_PLAN', tools),
                                         trajectory, None, {'policy': '', 'tools': tools, 'python_timeout': 3})
        self.assertFalse([r for r in rows if r['verdict'] == 'ERROR'], rows)
        self.assertEqual([(r['assertion_name'], r['step_index']) for r in rows if r['verdict'] == 'VIOL'],
                         [('p1_seed_nonempty_progress', 3)])


if __name__ == '__main__':
    unittest.main()
