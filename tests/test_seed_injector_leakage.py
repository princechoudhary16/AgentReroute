"""Phase 0 debt marker: the mp_rl seed checks recognise faults by the injector's own wording.

``WRITE_CLAIM`` and ``STUCK_TOOL`` in ``agentrx/perspectives.py`` are written to match the exact
completion strings (``CLAIMS``) and the ``PROCESSING`` payload that ``benchmark/faults/operators.py``
injects. So on our own benchmark the seed layer partly detects faults by their known surface form,
not by reasoning about the trace - which inflates every mp_rl seed-only number on that benchmark.

This test asserts the coupling is gone. It MUST fail today, and is marked ``expectedFailure`` so the
suite stays green while the debt stays visible. The fix is the Phase 1 fault-content refresh: reword
the operator content tables from real failure shapes (AgentRx / Who&When) so the seed patterns no
longer match the injector's output. When that lands, this test passes, ``expectedFailure`` turns it
into an unexpected success (a failure), and that is the signal to delete the decorator.

Phase 0 scope: do NOT weaken the assertion, and do NOT edit the operators or seed patterns to make
it pass early.
"""
import re
import unittest

from agentrx.perspectives import STUCK_TOOL, WRITE_CLAIM
from benchmark.faults.operators import CLAIMS, PROCESSING, _Fields

# Any concrete argument values: the coupling is in the templates' fixed phrasing, not the fields.
_FIELDS = _Fields({'order_id': '#W1234567', 'reason': 'no longer needed', 'item_ids': 'A1',
                   'new_item_ids': 'B2', 'payment_method_id': 'paypal_1', 'address1': '1 Main St',
                   'city': 'Springfield', 'user_id': 'user_1'})


class SeedInjectorLeakageTest(unittest.TestCase):
    @unittest.expectedFailure
    def test_seed_patterns_do_not_match_injector_content(self):
        shared = []
        if re.search(STUCK_TOOL, PROCESSING):
            shared.append(('STUCK_TOOL', 'PROCESSING'))
        for tool, templates in CLAIMS.items():
            for template in templates:
                if re.search(WRITE_CLAIM, template.format_map(_FIELDS), re.I):
                    shared.append(('WRITE_CLAIM', tool))
        self.assertEqual(shared, [], f'seed checks match injector content (Phase 1 refresh pending): {shared}')


if __name__ == '__main__':
    unittest.main()
