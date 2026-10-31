"""[B5] LLM fault selection over the perspective analysis.

Why this stage exists. The linear policy ranks candidates by theta.h over ten numeric features and
scores 13.8% step accuracy on the published tau subset. Two attempts to improve that ranking failed
and were measured: training theta by REINFORCE cost eight points on a held-out split (44% to 36%,
McNemar p=0.0078), and a repair-and-verify cascade over the top three candidates changed no answer
at all, because "violations fell after repairing step X" is no evidence that X was the fault.

Splitting results by whether the true step was in the candidate pool showed why. The decision stage
converts in-pool evidence at 70% on tau and 89% on magentic, and out-of-pool evidence at 16% and 9%.
Accuracy is governed by pool recall, not by how the pool is ranked. So this module replaces the
ranking with one model call that reads the evidence and decides, and lets it range over the whole
trace rather than confining it to the pool.

Two scopes:

``pool``   choose only among the ranked candidates. Ceiling is pool recall: 41.4% on tau.
``trace``  choose any step, with the analysis as fallible evidence. Ceiling is 100%. Measured 34.5%
           against 24.1% for ``pool``, so the pool was a ceiling rather than a help.

No gold is read here and theta is never written: this is inference only.
"""
from .predictions import validate_prediction
from .providers import ProviderError


def candidate_category(candidate):
    """Category of a candidate in either shape (see ``_candidate_view``)."""
    return candidate.get('dominant_category') or candidate.get('category')


def _candidate_view(candidate, rank=None):
    """Compact candidate for the prompt: the evidence, not the feature vector.

    Accepts a candidate in either shape this codebase produces. ``encode_candidates`` emits
    ``dominant_category``/``dominant_perspective`` and ``select_action`` adds ``rank``, while
    ``pipeline._compact`` rewrites those to ``category``/``perspective`` and drops ``rank`` before
    they reach ``result['ranked_candidates']``.
    """
    checks = candidate.get('violated_checks') or []
    return {'rank': candidate.get('rank', rank), 'step_index': candidate['step_index'],
            'agent': candidate.get('raw_agent') or candidate.get('agent'),
            'suggested_category': candidate_category(candidate),
            'dominant_perspective': candidate.get('dominant_perspective') or candidate.get('perspective'),
            'policy_probability': round(float(candidate.get('probability', 0.0)), 4),
            'violations': [{'perspective': c.get('perspective'), 'check': c.get('assertion_name'),
                            'step_index': c.get('step_index'),
                            'detail': str(c.get('reason') or c.get('detail') or '')[:280]}
                           for c in checks[:6]]}


def select_payload(trajectory, blame, ranked, scope, pool):
    candidates = [_candidate_view(c, i) for i, c in enumerate(ranked[:max(1, int(pool))], 1)]
    return {'trajectory': trajectory, 'scope': scope,
            'analysis': {'blame_matrix': blame, 'ranked_candidates': candidates,
                         'pool_note': ('The true fault may be absent from this pool; on the measured '
                                       'subset it is present in roughly 2 of 5 traces.')}}


def llm_select(trajectory, blame, ranked, provider, scope='trace', pool=5):
    """One model call that picks the decisive fault. Falls back to the ranked top-1 on any failure.

    Returns (prediction, log). The prediction is validated against the trajectory first, so a model
    that invents a step or an agent degrades to the policy's answer instead of voiding the trace.
    The log records ``status`` so a run whose calls were refused can be told apart from one whose
    model simply chose badly; ``analyze_run.py`` refuses to report an arm that failed too often.
    """
    top1 = ranked[0] if ranked else None
    log = {'scope': scope, 'pool': int(pool), 'status': 'ok'}
    if top1 is None:
        log['status'] = 'no_candidates'
        return None, log
    fallback = validate_prediction({'step_index': top1['step_index'], 'agent': top1['raw_agent'],
                                    'category': top1['dominant_category'], 'rationale': top1['rationale'],
                                    'evidence_refs': top1['evidence_refs']}, trajectory)
    try:
        answer = provider.complete('mp_rl_select', select_payload(trajectory, blame, ranked, scope, pool))
    except (ProviderError, ValueError, KeyError, TypeError) as exc:
        log.update(status='error', error=f'{type(exc).__name__}: {exc}', fell_back=True)
        return fallback, log
    try:
        prediction = validate_prediction(answer, trajectory)
    except (ValueError, KeyError, TypeError) as exc:
        log.update(status='invalid_prediction', error=f'{type(exc).__name__}: {exc}', fell_back=True)
        return fallback, log
    allowed = {c['step_index'] for c in ranked[:max(1, int(pool))]}
    if scope == 'pool' and prediction['step_index'] not in allowed:
        # The model left the pool it was told to choose from; honour the declared scope.
        log.update(status='out_of_pool', chose=prediction['step_index'], fell_back=True)
        return fallback, log
    log.update(fell_back=False, chose=prediction['step_index'],
               in_pool=prediction['step_index'] in allowed,
               moved=prediction['step_index'] != top1['step_index'])
    return prediction, log
