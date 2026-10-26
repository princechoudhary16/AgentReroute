"""[B5] LLM fault selection over the perspective analysis, with an optional offline lesson bank.

Why this exists. The linear policy ranks candidates by theta.h over 10 numeric features. Measured
on the published tau subset it scores 13.8% acc@1, while gold sits anywhere in the candidate pool
in only 41.4% of traces. The repair-and-verify cascade could not improve on that: the verifier's
'+' signal means "violations fell after repairing step X", which is no evidence that X was the
fault, so it accepted rank 1 in 20 of 29 traces and the 6 answers it did move all landed on
another wrong step.

So this module puts a model in the decision path. It receives the trajectory, the blame matrix and
the ranked candidates with the specific constraint violations behind each, and returns one
attribution in the same schema as the baseline judge. Two scopes:

``pool``   choose only among the ranked candidates. Ceiling is pool recall: 41.4% on tau.
``trace``  choose any step in the trace, with the analysis as fallible evidence. Ceiling is 100%,
           and comparing it against the baseline joint judge isolates whether the perspective
           evidence adds anything over judging the raw trace.

Gold boundary. Only ``write_lessons`` reads gold, it runs offline on the train split, and the bank
it writes is frozen before evaluation; ``bank_digest`` enters the experiment identity so every
result records which bank produced it. Nothing in the selection path reads gold.
"""
import re
from pathlib import Path

from .io import digest, read_json, write_json
from .predictions import validate_prediction
from .taxonomy import category as canonical_category
from .providers import ProviderError

# The fault injector's own operator vocabulary. A lesson that names one of these is describing how
# our generator writes faults, not how agents fail, and would not transfer to the published subset.
OPERATOR_WORDS = ('drop_context', 'irrelevant_tool', 'misread_value', 'misroute', 'omit_detail',
                  'policy_contradiction', 'skip_write', 'stale_output', 'stuck_processing', 'tool_error')
# Record-specific identifiers: memorising these is memorising cases, not failure behaviour.
_ID_PATTERNS = (re.compile(r'#[A-Z]\d{4,}'), re.compile(r'\b\d{10}\b'),
                re.compile(r'\b(?:credit_card|gift_card|paypal)_\d{4,}\b'), re.compile(r'\btr-[0-9a-f]{6,}\b'))
LESSON_KEYS = ('category', 'perspective', 'signal', 'trap')


def leakage_reason(lesson):
    """Why this lesson must be rejected, or None when it is safe to keep."""
    text = ' '.join(str(lesson.get(k, '')) for k in LESSON_KEYS)
    lowered = text.lower()
    for word in OPERATOR_WORDS:
        if word in lowered:
            return f'names injector operator {word!r}'
    for pattern in _ID_PATTERNS:
        # Matched against the original casing: order ids are '#W2378156', not '#w2378156'.
        found = pattern.search(text)
        if found:
            return f'contains record identifier {found.group(0)!r}'
    return None


class LessonBank:
    """Written lessons grouped by taxonomy category, retrieved locally at selection time.

    Capped per category because a bank that grows one lesson per training episode floods the
    selector's context and dilutes the evidence it is meant to sharpen.
    """

    def __init__(self, lessons=(), per_category=4):
        self.per_category = max(1, int(per_category))
        self.lessons = [dict(x) for x in lessons]

    @classmethod
    def load(cls, path, per_category=4):
        if not path:
            return cls((), per_category)
        if not Path(path).exists():
            raise ValueError(f'Lesson bank {path!r} does not exist; build it with '
                             '"agentrx build-lessons" before running a variant that uses it')
        stored = read_json(path)
        if isinstance(stored, dict):
            return cls(stored.get('lessons', ()), stored.get('per_category', per_category))
        return cls(stored, per_category)

    def save(self, path):
        write_json(path, {'per_category': self.per_category, 'bank_digest': self.bank_digest,
                          'count': len(self.lessons), 'support_profile': self.support_profile(),
                          'category_counts': self.category_counts(), 'lessons': self.lessons})

    @property
    def bank_digest(self):
        return digest([{k: x.get(k) for k in LESSON_KEYS} for x in self.lessons])

    def add(self, lesson):
        """Merge a lesson in, counting a repeat as confirmation rather than storing it twice."""
        reason = leakage_reason(lesson)
        if reason is not None:
            return reason
        key = digest({k: str(lesson.get(k, '')).strip().lower() for k in LESSON_KEYS})
        for existing in self.lessons:
            if existing.get('key') == key:
                existing['confirmed'] = existing.get('confirmed', 1) + 1
                return None
        self.lessons.append({**{k: lesson.get(k) for k in LESSON_KEYS}, 'key': key, 'confirmed': 1})
        return None

    def prune(self):
        """Keep the most-confirmed lessons per category, up to the cap."""
        by = {}
        for lesson in self.lessons:
            by.setdefault(lesson.get('category'), []).append(lesson)
        kept = []
        for group in by.values():
            group.sort(key=lambda x: (-x.get('confirmed', 1), str(x.get('signal', ''))))
            kept.extend(group[:self.per_category])
        self.lessons = kept
        return self

    def support(self, lesson):
        """Independent training episodes attesting this lesson's CATEGORY.

        Free-text lessons almost never match each other exactly, so ``confirmed`` (exact-duplicate
        merges) came out 1 for all 46 lessons of the first real build. The attestation that does
        exist is categorical: 20 of those 46 episodes failed the same way. So support is counted
        per category, and ``min_confirmed`` reads as "only use lessons from failure modes seen at
        least N times in training", which is a frequency prior rather than a claim about one
        lesson's wording.
        """
        return max(int(lesson.get('category_support', 0)), int(lesson.get('confirmed', 1)))

    def recount_support(self):
        """Stamp each lesson with how many episodes produced a lesson in its category."""
        totals = {}
        for lesson in self.lessons:
            totals[lesson.get('category')] = totals.get(lesson.get('category'), 0) + lesson.get('confirmed', 1)
        for lesson in self.lessons:
            lesson['category_support'] = totals.get(lesson.get('category'), 0)
        return self

    def retrieve(self, categories, limit=8, min_confirmed=1):
        """Lessons for the categories in play, best attested first. No model call.

        ``min_confirmed`` thresholds on category support (see ``support``) and ``limit`` is the
        capacity cap. Both act at retrieval time, so one bank file serves a whole capacity sweep
        with no rebuild. Lessons are drawn only from the categories actually in play; falling back
        to the entire bank would hand the model guidance for failure modes it is not looking at.
        """
        wanted = {c for c in categories if c}
        floor = max(1, int(min_confirmed))
        hits = [x for x in self.lessons if x.get('category') in wanted and self.support(x) >= floor]
        hits.sort(key=lambda x: (-self.support(x), str(x.get('signal', ''))))
        return [{k: x.get(k) for k in LESSON_KEYS} for x in hits[:max(0, int(limit))]]

    def support_profile(self):
        """How many lessons survive each support threshold: the capacity curve, for the report."""
        return {t: sum(1 for x in self.lessons if self.support(x) >= t) for t in (1, 2, 3, 5, 10, 20)}

    def category_counts(self):
        return {c: sum(1 for x in self.lessons if x.get('category') == c)
                for c in sorted({x.get('category') for x in self.lessons})}

    def __len__(self):
        return len(self.lessons)


def candidate_category(candidate):
    """Category of a candidate in either shape (see ``_candidate_view``)."""
    return candidate.get('dominant_category') or candidate.get('category')


def _candidate_view(candidate, rank=None):
    """Compact candidate for the prompt: the evidence, not the feature vector.

    Accepts a candidate in either of the two shapes this codebase produces. ``encode_candidates``
    emits ``dominant_category``/``dominant_perspective`` and ``select_action`` adds ``rank``, while
    ``pipeline._compact`` rewrites those to ``category``/``perspective`` and drops ``rank`` before
    they reach ``result['ranked_candidates']``. ``llm_select`` is handed the raw list and
    ``write_lesson`` the compacted one, so reading only one spelling raised KeyError on every
    lesson and silently produced an empty bank.
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


def select_payload(trajectory, blame, ranked, scope, pool, lessons):
    candidates = [_candidate_view(c, i) for i, c in enumerate(ranked[:max(1, int(pool))], 1)]
    return {'trajectory': trajectory, 'scope': scope,
            'analysis': {'blame_matrix': blame, 'ranked_candidates': candidates,
                         'pool_note': ('The true fault may be absent from this pool; on the measured '
                                       'subset it is present in roughly 2 of 5 traces.')},
            'lessons': list(lessons)}


def llm_select(trajectory, blame, ranked, provider, scope='trace', pool=5, bank=None, lesson_limit=8,
               lesson_min_confirmed=1):
    """One model call that picks the decisive fault. Falls back to the ranked top-1 on any failure.

    Returns (prediction, log). The prediction is already validated against the trajectory, so a
    model that invents a step or an agent degrades to the policy's answer instead of voiding it.
    """
    top1 = ranked[0] if ranked else None
    log = {'scope': scope, 'pool': int(pool), 'status': 'ok', 'lessons_used': 0,
           'lesson_limit': int(lesson_limit), 'lesson_min_confirmed': int(lesson_min_confirmed),
           'bank_digest': bank.bank_digest if bank is not None else None}
    if top1 is None:
        log['status'] = 'no_candidates'
        return None, log
    fallback = validate_prediction({'step_index': top1['step_index'], 'agent': top1['raw_agent'],
                                    'category': top1['dominant_category'], 'rationale': top1['rationale'],
                                    'evidence_refs': top1['evidence_refs']}, trajectory)
    lessons = []
    if bank is not None and len(bank):
        lessons = bank.retrieve({candidate_category(c) for c in ranked[:max(1, int(pool))]},
                                lesson_limit, lesson_min_confirmed)
    log['lessons_used'] = len(lessons)
    try:
        answer = provider.complete('mp_rl_select', select_payload(trajectory, blame, ranked, scope, pool, lessons))
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


def normalize_lesson(lesson):
    """Canonicalise the lesson's category, or return None when the model invented one.

    The first real build produced 'Intent/PlanAdherenceFailure', which is not in the taxonomy. Stored
    unvalidated it can never match a candidate's category, so it is retrieval-dead weight.
    """
    try:
        label = canonical_category(lesson.get('category'))
    except (ValueError, KeyError, TypeError):
        return None
    return {**lesson, 'category': label}


def write_lesson(trajectory, gold, prediction, ranked, provider):
    """Offline, train-split only: show the model its miss and the gold label, ask for one lesson.

    This is the ONLY function in the selection stack that reads gold. It is never called from
    ``diagnose_mp_rl``; it runs from the ``build-lessons`` command before any evaluation.
    """
    payload = {'trajectory': trajectory,
               'gold': {'step_index': gold['step_index'], 'agent': gold['agent'], 'category': gold['category']},
               'selected': {'step_index': prediction and prediction.get('step_index'),
                            'agent': prediction and prediction.get('agent'),
                            'category': prediction and prediction.get('category'),
                            'rationale': prediction and prediction.get('rationale')},
               'ranked_candidates': [_candidate_view(c, i) for i, c in enumerate(ranked[:5], 1)]}
    answer = provider.complete('mp_rl_lesson', payload)
    lesson = answer.get('lesson') if isinstance(answer, dict) else None
    if not isinstance(lesson, dict):
        raise ValueError('Lesson stage must return {"lesson": {...}}')
    return {k: str(lesson.get(k, '')).strip() for k in LESSON_KEYS}


def build_lesson_bank(cases, config, provider, out_path, per_category=4, limit=None, on_case=None):
    """OFFLINE TRAINING. Reads gold on the supplied split, writes a frozen lesson bank.

    For each case: run the full mp_rl pipeline with the selector and NO bank, compare the pick to
    gold, and on a miss ask the model to write one transferable lesson. Lessons that name injector
    operators or record identifiers are rejected by ``leakage_reason`` rather than stored, because
    the bank is built on our generated faults and evaluated on the published subset.

    No weights are touched anywhere in here: the output is text, and theta stays at its loaded
    value. Callers must freeze the bank before evaluation; ``bank_digest`` goes into the experiment
    identity so each result records the bank it used.
    """
    from .pipeline import diagnose_mp_rl  # deferred: pipeline imports this module
    import tempfile

    bank = LessonBank((), per_category)
    stats = {'cases': 0, 'hits': 0, 'misses': 0, 'lessons_written': 0,
             'rejected_leakage': 0, 'rejected_category': 0, 'rejection_reasons': {}, 'errors': 0}
    selected = list(cases)[:limit] if limit else list(cases)
    with tempfile.TemporaryDirectory() as artifacts:
        for case in selected:
            trajectory, gold = case['trajectory'], case['gold']['critical']
            stats['cases'] += 1
            try:
                result = diagnose_mp_rl(trajectory, case['policy'], case['tools'], config,
                                        provider, artifacts)
            except Exception as exc:                      # one bad case must not end the build
                stats['errors'] += 1
                if on_case: on_case(stats, trajectory, f'pipeline error: {type(exc).__name__}: {exc}')
                continue
            prediction = result.get('prediction')
            if prediction and prediction.get('step_index') == gold['step_index']:
                stats['hits'] += 1
                if on_case: on_case(stats, trajectory, 'hit')
                continue
            stats['misses'] += 1
            try:
                lesson = write_lesson(trajectory, gold, prediction, result.get('ranked_candidates', []), provider)
            except Exception as exc:
                stats['errors'] += 1
                if on_case: on_case(stats, trajectory, f'lesson error: {type(exc).__name__}: {exc}')
                continue
            normalized = normalize_lesson(lesson)
            if normalized is None:
                stats['rejected_category'] += 1
                reason = f"category {lesson.get('category')!r} is not in the taxonomy"
                stats['rejection_reasons'][reason] = stats['rejection_reasons'].get(reason, 0) + 1
                if on_case: on_case(stats, trajectory, f'REJECTED ({reason})')
                continue
            lesson = normalized
            reason = bank.add(lesson)
            if reason is None:
                stats['lessons_written'] += 1
                if on_case: on_case(stats, trajectory, f"lesson: {lesson.get('category')}")
            else:
                stats['rejected_leakage'] += 1
                stats['rejection_reasons'][reason] = stats['rejection_reasons'].get(reason, 0) + 1
                if on_case: on_case(stats, trajectory, f'REJECTED ({reason})')
    # Deliberately NOT pruned here: the bank stores every distinct lesson with its support count,
    # and lesson_limit / lesson_min_confirmed choose the capacity at evaluation time. Pruning at
    # build time would bake one capacity in and force a rebuild per point of the sweep.
    bank.recount_support().save(out_path)
    stats.update(bank_size=len(bank), bank_digest=bank.bank_digest, out=str(out_path),
                 support_profile=bank.support_profile(), category_counts=bank.category_counts())
    return bank, stats
