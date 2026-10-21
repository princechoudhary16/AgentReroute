# Track B: divergence boundaries and ranked recovery handoff

## Approved intent

The user approved the proposed Track B approach: use a divergence threshold based
on applicable conclusive checks; emit up to three ranked critical-failure hypotheses
with typed categories, confidence, rationale and evidence for Recovery. The scores
are uncalibrated model estimates, not calibrated probabilities or vote shares.
This is an explicit extension to the independent paper reconstruction. Preserve
single-output benchmark variants and add an opt-in ranked protocol/config.

## Decisions under that authorization

- Add protocol `ranked`, requiring evidence generation and an explicit numeric
  `divergence_threshold` in [0,1]. `max_candidates` is 2 or 3 (default3). The local
  demonstration config uses0.30; it is an example setting, not an empirically tuned
  boundary. Strict comparison is divergence > threshold, so equality is tolerated.
- Boundary counts unique **rule/step evaluations**, not generated-rule count. Skip,
  checker errors and unresolved semantic judgments do not enter the denominator.
  A semantic check with a definite failing criterion is a violation even if other
  criteria are unclear; a no-failure check containing any UNCLEAR is unresolved.
- Deduplicate mechanically identical definitions at the same step, including harmless
  Python whitespace/comments/function-name differences. Preserve guard and checker
  semantics. Do not claim to prove arbitrary code/text semantic equivalence.
  Conflicting conclusive results for an identical rule/step are unresolved and
  reported, not arbitrarily selected to change the threshold.
- No conclusive checks => boundary decision `inconclusive`, rate null. Otherwise
  `repair_required` when strictly above threshold, `within_boundary` when not.
  Within boundary does not establish task success or "creative validity".
- The gate decides whether to request attribution. Above threshold, one final
  ranked judge call returns 0..max_candidates distinct observed step hypotheses.
  Do not invent additional candidates to fill the list. Empty candidates require
  an abstention reason. Below threshold/no evidence, skip ranked judging and
  produce a compatible Inconclusive legacy prediction with a clear reason.
- Candidates contain rank, step_index, observed agent, canonical category,
  confidence0..1, nonempty rationale and nonempty valid evidence_refs. Reject bool,
  non-finite/out-of-range confidence, unknown locations/categories/agents, duplicate
  steps and malformed evidence. Rank by confidence descending, earliest step on ties.
  Do not normalize scores: this is not a full probability distribution.
- Use taxonomy `agentrx-9-v1` (nine failure types; Inconclusive is separate abstention).
  Do not invent a 19-mode taxonomy. Future extension needs explicit vocabulary and
  mapping. Ranking considers causality and subsequent recovery, not violation counts.

## Handoff contract and compatibility

Save `recovery_handoff.json` per job and also under `result.recovery_handoff`:

    schema_version: agentrx-recovery-v1
    trajectory_id, domain, measurement_kind
    status: ok | partial_checker_errors | error
    decision: repair_required | within_boundary | inconclusive
    boundary: threshold, comparison(>), violated_checks, satisfied_checks,
              conclusive_checks, divergence_rate, skipped_checks, error_checks,
              unresolved_checks, duplicate_evaluations, conflicting_checks
    taxonomy_version: agentrx-9-v1
    confidence_kind: uncalibrated_model_estimate
    attribution_status: ranked | abstained | not_requested | error
    candidates: [rank, step_index, agent, category, confidence, rationale, evidence_refs]
    abstention_reason: string or null

The boundary decision remains repair_required even if the ranker abstains or errors:
a divergence alarm must not disappear due to missing attribution. Recovery may act
on candidates only if decision is repair_required, attribution_status ranked, and
handoff status is not error. Checker errors remain visible as partial status.
A strict loader validates a payload against the corresponding trajectory before a
consumer acts on it. It enforces count/rate/decision consistency and ranking types.

Preserve `result.prediction` as the highest ranked candidate (without score/rank)
for existing benchmark metrics, or a validated Inconclusive prediction otherwise.
On ranker failure existing result.status becomes error/prediction null while the
handoff still records the boundary alarm and attribution error. Always write the
handoff, including pipeline generation/checker/judge failures.

The stored `origin/orchestrator` RCAPayload pools repeated-run vote shares and has
no native model-confidence semantics. Do not write new data to its legacy
`rca_payload.json` or masquerade estimates as votes/share. Publish a versioned,
validated consumer contract and document the required orchestrator adapter.
Do not merge or edit another team's branch implicitly.

## Pipeline and transport

Existing generation/checker behavior remains available. Add a canonical rule
fingerprint to check records, including skipped/error records when possible.
Legacy semantic SAT labels stay unchanged for paper compatibility; the boundary
module explicitly handles their UNCLEAR counters/rubric results as unresolved.

A new `ranked_judge` prompt sees full observable trajectory, policy/tools, violations,
checker errors, divergence boundary and taxonomy, never gold labels. On native
Ollama use the existing schema-constrained response facility with schema-bound
steps/agents/categories and max candidate count. Other clients retain strict JSON
prompt/validation. Existing single/joint/localize/categorize prompts retain semantics.

## Delivery and evaluation

Provide config `configs/track_b_ollama.json`, offline `smoke-track-b`, real run/diagnose
integration, saved versioned handoff, and aggregate ranked reports. Keep old paper
scores on top1 unchanged; separately report top1/2/3 step hits with fixed denominators,
decision/abstention/error counts and exploratory confidence reliability bins. Name
these confidence bins explicitly uncalibrated; do not advertise calibration from a
small pilot. Gold labels enter evaluation only, not gating/ranking.

Tests cover exact threshold equality,0/1 boundaries, skipped/unknown/errors, duplicated
and conflicting checks, empty traces/checks, category/step/evidence validation,
unsorted/tied scores, missing/duplicate candidates, generator/ranker errors, unchanged
legacy mode, CLI/cache identity/report artifacts and wheel install. Live Ollama pilot
may be inconclusive when generated checks do not apply. Separately verify live
ranking using explicitly controlled checks, labeled as a hybrid integration test.

## Review focus

False zero-violation conclusions from empty checks; duplicated rule inflation;
UNCLEAR-as-SAT leakage; bool/NaN/invalid confidence; dropped boundary alarms after
ranker errors; fabricated/unobserved evidence; legacy vote-confidence confusion;
fixed-denominator scoring and accidental gold leakage.
