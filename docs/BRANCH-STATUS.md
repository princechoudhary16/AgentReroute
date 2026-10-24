# Branch status: mp_rl-gemini-actor

Working branch for the Gemini-backed benchmark actor. Read this before running anything.

## Read these warnings first

**1. `benchmark/runs/corpus-v1` is contaminated. Do not use or extend it.**
~89% of its episodes contain at least one degraded actor response. Those become empty actions
in the trace, indistinguishable from genuine agent errors, so an unknown subset carries wrong
gold labels. It is gitignored so it does not travel with this branch — if you regenerate with
the same settings you will reproduce the contamination. The retry/config fix must land first.

**2. There is a deliberately failing test. Do not "fix" it.**
`tests/test_seed_injector_leakage.py` is marked `@unittest.expectedFailure`. It documents real
debt: the mp_rl seed checks (`WRITE_CLAIM`, `STUCK_TOOL` in `agentrx/perspectives.py`) are
written against the exact strings the fault injector emits (`CLAIMS`, `PROCESSING` in
`benchmark/faults/operators.py`), so on our own benchmark the detector partly recognises faults
by their known wording rather than by reasoning. 15 couplings: 14 WRITE_CLAIM x CLAIMS, plus
STUCK_TOOL x PROCESSING. The fix is the Phase 1 fault-content refresh (reword operator content
from real failure shapes). Weakening the assertion destroys the tripwire.
Note: raw `python -m unittest` exits 0 on an unexpected success. Gate on `python -m agentrx
check`, which fails it properly.

**3. The actor is nondeterministic at temperature 0.**
Two cold-cache runs of the same task with the same seed diverge at step 1 and produce different
step counts (38 vs 36 observed). Consequences:
- The corpus is the artifact, not the recipe. `llm_cache/` is part of the deliverable.
- Regenerating from the same command yields a *different* corpus.
- Two people running the same command will not get comparable data. Do not diff your numbers
  against someone else's regenerated corpus.
Pairing still holds *within* a run, because injected runs replay the clean prefix from cache.

**4. Do not change the attributor path.**
The actor runs on Gemini native `generateContent`. `complete()` (the agentrx diagnosis path) is
deliberately left as it was, so the previously measured agentrx column (31.0% tau / 18.2%
magentic on gemini-3.1-flash-lite) stays comparable. Changing it invalidates that baseline and
it would need re-running.

**5. API keys come from the environment only.** `GEMINI_API_KEY`. Never commit a literal.

## Known open bug (unresolved)

The actor returns unusable replies at a rate that varies run to run:
- `MALFORMED_RESPONSE` with zero output tokens: 7.5%-16.7% of calls. Constrained decoding under
  `responseMimeType: application/json` aborts before emitting anything. Not truncation, not
  thinking (thoughtsTokenCount is 0).
- A second class the leniency counter does not catch: `STOP` responses whose content is present
  but is not a valid action. Combined unusable rate ~13%-32%.

Unanswered and blocking: of that second class, how much is valid JSON failing action validation
(genuine agent error, keep it) versus not-JSON-at-all (constraint failure, fix it). A three-way
config comparison is pending: current JSON mode / no `responseMimeType` / explicit
`responseSchema`.

## What works

- Gemini actor on native `generateContent`; system prompt correctly in `systemInstruction`.
- Leniency detection and raw-body capture (`lenient.jsonl`, `lenient_raw.jsonl`).
- No-progress guard (planner-scoped; specialists intentionally excluded because the
  `stuck_processing` fault *is* a specialist loop).
- Checkpoint read-only on evaluation; model id pinned in the settings lock.
- Real tau-bench data: 115 tasks via `python3 -m benchmark fetch-tau`.

## Measured rates (treat as indicative - they move run to run)

| | value |
|---|---|
| Eligibility (pass^1 clean) | 60-80% |
| Activation (injected -> failed) | ~50% |
| Cost per failed episode | ~$0.03 |
| Spend to date | ~$3.50 |
