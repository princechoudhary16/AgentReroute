# Recovery orchestrator: known issues

| | |
|---|---|
| Baseline | `main` @ `ed28bfb` (Merge branch 'mp_opt' into main) |
| Fixes | uncommitted working tree, 2026-10-22 (orchestrator tests: 66 OK) |
| Found by | the end-to-end run on the merged `main` (benchmark smoke → FA smoke / Track B smoke → orchestrator in the benchmark recovery harness) |

Line numbers under **main** refer to `ed28bfb`. Line numbers under **fixed** refer
to the current working tree. Every bug marked *reproduced* was run against the
committed code (`git archive HEAD`) and against the working tree. The outputs are
quoted.

## Summary

| ID | Bug | Where (main) | Severity | Status |
|---|---|---|---|---|
| [B1](#b1) | Track B output (`recovery_handoff.json`) cannot be read | `rca.py` `is_full_agentrx` L131–134, `load_payload` L165–184 | **High**: breaks FA → orchestrator handoff | Fixed |
| [B2](#b2) | FA's divergence gate (`within_boundary`) is ignored | `policy.py` `_gate` L92–94 | Medium | Fixed |
| [B3](#b3) | Failed repairs scoring 0 are reported as `partial` | `orchestrator.py` `Orchestrator.run` L143 | Medium (latent in main) | Fixed |
| [B4](#b4) | Loop keeps replaying after the environment refuses for budget | `orchestrator.py` `Orchestrator.run` L117–141 | Low (latent in main) | Fixed |
| [B5](#b5) | Leakage test misses sub-packages and other team packages | `tests/test_leakage.py` `test_decision_path_never_imports_evaluation_or_agentrx` L44–47 | Low (test gap) | Fixed |
| [G1](#g1) | No way to run the orchestrator in the benchmark harness | none (missing integration) | Gap | Added |
| [O1](#o1) | `mp_rl` `rca_payload.json` is never used | `rca.py` `find_runs` L143–158, `load_payload` L180–181 (fixed tree) | **High** | **Open** |
| [O2](#o2) | Replay cost underestimated (400 vs ~730 tokens/step) | `policy.py` `EVConfig.replay_tokens_per_step` L64 | Medium | **Open** |
| [O3](#o3) | Mislabelled `tool_failure` → retry, which cannot fix agent faults | `taxonomy.py` `HANDLING` L68; `policy.py` `EVPolicy.options` L188 | Medium | **Open** |
| [O4](#o4) | Benchmark adapter cannot screen controls; stand-in verifier is weak | `adapters/benchmark.py` `CallbackVerifier` L86, `OrchestratorPolicy.__init__` L116 | Medium | **Open** |
| [O5](#o5) | Budget refusal detected by exception class *name* | `adapters/benchmark.py` `BenchmarkEnvironment.rollout` L67–68 | Low | **Open** |
| [O6](#o6) | Runtime loop detector cannot stop a benchmark replay early | `adapters/benchmark.py` `BenchmarkEnvironment.rollout` L77 | Low | **Open** |

---

## Fixed

### B1
**Track B output (`recovery_handoff.json`) cannot be read.** *Reproduced.*

- **Where (main):** `recovery_orchestrator/rca.py`
  - `is_full_agentrx()`, L131–134, accepts only `protocol == 'joint'`.
  - `find_runs()`, L149, applies that filter.
  - `load_payload()`, L165–184, knows only `result.json` and `rca_payload.json`.
- **Description:** FA's Track B variant runs with `protocol: 'ranked'` and writes
  its ranked candidates to `recovery_handoff.json` (schema `agentrx-recovery-v1`,
  see `docs/track-b.md`). `main` filters those runs out, so the default call fails:
  ```
  FileNotFoundError: No FA results for tau/fixture-count-mismatch in runs/e2e/fa-track-b/experiment
  ```
  Forcing the variant (`--variants ranked`) avoids the crash but reads only the top-1
  `result.json` prediction. Model confidence 0.85 then becomes run *agreement* 1.0
  (3/3 runs agree), and candidates 2–3 are lost:
  ```
  --variants ranked: 1.0 [(3, 1.0)]
  ```
- **Fix:**
  - `rca.py` `is_full_agentrx()` L136–140 also accepts `ranked`.
  - `load_payload()` L183 reads `recovery_handoff.json` when every pooled run has one.
  - The new `aggregate_handoffs()` L280 applies FA's own gate: a run gives candidates
    only if `decision == repair_required`, `attribution_status == ranked` and
    `status != error`. A step's share is the mean confidence over *all* runs, with 0
    where a run did not list the step.
  - New fields record where the numbers came from: `fa_decision` and
    `confidence_kind = 'uncalibrated_model_estimate'` (L49–50).
  - Now: `confidence 0.85, top_k [(3, MisinterpretationOfToolOutput, 0.85)]`.
- **Why this is right:**
  - The gate is the one `docs/track-b.md` tells consumers to apply ("act on
    candidates only when decision is `repair_required`, attribution is `ranked`, and
    status is not `error`").
  - Counting absent steps as 0 keeps one confident outlier run from dominating, and
    runs that FA gated out count toward `inconclusive_share`.
  - With one run, the share is the model's confidence, so nothing is invented.
- **Remaining risk:**
  - Track B confidences are explicitly *uncalibrated*. EV now multiplies them by
    priors as if they were probabilities. Calibrate them (`evaluation.calibration_table`)
    before trusting EV with live FA output.
  - The schema check accepts only `agentrx-recovery-v1`. A future v2 fails loudly,
    which is intended.

### B2
**FA's divergence gate (`within_boundary`) is ignored.**

- **Where (main):** `recovery_orchestrator/policy.py` `_gate()`, L92–94. `RCAPayload`
  had no field for FA's decision.
- **Description:** Track B can conclude that a run is `within_boundary` (no repair
  required). `main` has no way to see that decision. Such runs carry no ranked
  candidates, so they surface only as "no step localized", and the reason given is
  wrong. Worse, any mixed pooling could act on candidates from runs FA judged fine.
- **Fix:** `policy.py` `_gate()` L95 abstains when `payload.fa_decision == 'within_boundary'`,
  with the reason "FA divergence gate: within boundary (no repair required)". Test:
  `tests/test_adapters.py::TrackBHandoffTest::test_within_boundary_abstains`.
- **Why:** the boundary decision is FA's evidence that nothing needs fixing. Acting
  anyway would add unnecessary interventions, which is one of our reported metrics.
- **Remaining risk:** FA notes the 0.30 threshold is "a demonstration setting, not a
  tuned operating point". A badly tuned threshold now suppresses repairs silently.
  The abstention reason is logged, so it stays auditable.

### B3
**Failed repairs scoring 0 are reported as `partial`.** *Reproduced.*

- **Where (main):** `recovery_orchestrator/orchestrator.py` `Orchestrator.run()`, L143:
  `if best is not None and (reference is None or best[0] > reference):`
- **Description:** if the original trace is not verified (`verify_original=False`),
  `reference` is `None`, so *any* finished attempt counts as an improvement, even
  with score 0. The outcome becomes `partial` ("best repair improved but did not
  pass verification") when nothing improved:
  ```
  main: scores [0.0, 0.0, 0.0] -> status partial
  ```
- **Fix:** `orchestrator.py` L147–148. Without a reference, `partial` now requires a
  score above 0:
  `improved = best is not None and (best[0] > reference if reference is not None else best[0] > 0)`.
  Now: `all-zero repairs -> escalated`.
- **Why:** `partial` means "deliver the best unverified result with gaps flagged". If
  it fires on zero-score rollouts, a broken rollout could be delivered (when
  `deliver_partial=True`), and `partial` stops meaning anything in the reports.
- **Remaining risk:**
  - **Latent in main:** `verify_original` defaults to `True`. It is reached via the
    API, and by the new benchmark adapter, which disables it (see [O4](#o4)).
  - Any score > 0 still counts as an improvement when there is no reference. That is
    a weak signal with a 0/1 verifier.

### B4
**Loop keeps replaying after the environment refuses for budget.** *Reproduced.*

- **Where (main):** `recovery_orchestrator/orchestrator.py` `Orchestrator.run()`, the
  attempt loop L117–141. There is no check for `rollout.stopped_early == 'budget'`.
- **Description:** when the environment refuses a replay (the benchmark harness
  raises `BudgetExceeded`), the orchestrator treats it as one failed attempt and asks
  for the next option. Every request is refused again. Each failure also lowers the
  belief in the hypothesis (`observe_failure`), so the run ends with a misleading reason:
  ```
  main: replays after refusal: 3 | status: escalated | no verified repair (no option with EV > 0.0 ...)
  ```
  With original verification on, it stopped only because the loop guard misread the
  refusal as "repair worsened the verifier score".
- **Fix:** `orchestrator.py` L126–129. A budget refusal from the environment ends the
  loop with `stop_reason = 'budget: environment refused the replay'`:
  ```
  now: replays after refusal: 1 | status: aborted | budget: environment refused the replay
  ```
  Test: `tests/test_adapters.py::BenchmarkAdapterTest::test_harness_budget_is_respected`.
- **Why:**
  - The external budget is final, so further requests are wasted calls.
  - The outcome should be `aborted` (budget), not `escalated` (repair failed): the
    benchmark reports these separately.
  - A refusal says nothing about whether the hypothesis is right, so the Bayes update
    must not run.
- **Remaining risk:**
  - Latent in main: no environment in `main` returns `'budget'`.
  - Detection relies on the string `'budget'` set by the adapter (see [O5](#o5)).

### B5
**Leakage test misses sub-packages and other team packages.**

- **Where (main):** `recovery_orchestrator/tests/test_leakage.py`
  `test_decision_path_never_imports_evaluation_or_agentrx()`, L44–47. It uses
  `PACKAGE.glob('*.py')` (top level only) and checks only for `agentrx`.
- **Description:** the guarantee is "nothing on the decision path reads gold or
  imports other teams' packages". The test could not see code in sub-packages (the
  new `adapters/`), and it did not check for `benchmark` or `repair_strategies`. Those
  two packages now share the repo, and the benchmark package can reach gold.
- **Fix:** L45–50 use `rglob` (excluding `tests/` and `integration/`, which are
  harnesses by design), assert that `adapters/benchmark.py` is actually scanned, and
  check `agentrx`, `benchmark` and `repair_strategies`.
- **Why:** after the merge, a stray `from benchmark...` in the adapter would make
  gold reachable from the orchestrator. The test now fails on that.
- **Remaining risk:** a static import scan cannot catch dynamic access, such as
  `importlib`, or a policy walking private fields of the harness's environment. The
  benchmark README says the same about its ledger ("hidden by convention, not by a
  sandbox").

### G1
**No way to run the orchestrator in the benchmark harness** (integration gap, not a bug).

- **Where (main):** no code existed. The benchmark's `recover` CLI only knows its own
  `POLICIES`, and its hypotheses use 7 mode ids, which `normalize_category()`
  (`taxonomy.py` L38–47) rejects with `ValueError: Unknown failure category`.
- **Added:**
  - `taxonomy.py` `MODE_TO_CATEGORY` L83 and `category_from_mode()` L95, mirroring
    `benchmark/resources/seven_modes.json`.
  - `adapters/benchmark.py` (`OrchestratorPolicy`, `BenchmarkEnvironment`,
    `payload_from_hypotheses`, `CallbackVerifier`), duck-typed with no `benchmark`
    import.
  - `integration/e2e_benchmark.py`.
- **Deliberate deviation:** `cycles` maps to the local `Cycle` label, not to
  plan adherence as in the benchmark's `to_agentrx` table, because the orchestrator
  and Repair team handle loops separately.
- **Risk:** the mode mapping is lossy (`context_propagation` and
  `misinterpretation_of_tool_output` both become MisinterpretationOfToolOutput). The
  local copy can drift from `seven_modes.json` if the benchmark changes it.

---

## Open (not fixed)

### O1
**`mp_rl` `rca_payload.json` is never used.** *Reproduced.*

- **Where:** `rca.py` `find_runs()` L143–158 and `is_full_agentrx()` L136–140
  (accepts `joint`/`ranked`, not `mp_rl`); `load_payload()` L180–181.
- **Description:** the Phase 2 multi-perspective protocol (`agentrx/pipeline.py`
  `diagnose_mp_rl`, `_rca_payload` L154) writes the `rca_payload.json` we asked FA
  for. Its format is compatible: fed to `RCAPayload.from_dict()`, it gives
  `[(7, MisinterpretationOfToolOutput, 0.55), (9, Instruction/PlanAdherenceFailure, 0.3)]`.
  But:
  1. `mp_rl` runs are filtered out by default, so `load_payload` raises `FileNotFoundError`.
  2. With `--variants mp_rl`, the official file is skipped (`variants is None` check,
     L181) and the top-1 `result.json` aggregation is used instead.
  3. Even when read, only `runs[0]`'s file is used. `mp_rl` writes one payload *per
     repetition* (`n_runs: 1`), so the other repetitions are silently ignored.
- **Risk:** the newest FA mode cannot drive the orchestrator, or does so from one
  repetition only.
- **Suggested fix:**
  - Accept `mp_rl` in the filter.
  - Read `rca_payload.json` regardless of `variants`.
  - Pool per-repetition payloads the same way as `aggregate_handoffs`.
  - Ask FA whether `probability` is calibrated.

### O2
**Replay cost underestimated.**

- **Where:** `policy.py` `EVConfig.replay_tokens_per_step = 400`, L64, used by `_cost()`.
- **Evidence:** on the benchmark fixture, replaying a step costs median **732** tokens
  (range 678–1036). EV underestimates cost by about 1.8×, and 6 of 35 failed episodes
  ended `aborted (budget: tokens)` under the heuristic attribution.
- **Risk:** EV prefers early rollbacks and resamples that the budget cannot pay for.
  Cost-per-recovery is understated in planning.
- **Suggested fix:** derive tokens per step per episode. The harness's cap is
  `token_multiple × original tokens`, which gives `original_tokens / len(steps)`.

### O3
**Mislabelled `tool_failure` → retry, which cannot fix agent faults.**

- **Where:** `taxonomy.py` `HANDLING` L68 (`SystemFailure → RETRY`);
  `policy.py` `EVPolicy.options()` L188 assumes the hypothesis category is right.
- **Evidence:** the 9 drift episodes, where the heuristic says `tool_failure`
  three times. Recovery: fixed rules 3/9, repair policy 6/9, orchestrator EV 0/9. The
  retry strategy (`tool_outcome_routing`) is cheap with a high prior, so it wins EV.
  But in the benchmark only *guidance* removes agent faults.
- **Risk:** the main reason the orchestrator trails the baselines under noisy
  attribution (0.37 vs 0.46 / 0.57). With gold attribution it ties for best (0.77).
- **Suggested fix:** model label uncertainty, so that guidance (useful whatever the
  label) beats retry when category confidence is low. Treat this as a design change
  and validate it beyond the 8-task fixture.

### O4
**Benchmark adapter cannot screen controls; stand-in verifier is weak.**

- **Where:** `adapters/benchmark.py` `CallbackVerifier` L86;
  `OrchestratorPolicy.__init__` L116 (`verify_original=False`).
- **Evidence:**
  - False-intervention rate on controls is 1.00 under the heuristic attribution
    (same for every attribution-based policy).
  - 10 rollouts passed the observable check (`looks_ok`) but had reward 0.
- **Risk:**
  - Without a real verifier, the orchestrator cannot tell that a control run was
    fine, so it intervenes anyway.
  - "Recovered" claims are only as good as `looks_ok`, the same check the baselines use.
- **Suggested fix:** an Ollama `LLMClient` for `FreshContextVerifier`, then enable
  `verify_original`.

### O5
**Budget refusal detected by exception class *name*.**

- **Where:** `adapters/benchmark.py` `BenchmarkEnvironment.rollout()`, L67–68
  (`type(exc).__name__ != 'BudgetExceeded'`).
- **Risk:** renaming the harness exception would turn budget refusals into crashes.
  An unrelated exception with the same name would be swallowed.
- **Suggested fix:** pass the exception class in, as is already done for the `Directive` class.

### O6
**Runtime loop detector cannot stop a benchmark replay early.**

- **Where:** `adapters/benchmark.py` `BenchmarkEnvironment.rollout()`, L77. The
  monitor sees the replay only after it has finished.
- **Risk:** loops are still classified as failures, and the verifier call is skipped.
  But the replay tokens are already spent, so the detector saves no cost here.
- **Suggested fix:** needs a per-step hook in the benchmark's `Workflow`, which belongs
  to the benchmark team.

---

## Not orchestrator issues (for reference)

- `repair_strategies/cli.py` `_json_arg`: `Path(value).is_file()` on inline JSON
  raises `OSError: [Errno 36] File name too long` on Linux. It fails
  `test_cli.CliTest.test_explain_history_and_exhausted_ladder`. This was reported
  2026-10-20 and is still present in `main`.

## Reproducing

```bash
python3 -m unittest discover -s recovery_orchestrator/tests -t .
python3 -m benchmark smoke --out runs/e2e/benchmark
python3 -m agentrx smoke-track-b --out runs/e2e/fa-track-b
python3 recovery_orchestrator/integration/e2e_benchmark.py --corpus runs/e2e/benchmark/corpus \
    --submission runs/e2e/benchmark/submissions/heuristic.json --out runs/e2e/recovery
```

All numbers come from the offline fixture (scripted agents, 35 failed and 17 control
episodes). They show the pipeline works, not how a real model performs.
