# Recovery Orchestrator

Part 4 of AgentReroute. Given a failed agent run and the fault-attribution (FA)
diagnosis, it decides **whether, where and how** to intervene: roll back to a
step, apply a repair strategy, replay, and accept the result only if an
independent fresh-context verifier passes it, all within a per-task budget.

Stdlib only (Python 3.10+). It reads FA **only through FA's JSON output
files**, never by importing `agentrx`, so this branch merges independently of
`faultattribution`.

AgentReroute has four parts. This folder is part 4:

1. Benchmark and evaluation metrics
2. Fault attribution: *which step failed, which agent is responsible, and why* (branch `faultattribution`)
3. Repair strategies: *what to change* (Repair team; consumed through the strategy registry)
4. **Recovery orchestrator: *whether, where, when and how to apply a repair*, under uncertainty and budget**

## Idea

Four observations from the literature (see [Related work](#related-papers-and-source-code)) shape the design:

1. **Agents are poor at catching their own errors.** Intrinsic self-correction
   mostly fails without external feedback. It works when the feedback comes from
   tests, execution or a separate checker. Detection is the bottleneck: given
   the error's location, models can often fix it. The orchestrator therefore
   takes the error location from a separate FA module and never asks the agent
   to diagnose itself.
2. **Detection and correction are different skills, and diagnosis is not repair.**
   "Detect, then miscorrect" is a known failure, and guidance decays when the
   agent re-executes. So a repair is never trusted just because it was applied.
   A **fresh-context verifier** that sees neither the repair nor its reasoning
   decides whether it worked.
3. **Benchmarks and graders can be gamed.** Graders that share a sandbox with the
   agent have been gamed to near-100%, and agents reward-hack spontaneously. Gold
   labels and rewards are therefore kept on a separate side of the code that the
   decision and repair path cannot see, and tests enforce this.
4. **FA output is uncertain and post-hoc.** FA runs on a saved, already failed
   trace, and its runs can disagree. The orchestrator treats FA as a *distribution
   over hypotheses*. It intervenes only when the expected value is positive, rolls
   back early enough to cover near-tied hypotheses, and abstains or escalates
   otherwise. Cheap rule-based detectors cover what FA cannot see online (loops,
   tool errors).

The resulting loop is:

```
RCA payload ─► decide (intervene? where? which strategy?) ─► roll back to step k
   ─► apply repair ─► replay ─► fresh-context verify ─► accept / try next / fall back ─► log
```

## Layers

```
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ 5. Evaluation (gold side, offline)   evaluation.py, cli.py                │
 │    oracle / mis-attribution inputs, controls, metrics, calibration        │
 ├──────────────────────────── leakage boundary ─────────────────────────────┤
 │ 4. Verification                       verifier.py, llm.py                 │
 │    fresh-context verifier: task view + transcript only                    │
 ├───────────────────────────────────────────────────────────────────────────┤
 │ 3. Execution                          orchestrator.py, replay.py, budget.py│
 │    core loop, prefix replay, repair directives, budget caps, loop guard,  │
 │    fallbacks, event log, ScoreLedger (written by env, read only by 5)     │
 ├───────────────────────────────────────────────────────────────────────────┤
 │ 2. Decision                           policy.py, registry.py, detectors.py│
 │    baseline / EV / control policies, strategy registry, runtime detectors │
 ├───────────────────────────────────────────────────────────────────────────┤
 │ 1. Input                              rca.py, taxonomy.py, trace.py,      │
 │                                       tau_retail.py                       │
 │    FA files → RCAPayload; traces, rollback points, irreversible steps     │
 └───────────────────────────────────────────────────────────────────────────┘
        ▲ FA JSON files (faultattribution)      ▲ strategy registry (Repair team)
```

| Layer | Question it answers | Main types |
|---|---|---|
| 1. Input | What does FA believe, and what does the trace look like? | `RCAPayload`, `Candidate`, `Trace`, `Prefix` |
| 2. Decision | Should we intervene, where, and with which strategy? | `Decision`, `Option`, `Strategy`, `Finding` |
| 3. Execution | Apply it safely within budget, and what if it fails? | `Orchestrator`, `Outcome`, `Rollout`, `RepairDirective`, `BudgetManager`, `LoopGuard` |
| 4. Verification | Did it actually work, judged independently? | `FreshContextVerifier`, `VerifierResult` |
| 5. Evaluation | How good is the orchestrator against ground truth? | `GoldLabel`, `run_condition`, `recovery_metrics` |

Layers 1–4 make up the decision path and never import layer 5.
`tests/test_leakage.py` checks this.

## Structure

```
recovery_orchestrator/
├── README.md               this file
├── pyproject.toml          packaging (stdlib only, no dependencies)
├── __main__.py, cli.py     `python3 -m recovery_orchestrator payload | decide | eval-decisions`
│
├── taxonomy.py         [1] AgentRx categories, label normalization, category → handling
├── rca.py              [1] RCAPayload + FA adapter (Tier A files → Tier B aggregates)
├── trace.py            [1] Trace/Step/ToolCall, rollback points, prefixes, irreversible steps
├── tau_retail.py       [1] gold-free tau-bench loaders; TauBenchEnvironment stub (plan)
│
├── registry.py         [2] strategy schema + validation
├── resources/
│   ├── repair_registry.json [2] Repair team's export_registry() snapshot (CLI default)
│   └── mock_registry.json  [2] small MOCK registry used by unit tests
├── policy.py           [2] BaselinePolicy, EVPolicy, UnmodifiedReplayPolicy, EVConfig
├── detectors.py        [2] loop and tool-error detectors, RuntimeMonitor
│
├── orchestrator.py     [3] core loop, fallbacks, JSONL event log
├── replay.py           [3] TaskView, RepairDirective, TemplateRepairer, Rollout,
│                           ScoreLedger, ScriptedEnvironment (offline)
├── budget.py           [3] BudgetCaps/BudgetManager, LoopGuard
│
├── verifier.py         [4] FreshContextVerifier, transcript rendering, verdict parsing
├── llm.py              [4] LLMClient interface, ScriptedLLM
│
├── evaluation.py       [5] gold loading, oracle/mis-attribution, decision eval,
│                           calibration, recovery metrics
├── adapters/
│   └── benchmark.py        OrchestratorPolicy for benchmark's recovery harness (duck-typed, no import)
├── integration/
│   ├── e2e_benchmark.py    all recovery conditions on one benchmark corpus, same harness and scoring
│   ├── cross_branch_check.py  orchestrator × repair-strategies × faultattribution check
│   └── REPORT.md           latest cross-branch results and open questions
└── tests/
    ├── test_rca.py, test_registry_trace.py, test_policy.py,
    ├── test_orchestrator.py, test_evaluation.py, test_leakage.py
    ├── helpers.py          scripted environment / verifier used by tests
    └── fixtures/
        ├── smoke_run/          real FA smoke output (faultattribution @ c8a75c2)
        ├── synthetic_run/      hand-written FA-format run with disagreement + an error
        ├── cases.json, tau_raw.json   synthetic gold + tau raw (gold values marked GOLDSENTINEL)
        └── make_synthetic.py   regenerates the synthetic fixtures
```

## Quick start

From the repository root:

```bash
python3 -m unittest discover -s recovery_orchestrator/tests -t .
```

With FA output (e.g. from a `faultattribution` worktree at `../fa`):

```bash
# Tier A files -> RCA payload with Tier B aggregates (all traces, or --trajectory-id)
python3 -m recovery_orchestrator payload --fa-run ../fa/runs/smoke/experiment

# Decision for one trace, with the ranked options
python3 -m recovery_orchestrator decide --fa-run ../fa/runs/smoke/experiment --trajectory-id fixture-count-mismatch

# Offline decision evaluation against gold (works for Magentic-One too)
python3 -m recovery_orchestrator eval-decisions --fa-run <fa-run> --cases ../fa/data/benchmark/cases.json
python3 -m recovery_orchestrator eval-decisions --oracle --cases ../fa/data/benchmark/cases.json
```

`--allow-irreversible` lets rollbacks cross executed write actions, which is safe
only in simulation (see [Irreversible steps](#irreversible-steps)).

## Input contract with FA

The adapter reads, for every `jobs/<variant>/rep-<n>/<domain>/<sha256(trajectory_id)>/`:
`result.json` (prediction, status), `validation_log.json` (violations), and
`trajectory_ir.json` (the trace). By default it pools the full AgentRx variants
(`evidence` and `checklist` both on, `joint` protocol). Use `--variants` to override.

Tier B fields computed by the adapter:

| Field | Definition here |
|---|---|
| `total_runs` | All pooled result files, errors included |
| `n_runs` | Pooled runs that localized a step (not Inconclusive, not `error`) |
| `top_k` | ≤3 distinct steps by votes (ties → earlier step); majority agent and category within the step |
| `share` / `confidence` | votes / `total_runs`, so errors and abstentions dilute confidence; `confidence` = top share |
| `inconclusive_share` | Inconclusive or `error` runs / `total_runs` |
| `violations_by_step` | `{step: {count, runs, categories}}` over pooled validation logs |
| `status` | `error` if every run errored, `ok` if all ok, else `partial_checker_errors` |

Tier C fields (`ranked_probabilities`, `all_failures`, `causal_chain`) are optional
and only read from an official `rca_payload.json`.

**Track B (`recovery_handoff.json`, schema `agentrx-recovery-v1`)** is read
automatically when every pooled run has one (`rca.aggregate_handoffs`). A run gives
candidates only if FA's own gate allows it: `decision=repair_required`,
`attribution_status=ranked`, and `status` not `error`. A step's share is the mean
confidence over all runs, counting 0 where a run didn't list it. The majority
`decision` is kept as `fa_decision`, and `within_boundary` makes the policy abstain.
Shares are then uncalibrated model estimates (`confidence_kind`).

**Benchmark hypotheses** (7 modes) are mapped by `taxonomy.MODE_TO_CATEGORY`. This
mirrors `benchmark/resources/seven_modes.json`, except that `cycles` maps to the local
`Cycle` label instead of plan adherence.

**Ask for the FA team:** emit `rca_payload.json` per trace with the Tier A
`prediction` plus `confidence`, `top_k` (max 3), `n_runs`, `total_runs`,
`inconclusive_share` and `status`, in the format `RCAPayload.to_dict()` writes
(`payload --out DIR` writes examples).

Run agreement measures stability, not correctness. Check `eval-decisions`'
`calibration` table on the labeled traces and pass
`evaluation.binned_calibrator(table)` as `EVConfig.calibrator` before trusting EV.

## Decision logic

**Category handling** (`taxonomy.HANDLING`). Inconclusive → abstain.
SystemFailure → retry. UnderspecifiedUserIntent → clarify.
IntentNotSupported and GuardrailsTriggered → escalate, if they are the top
hypothesis with confidence ≥ 0.5. All other categories → roll back and repair
from the registry.

**Rollback point** of an attributed step: the step itself if an agent produced
it. For a tool output, the agent step that made the call. For a user/system turn,
the agent's next step; if the run ended there, the resume point `last + 1`.

**EV policy.** For each rollback point *s* and strategy *σ*:

```
P(success) = Σ_{candidates c with rollback(c) ≥ s, σ targets c}  belief(c) · prior(σ) · 0.97^(rollback(c) − s)
cost       = (tokens(σ) + 400 · steps replayed from s + 3000 verifier) / 1000 · 0.01
EV         = P(success) · 1.0 − cost              intervene only if EV > 0
```

An earlier rollback covers every candidate downstream of it, so near-tied
candidates lead to the earliest point (ties go to the earlier step). The
discount models guidance decaying when it is applied further from the error.
After a verified failure, each covered candidate's belief is updated by Bayes,
`b(1−q)/(1−bq)`, so the next option moves to another hypothesis. All constants
are in `EVConfig` and are **placeholders**.

**Baseline.** The top-1 candidate and its best-prior strategy, retried up to N times
(repeats allowed). Improvements must beat it.

**Runtime detectors.** Loops (the same call with the same output 3×, with no
write in between, or the same agent message 3× in a row) and tool errors. They
add low-weight candidates (`detector_weight`; loops use the local `Cycle` label)
and stop a replay early. A stopped rollout fails without spending verifier tokens.

**Budget and guards.** There are per-task caps on tokens, time, attempts and
distinct rollback points. The guard never repeats a (step, strategy) pair, and
stops if a repair's verifier score falls below the original's.

**Outcomes.** `recovered` (verifier passed; never set otherwise), `partial`
(the best repair improved the verifier score but did not pass; gaps flagged),
`escalated`, `abstained`, `aborted` (budget or guard stop).

### Irreversible steps

tau write tools (`tau_retail.WRITE_TOOLS`) are non-rollback points by default. A
rollback that would undo an executed write is blocked (`respect_irreversible`).
With gold attribution this blocks **21 of the 29 tau traces**, because a write
follows the critical step. Replay rebuilds the environment from scratch, so
crossing a write is safe in the experiments. Run them with
`respect_irreversible=False` (`--allow-irreversible`) and report this setting.
Compensation actions (Saga-style) would be needed for live use.

## Leakage boundary

* Gold `actions`, `outputs`, `reward` and failure labels are read only in
  `evaluation.py`. `tests/test_leakage.py` checks with an AST scan that no other
  module imports it (or `agentrx`).
* `tau_retail` returns only the instruction, policy and message history.
* An environment writes ground-truth rewards only to its `ScoreLedger`.
  `Rollout` has no reward, and a test fails if the orchestrator touches the ledger.
* The verifier sees the task view and the rollout transcript only. Injected
  guidance, the RCA rationale, the strategy and the original trace are excluded
  (tested).
* Gold reaches the decision path only on purpose, through the oracle and
  mis-attribution conditions. These pass a step and category, never actions or rewards.

## Evaluation

`evaluation.run_condition` runs one condition and scores it from the ledger.
`recovery_metrics` reports, per condition:

* `recovery_rate`: failed traces whose delivered result has reward 1. Unverified repairs are never delivered.
* `verified_rate`, `false_recovery_rate`: verifier passes whose reward is 0.
* `harm_rate`: interventions ending below the original reward, or below the unmodified-replay control's.
* `unnecessary_intervention_rate`: interventions where the unmodified replay succeeded anyway.
* Tokens, seconds and attempts per recovered task.

Conditions are `orchestrator`, `baseline`, `unmodified` (the control, needed
because the user simulator is stochastic), `oracle` (gold critical step, an
upper bound) and `misattribution` (a non-critical gold failure, for robustness).
Real recovery runs only on the 29 tau traces, so report per-trace results without
significance claims. Magentic-One supports offline decision evaluation only.

## Status

Done (offline core): adapter, registry schema and mock, policies, budget,
guards, detectors, verifier gate, orchestrator loop, controls, metrics, CLI, and
tests (synthetic environment and scripted LLM).

Repair integration: the Repair team's registry is bundled
(`resources/repair_registry.json`) and is the CLI default. Their
`OrchestratorRepairer` plugs in as `Orchestrator(..., repairer=...)`; the loop
calls the repairer's optional `record_outcome(trajectory_id, verified, rollout)`
after every repaired attempt, with the verifier's verdict (never the reward).
A repairer directive of kind `none` means "not applicable": it is skipped, not
replayed. If the repairer substitutes a strategy, that strategy is logged as
`applied_strategy` and not offered again on the same step. Results and open
questions: [integration/REPORT.md](integration/REPORT.md).

Not yet:
* **tau-bench replay.** `TauBenchEnvironment` documents the plan (prefix tool
  re-execution, user-simulator history restore, ledger-only reward). The tau-bench
  dependency still needs team agreement.
* **Live LLM client** for the verifier and replay agent. `llm.LLMClient` is the interface.
* **Registry priors** are the Repair team's estimates, not measured (see REPORT R3).
* **Calibration** of FA agreement against gold, once live FA runs exist (none are measured yet).

## Fixtures

`tests/fixtures/smoke_run/` is copied from FA's offline smoke run (`faultattribution`
@ `c8a75c2`; `checks.json` and per-step files omitted). `synthetic_run/`,
`cases.json` and `tau_raw.json` are hand-written in FA's formats by
`tests/fixtures/make_synthetic.py`. Gold-only values contain `GOLDSENTINEL`.

## Related papers and source code

These references come from the team's background research (handoff of
2026-10-15). arXiv IDs are as recorded there. The "Code" column lists
repositories only where one was found; "—" means no confirmed public code.

### Design templates

| Reference | What we take from it | Where in this code | Code |
|---|---|---|---|
| **Self-Healing Agentic Orchestrators**, arXiv 2606.01416 | Main template: failure signal → failure class → targeted recovery under a budget → verify. Its baselines (static, retry-only, ReAct, full replanning) and budget sweep shape our conditions. | `orchestrator.py` loop; `BaselinePolicy` (retry-only); `evaluation.CONDITIONS` | — |
| **Rollback-Induced Reflection**, arXiv 2609.18304 (related: AgentRewind, GA-Rollback, WebRollback) | Where and when to roll back, and that reflection is injected at the rollback point | `trace.rollback_point`, `build_prefix`; `EVPolicy` rollback choice | not confirmed |
| **CodeRescue**, arXiv 2607.19338 | Budget routing: cheap retry vs escalate, with risk control (Conformal Risk Control) | `budget.py`, EV cost term, escalate fallbacks; CRC thresholds are future work | [Qijia-He/agent-budget-control](https://github.com/Qijia-He/agent-budget-control) |
| **AgentTether**, arXiv 2607.06273 | Diagnosis ≠ repair; guidance decays on re-execution, so interventions must be guarded | `EVConfig.distance_discount`; verifier gate; `LoopGuard` | not confirmed |
| **SagaLLM** (VLDB 2025) | Irreversible steps and Saga-style compensation | `tau_retail.WRITE_TOOLS`, `respect_irreversible`; compensation is future work | not confirmed |
| **AgentDebug** (ICML 2026) | Root cause + targeted feedback (reference only: its ALFWorld/WebShop/GAIA environments don't match our data) | registry `template` guidance | [ulab-uiuc/AgentDebug](https://github.com/ulab-uiuc/AgentDebug) |
| **AgentRx**, arXiv 2602.02475 | Fault attribution and the 10-category taxonomy we consume | `taxonomy.py`, `rca.py` (input contract) | [microsoft/AgentRx](https://github.com/microsoft/AgentRx); our reconstruction: branch `faultattribution` |

### Evidence behind the design choices

| Finding | Source | Design consequence |
|---|---|---|
| Intrinsic self-correction mostly fails without reliable external feedback | Kamoi et al. (survey); ReaLMistake | External FA and an independent verifier, not self-reflection |
| Detection, not correction, is the bottleneck; given the location, models often fix it | BIG-Bench Mistake | Feed the attributed step and rationale into the repair |
| Detection rate doesn't predict correction; location hints can hurt some models | arXiv 2601.00828 | Never assume a repair worked; verify every attempt |
| "Detect, then miscorrect" in multi-stage pipelines | arXiv 2605.27559 | Verification gate; stop when a repair worsens the score |
| Models correct a claim more often when it appears under an external role (single paper) | arXiv 2606.05976 | **Fresh-context** verifier with no access to the repair's reasoning |
| Long-horizon agents hit a "no-recovery bottleneck" and declare success unchecked | IT-Bench + MAST, arXiv 2607.05775 | Never declare success without verification |
| Multi-agent setups add little; many failures come from agent interaction | MAST | Agent-agnostic rollback points; responsibility ≠ location |
| SWE-bench Verified audit: 59.4% of 138 hard tasks flawed; SWE-bench Pro mis-grading (unconfirmed) | OpenAI (Feb 2026); Datacurve (May 2026) | Report per-trace results; don't trust a single grader |
| Agent benchmarks gamed to near-100% when graders share the sandbox; spontaneous reward hacking | BenchJack / Berkeley RDI; arXiv 2605.12673; arXiv 2607.22368 | Leakage boundary: `ScoreLedger` and gold outside the decision path, tested |

### Data and environments

| Resource | Use here | Code |
|---|---|---|
| tau-bench (retail) | 29 labeled failed traces; re-executable, scored by reward → real recovery experiments | [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench) |
| Magentic-One traces (Who&When-style) | 44 labeled traces; live web tasks, not re-executable → offline decision evaluation only | [microsoft/autogen](https://github.com/microsoft/autogen) (Magentic-One) |
| AgentRx release data | Gold labels and raw traces, fetched by FA's `prepare --fetch` | [microsoft/AgentRx](https://github.com/microsoft/AgentRx) |
