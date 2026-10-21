# AgentReroute benchmark design (KAN-4)

Status: implemented in `benchmark/` (v0.1.0, 2026-10-15). Verified offline with scripted agents;
no model-backed results yet. See section 14 for what differs from the original plan.
Owner: Benchmark team. Consumers: Fault Attribution (KAN-5), Repair Strategy
(KAN-6), Recovery Orchestrator (KAN-7).

## 1. Purpose

The benchmark is the shared ground truth for AgentReroute. It provides:

1. a corpus of failed multi-agent runs in one trace format;
2. gold labels (responsible agent, step and failure mode), hidden from the
   systems under test;
3. a re-executable environment so recovery can be tested by rollback and replay,
   not only diagnosed after the fact;
4. a scoring harness, fixed baselines and a dashboard so every team's results
   are comparable.

It builds on **AgentRx** [1] (trace format, critical-failure labelling,
attribution metrics) and extends it where AgentRx is limited:

| AgentRx limitation | Source we borrow from | What we take |
|---|---|---|
| Small released corpus (73 of 170 traces); labels are post-hoc human annotations | **MAS-FIRE** [2] | Controlled fault injection (prompt mutation, response rewriting, message routing), so gold labels are known by construction |
| No systematic coverage of tool faults | **ToolMaze** [3] | 2×2 tool-fault grid: explicit/silent × transient/persistent; Perturbation Recovery Rate |
| Output-only logs; traces cannot be re-executed | **TraceElephant** [4] | Fully observable traces (inputs, context, state) in an executable environment |
| Recovery is not measured | **τ-bench** [5], **Who&When** [6] | State-based success, pass^k consistency; agent/step attribution protocol and baselines |

## 2. Scope decisions

- **Failure taxonomy:** the team's 7 modes (section 4). AgentRx categories that
  do not map are kept in an explicit `out_of_scope` bucket and reported.
- **Models:** open-source only, served locally by **Ollama**. No paid API calls.
- **Primary domain:** τ-bench retail, wrapped in a multi-agent workflow.
  **Held-out domain:** τ-bench airline, used only for transfer evaluation.
- **Runtime:** Python 3.10+, standard library where possible, matching the
  `faultattribution` and `orchestrator` branches.

## 3. Environment

### 3.1 Multi-agent retail workflow

τ-bench retail tasks, tools and database, executed by a small multi-agent system:

```
user simulator ◄─► planner ─┬─► order_agent    (pending orders)
                            ├─► product_agent  (catalogue questions)
                            └─► returns_agent  (returns, exchanges)
```

The planner also writes the final user-facing answer (there is no separate
responder agent; see section 14).

Multiple agents are required so that handoff faults (context propagation) and
routing faults (drift) can occur at all. Write tools (orders, exchanges,
returns, address changes) are marked irreversible in the trace.

### 3.2 Models and determinism

- Default models: `qwen2.5:7b` and `llama3.1:8b` via Ollama (configurable).
  The model digest is recorded in every run manifest.
- Temperature 0, fixed seed per run.
- **Response cache** keyed by `sha256(model digest, prompt, params)`. Replaying
  an unchanged prefix returns identical outputs, which makes rollback-and-replay
  deterministic up to the point where something changes.
- The user simulator also runs on Ollama and is cached the same way.

### 3.3 Observability

Each step records the agent's full input (prompt and context), its output, tool
calls and results, the handoff message, and a database state version. Every trace
is exported in two views:

- `full`: everything above (TraceElephant-style);
- `output_only`: agent outputs and tool results only (AgentRx / Who&When-style).

This allows measuring the effect of observability on attribution.

## 4. Failure taxonomy and fault operators

| # | Mode | Definition | Injection operator | Variants |
|---|---|---|---|---|
| 1 | Instruction / plan adherence | Agent ignores directives, under- or over-executes against plan or policy | Remove or contradict a policy clause in the agent prompt; suppress a required confirmation | clause type, step position |
| 2 | Drift | Agent selects an irrelevant tool or next agent for the query | Reroute a handoff to an irrelevant agent; substitute an irrelevant tool in the call | target agent/tool |
| 3 | Cycles | Repeated invocation without progress | Tool persistently returns "pending, retry"; echo the agent's prior action back as new input | persistence length |
| 4 | Tool failure | API fails, times out or returns unexpected results, and the agent does not handle it | Explicit-transient (timeout/503, clears on retry); explicit-persistent (deprecated endpoint); silent (stale or wrong values with valid schema) | ToolMaze 2×2 cell |
| 5 | Misinterpretation of tool output | Tool output is correct but the agent misreads it | Rewrite the agent's reasoning over the output (wrong count, unit, ID) | field, error size |
| 6 | Context propagation | Required context is not passed to a downstream agent or tool | Drop or truncate required fields in a handoff message | field, truncation |
| 7 | Missing details in final output | Final answer omits requested information | Rewrite the final response to drop a requested item | item type |

Every operator is also parameterised by **timing** (early / middle / late in the
run). Operators and their parameters are logged separately from the trace; the
log is gold-side data.

A written decision guide with boundary rules (for example, mode 4 silent vs
mode 5) ships with the taxonomy file, `taxonomy/seven_modes.json`.

## 5. Gold-label construction

For each (task, operator, parameters):

1. **Task eligibility.** Run the clean workflow 3 times. Keep the task only if
   all 3 succeed (pass^3 = 1), so later failures are attributable to the
   injection, not to the model's baseline weakness.
2. **Paired runs.** Run clean and injected with the same seed and cache.
3. **Activation and causation check.** The injected run must fail on τ-bench's
   state-based outcome check while the clean run passes. Record whether the
   fault was activated (the faulty value was actually consumed).
4. **Labels.** Each failed episode records:
   - `fault_step`: where the fault entered;
   - `critical_step`: first unrecoverable failure in AgentRx's sense, i.e. the
     agent step that should have caught or handled it;
   - `responsible_agent`, `mode`, operator parameters;
   - `acceptable_steps`: other defensible answers, for multi-perspective scoring.
5. **Tolerated runs.** Injected runs that still succeed are kept as
   `fault_tolerated` controls for false-alarm and harm measurement.

Natural failures (models failing without injection) are labelled manually with
the AgentRx protocol and a second annotator on a sample; agreement is reported.

## 6. Corpus

| Split of the corpus | Content | Labels | Replayable |
|---|---|---|---|
| Injected (core) | Retail runs × 7 modes × variants | By construction (section 5) | Yes |
| Natural | Unprompted failures in the same workflow | Manual, double-annotated sample | Yes |
| Controls | Clean successes and fault-tolerated runs | No failure | Yes |
| External | AgentRx release (29 τ retail + 44 Magentic-One), Who&When | Existing human labels mapped to 7 modes | No (attribution only) |

**Splits.** `train`, `calibration` and `test` are disjoint by task template
**and** by fault variant. Airline is a held-out domain. Teams tune only on
`train` and `calibration`; `test` is scored once per release.

**Trace format.** Compatible with the `faultattribution` branch IR
(`trajectory_id`, `domain`, `instruction`, `steps[].substeps[]` with `role`
and content), extended with optional `context`, `state_version` and `view`
fields that existing readers can ignore.

## 7. Scorecard

| Layer | Metrics |
|---|---|
| Detection (online) | Precision, recall, false alarms on controls, detection delay; scored on trace prefixes with future steps hidden |
| Attribution | Agent accuracy; step accuracy (exact, ±1, ±3) for both `fault_step` and `critical_step`; top-3 hit rate; acceptable-step accuracy; 7-mode macro-F1; expected calibration error |
| Recovery | Recovery rate, conditional recovery rate, cost as a multiple of one run |
| Agent robustness | Per fault: robustness score (MAS-FIRE RS); ToolMaze P1–P4 success (= PRR here), recovery cost RC, explicit − implicit gap |
| Harm and consistency | Harmful-intervention rate, collateral database changes, pass^k |
| Efficiency | Tokens, p95 wall time, budget breaches |

Statistics: paired runs on the same failed episode, bootstrap 95% confidence
intervals, McNemar's test for paired success. Results are always broken down by
mode and corpus split, never reported only in aggregate.

## 8. Baselines and ablations

- **Attribution:** random step, last step, all-at-once LLM judge, Who&When
  step-by-step, AgentRx (from the `faultattribution` branch).
- **Recovery:** no recovery, full restart, capped retry, generic reflection,
  fixed mode-to-action rules, oracle attribution (upper bound).
- **Ablations** (equal total budget): predicted vs oracle attribution;
  fault-aware vs fault-blind repair; adaptive policy vs fixed rules; with vs
  without verification; `full` vs `output_only` view; held-out domain.

## 9. Interfaces for other teams

- **Fault Attribution** reads `corpus/<split>/` traces (no gold) and submits a
  predictions file: per trace, a ranked list of up to 3 hypotheses
  `{step_index, agent, mode, confidence, rationale}`.
- **Recovery Orchestrator** implements a `RecoveryPolicy` interface; the harness
  hands it a failed episode and the replay environment and scores the outcome
  from a reward ledger the policy cannot read.
- **Leakage boundary:** gold labels, injection logs and rewards live only on the
  scoring side; tests check that no submission-side module imports them.

## 10. Dashboard

A single self-contained HTML file generated from scoring reports
(`python -m benchmark dashboard --runs <dirs> --out dashboard.html`):

1. corpus overview (counts by split, mode, domain; trace lengths; failure position);
2. attribution leaderboard with confidence intervals, confusion matrix and calibration;
3. recovery vs baselines, and recovery against cost;
4. trace explorer: steps with gold, predicted top 3, rollback point and irreversible steps;
5. per-trace results table with filters.

## 11. Repository layout (planned)

```
benchmark/
  env/          multi-agent retail workflow, Ollama client, response cache
  faults/       the 7 fault operators and their parameter spaces
  generate/     clean runs, eligibility, paired injection, causation check, labels
  corpus/       IR export, external importers, splits, manifest
  scoring/      metrics, statistics, baselines, ablation configs
  dashboard/    static HTML report generator
  taxonomy/     seven_modes.json, AgentRx mapping, decision guide
  tests/
docs/
  benchmark-design.md   this file
  datasheet.md          sources, counts, licences, limits (at release)
```

## 12. Milestones

| Days | Milestone |
|---|---|
| 1–2 | Contract: schemas, 7-mode spec, AgentRx mapping, mock data for other teams |
| 3–5 | Multi-agent workflow on Ollama, response cache, clean runs, eligibility filter |
| 5–8 | Fault operators, causation check, labels, 20-task pilot |
| 8–10 | Scoring, baselines, external importers, splits |
| 11–12 | Full generation run, dashboard, v1 release and datasheet |

## 13. Risks

- **Local compute.** About one task per minute on a 7B model; roughly 60 tasks
  × 7 modes × 2 variants is an overnight run. A 20-task pilot comes first.
- **Weak open models.** The pass^3 filter keeps labels clean but shrinks the
  task pool; we report the eligible fraction per model.
- **Realism of injected faults.** Checked against the natural-failure split and
  the external corpus.
- **Taxonomy boundaries.** Modes 4 and 5, and 1 and 2, overlap; the decision
  guide and `acceptable_steps` handle ambiguity, and agreement is reported.

## 14. Implementation status and deviations

Implemented as planned: environment, fault operators for all 7 modes, paired
generation with causation check and labels, corpus with splits and public/gold
separation, faultattribution export, external importers (AgentRx release,
Who&When), attribution/detection/recovery scoring with bootstrap CIs and McNemar,
baselines, recovery harness with budget caps, agent robustness metrics, the `evaluate`
suite, dashboard, CLI, 70 tests.

Deviations and open items:

- Workflow v2 (after the pilot's first tasks): the planner also has the read-only
  order/product tools, and specialists are told that authentication and user
  confirmation are already done by the planner. v1 caused artificial failures
  (unopenable orders at the planner, specialists refusing to act); the workflow
  version and a prompt digest are now part of the generation settings lock.
- No separate responder agent; the planner composes the final answer, and mode 7
  is injected there.
- τ-bench airline (held-out domain) is not implemented. The AgentRx Magentic-One
  traces imported as an external split serve as the out-of-domain check for now.
- Replay semantics are an explicit modelling choice (`faults/operators.py`).
  Agent faults are systematic: on replay they fire again at the same site
  unless guidance reaches the responsible agent at or before the faulty turn.
  Restarts, new seeds and notes to other agents do not remove them. The first
  version ("recurs only on an identical context") was too lenient on a real
  model and was replaced after the pilot. Environment faults persist per their
  transient/persistent setting.
- Drift and context-propagation operators have `once` and `persistent`
  variants; the pilot showed a 31B model routes around every once-only one.
- `skip_write` writes a specific, varied success claim built from the skipped
  call, instead of one canned sentence.
- Recovery budgets are relative by default (2× the original run's tokens).
- The faultattribution export labels the domain `tau` (that loader only accepts
  AgentRx domains) and maps the 7 modes to AgentRx categories, which is lossy.
- The reward ledger is hidden from policies by interface, not by a sandbox.
- Natural failures are exported for manual annotation (`natural-queue`); none
  are labelled yet.

## References

1. Barke et al. *AgentRx: Diagnosing AI Agent Failures from Execution Trajectories.* arXiv:2602.02475.
2. Jia et al. *MAS-FIRE: Fault Injection and Reliability Evaluation for LLM-Based Multi-Agent Systems.* arXiv:2602.19843.
3. Zhu et al. *When Tools Fail: Benchmarking Dynamic Replanning and Anomaly Recovery in LLM Agents* (ToolMaze). arXiv:2606.05806.
4. Chen et al. *Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems* (TraceElephant). ACL 2026.
5. Yao et al. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* ICLR 2025.
6. Zhang et al. *Which Agent Causes Task Failures and When?* (Who&When). ICML 2025, arXiv:2505.00212.
7. Cemri et al. *Why Do Multi-Agent LLM Systems Fail?* (MAST). NeurIPS 2025 Datasets and Benchmarks.
8. Zhang et al. *AgentForesight: Online Auditing for Early Failure Prediction in Multi-Agent Systems.* 2026 preprint.
