# AgentReroute benchmark (KAN-4)

The shared ground truth for AgentReroute. It produces failed multi-agent runs with
exact gold labels for the project's **7 failure modes**, replays them for recovery
experiments, and scores fault-attribution and recovery systems against fixed baselines.
It builds on AgentRx and adds fault injection (MAS-FIRE), a tool-fault grid
(ToolMaze), fully observable replayable traces (TraceElephant) and state-based
outcomes (τ-bench). The design rationale is in [docs/benchmark-design.md](../docs/benchmark-design.md).

Python 3.10+, standard library only. Models run locally through [Ollama](https://ollama.com).

## Quick start (offline, about 30 seconds)

```bash
python3 -m benchmark smoke --out benchmark/runs/smoke
python3 -m unittest discover -s benchmark/tests -t .
```

`smoke` runs the whole pipeline with **scripted agents** on the bundled 8-task fixture:

1. generation
2. corpus build
3. baselines
4. scoring
5. recovery
6. dashboard

It writes `benchmark/runs/smoke/report.md` and `benchmark/runs/smoke/dashboard.html`. Scripted runs are
labelled `offline_fixture` everywhere. They verify the pipeline, not any model.

## Real runs with Ollama

```bash
python3 -m benchmark check-ollama --model gemma4:31b-cloud
python3 -m benchmark fetch-tau --out benchmark/data/tau_retail          # tau-bench retail data and tasks, pinned and hashed
python3 -m benchmark generate --config benchmark/configs/pilot_gemma.json --out benchmark/runs/pilot/gen
python3 -m benchmark build-corpus --gen benchmark/runs/pilot/gen --out benchmark/runs/pilot/corpus
python3 -m benchmark evaluate --corpus benchmark/runs/pilot/corpus --split test --model gemma4:31b-cloud \
    --cache-dir benchmark/runs/pilot-gemma-v2/llm_cache --out benchmark/runs/pilot/eval
```

`evaluate` runs the standard suite (attribution baselines including the LLM judge, all
recovery conditions with controls, McNemar against full restart, agent robustness) and
writes `report.md` and `dashboard.html`. The first real-model pilot findings are recorded in
`benchmark/tests/test_pilot_fixes.py`.

`generate` resumes where it stopped and retries Ollama timeouts, 429 and 5xx with
backoff. LLM responses are cached (`cache_dir`), which makes reruns and rollback replays
deterministic. Changing the model, seeds, temperature, step limit, workflow version,
agent prompts or operator version requires a new output directory (the settings lock
refuses to mix them). On an 8 GB Intel Mac a local 7B model is too slow for a pilot;
`gemma4:31b-cloud` (Ollama-served open weights) runs at about 35 s per run.

## Pipeline

```
tasks ──► clean runs (k seeds) ──► eligible if pass^k = 1 ──► for each fault variant × timing:
                │                                               inject at a site from the clean run
                └─ failing clean runs ─► natural-failure queue   │
                                         (manual labels)         ▼
                                              paired run ─► fired? diverged? ─► failed (labelled) / tolerated / not_activated
                                                                                         │
                         corpus: splits by task template, held-out variants in test only ◄┘
                         public traces (full + output_only) │ hidden gold │ faultattribution export
```

### Environment
- **Data and tools:** τ-bench retail data, tasks and tools, reimplemented in the standard library (`env/retail.py`).
- **Agents:** a multi-agent workflow (`env/workflow.py`):
  - a `planner` that talks to the user, looks up orders and products (read-only), gets confirmation and routes each change
  - three specialists: `order_agent`, `product_agent` and `returns_agent`

  Specialists start fresh on every handoff, so context that isn't passed is really lost.
- **Outcome:** τ-bench's reward. The final database must equal the database after running the gold actions, and every required output must appear in a reply to the user.

### Fault operators (`faults/operators.py`)

| Mode | Operator | Method | Variants |
|---|---|---|---|
| Instruction or plan adherence | `skip_write` | response rewrite | – |
| | `policy_contradiction` | prompt mutation | `no_writes`, `skip_verification` |
| Drift | `misroute` | message routing | `once`, `persistent` |
| | `irrelevant_tool` | response rewrite | `once`, `persistent` |
| Cycles | `stuck_processing` | tool perturbation | – |
| Tool failure | `tool_error` | tool perturbation | `explicit_transient`, `explicit_persistent` |
| | `stale_output` | tool perturbation | `silent_transient`, `silent_persistent` |
| Misinterpretation of tool output | `misread_value` | response rewrite | `argument`, `claim` |
| Context propagation | `drop_context` | message manipulation | `drop_ids`, `truncate` × `once`, `persistent` |
| Missing details | `omit_detail` | response rewrite | – |

Each variant is injected early, in the middle, or late, at sites taken from the clean
run. Labels are produced as the fault is injected:

- `fault_step`: where the fault entered.
- `critical_step`: AgentRx's first unrecoverable failure, meaning the agent step that should have handled it.
- `responsible_agent` and `mode`.
- `acceptable_steps`: other defensible answers.

The decision rules for each mode are in `resources/seven_modes.json`.

### Replay semantics (recovery)
- **Agent faults** (rewrites, prompt mutation, routing, handoff manipulation) model a systematic mistake. On replay they fire again at the same site, whatever else changed: a new seed, a full restart, or a note to a different agent. The only thing that removes one is a `guidance` directive delivered **to the responsible agent at or before the faulty turn**, while the note is still in that agent's context (for a specialist, the same handoff conversation). This is what makes attribution matter for recovery. (The pilot showed that the earlier rule, "recurs only on an identical context", let any upstream change remove the fault.)
- **Persistent agent faults** fire at the site and at every later matching event, for example a planner that keeps misrouting. A once-only fault is often caught and repaired by a strong model on its own.
- **Environment faults** (tool errors, stale data, stuck tools) belong to the world. They fire again on replay with their transient or persistent behaviour.
- **Prefix cost:** steps before the rollback point come from the cache and cost no new tokens.

## Data contracts

| File | Side | Content |
|---|---|---|
| `public/<view>/<split>/<tid>.json` | public | Trace in the faultattribution format (steps → substeps). `full` adds each agent's input and state version |
| `public/<split>.json` | public | Trajectory ids per split |
| `gold/<split>.json` | scoring | `status`, `critical {step_index, agent, mode}`, `fault_step`, `acceptable_steps`, `injection` |
| `gold/episodes.json` | scoring | Maps each trajectory to the generation episode the recovery harness replays |
| `agentrx/<split>/cases.json` | FA team | Corpus that loads with `agentrx`'s `load_corpus`. Domain is `tau`; categories are mapped from the 7 modes (lossy) |

**Submission format** (attribution and, optionally, detection):

```json
{"system": "my-fa", "view": "output_only", "predictions": [
  {"trajectory_id": "tr-…", "alarm_step": 7,
   "hypotheses": [{"step_index": 9, "agent": "returns_agent", "mode": "tool_failure", "confidence": 0.7, "rationale": "…"}]}]}
```

Up to 3 hypotheses per trajectory. `alarm_step` (optional) is the step where an online detector first raises an alarm.

**Recovery policy interface** (orchestrator team): subclass `scoring.recovery.RecoveryPolicy` and implement `recover(env)`. Inside it:
- Read `env.trace`, `env.task_view`, `env.hypotheses` and `env.remaining()`.
- Call `env.replay(Directive(rollback_step, kind='guidance'|'resample'|'none', text))` or `env.restart()`.
- Return `{'delivered': rollout_id or None, 'status': ...}`.

Rewards go to a ledger that only the harness reads. The per-episode token cap defaults to 2× the original run's tokens (`--token-multiple`), so one full restart always fits; `--max-tokens` sets a fixed cap.

## Commands

```bash
python3 -m benchmark baseline --corpus C --split test --method heuristic --out sub.json   # random | last_step | heuristic | llm_judge | step_by_step
python3 -m benchmark from-agentrx-run --run-dir ../fa/runs/x --out sub.json               # faultattribution output -> submission
python3 -m benchmark score --corpus C --split test --submission sub.json --out score.json
python3 -m benchmark recover --corpus C --split test --policy fixed_rules --predictions sub.json --out rec.json
python3 -m benchmark recover --corpus C --split test --policy fixed_rules --oracle --out rec-oracle.json
python3 -m benchmark report --attribution score*.json --recovery rec*.json --out report.md
python3 -m benchmark dashboard --corpus C --attribution score*.json --recovery rec*.json --out dashboard.html
python3 -m benchmark evaluate --corpus C --split test --model gemma4:31b-cloud --out eval/         # all of the above at once
python3 -m benchmark robustness --gen benchmark/runs/pilot/gen --out robustness.json             # agent robustness per fault
python3 -m benchmark natural-queue --gen benchmark/runs/pilot --out benchmark/runs/annotation        # natural failures for manual labels
python3 -m benchmark build-corpus --gen benchmark/runs/pilot --natural-labels benchmark/runs/annotation/labels.json --out C
python3 -m benchmark import-agentrx --cases ../fa/data/benchmark/cases.json --corpus C   # external split
python3 -m benchmark import-whowhen --dir path/to/Who_and_When --corpus C                # external split
```

**Recovery policies included:**
- `no_recovery`
- `full_restart`
- `retry_last_step` (capped)
- `generic_reflection`
- `fixed_rules`: fault-aware. It uses `--predictions`, or the gold labels with `--oracle` as the upper bound. `--min-confidence` makes it abstain on weak hypotheses (tune it on the calibration split).

## Metrics

| Layer | Metrics |
|---|---|
| Attribution | Step accuracy (exact, ±1, ±3, acceptable, fault step), top-3 step, agent, mode, joint, 7-mode macro-F1, step ECE, confusion matrix, breakdowns by mode, method, level and timing |
| Detection | Precision, recall, false-alarm rate on controls, premature alarms, mean delay |
| Recovery | Recovery rate, conditional recovery, intervention rate, harmful- and false-intervention rates on controls, collateral writes, tokens per episode and per recovery, cost as a multiple of one run, attempts, p95 time, budget breaches |
| Robustness | Per fault variant and mode: robustness score (MAS-FIRE RS), ToolMaze cells P1–P4 with success rate (= PRR here), recovery cost RC and the explicit − implicit gap |
| Statistics | Bootstrap 95% CIs (seeded), exact McNemar for paired policy comparisons |

## Layout

```
benchmark/
  cli.py, __main__.py   commands
  taxonomy.py           7 modes, AgentRx/MAST mappings      resources/seven_modes.json
  schema.py             trace / gold / submission validation
  env/                  retail.py (data+tools), tasks.py (fixture, tau-bench fetch), llm.py (Ollama, cache),
                        workflow.py (agents, hooks, directives, trace export), scripted.py (offline fixture agents)
  faults/operators.py   fault operators, site enumeration, replay semantics
  generate.py           eligibility, paired injection, causation check, labels
  corpus.py             splits, public/gold separation, faultattribution export, annotation queue
  importers.py          AgentRx release and Who&When as external splits
  scoring/              attribution.py, stats.py, baselines.py, recovery.py, robustness.py, report.py
  evaluate.py           the standard evaluation suite (used by `evaluate` and `smoke`)
  dashboard.py          self-contained HTML dashboard
  configs/              generation configs
  tests/                70 unit and integration tests
```

## Limits (current version)
- **Small real-model sample.** One 20-task pilot on `gemma4:31b-cloud` (early timing only); see the pilot report. The full 115-task run has not been done.
- **Held-out domain (τ-bench airline) isn't implemented.** The AgentRx Magentic-One traces (external split) are the only out-of-domain test for now.
- **The ledger is hidden by convention, not by a sandbox.** It isn't exposed as an attribute of the environment, but a Python policy could still reach it through private fields.
- **Tools are reimplemented.** τ-bench retail tool logic is rewritten here, so outcomes are consistent within this benchmark but not bit-identical to τ-bench.
