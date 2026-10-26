# AgentReroute: brief for an AI assistant

Dense orientation for a coding assistant joining this repo. Facts are measured unless marked
as an estimate. Figures carry their n. Read `docs/ONBOARDING.md` instead if you are a person.

Last updated: 2026-10-26. Branch `mp_rl-gemini-actor`.

---

## 1. What the project is

Given a multi-agent trajectory that failed, name the step that caused it and which agent owned
that step. Two systems are compared:

- **agentrx**: reconstruction of a published baseline. One LLM call per trace over the whole
  trajectory (`generation: oneshot`, `protocol: joint`). Published 31.03% step accuracy on tau
  and 18.18% on magentic using `gemini-3.1-flash-lite`.
- **mp_rl**: this project. Executable constraints over three perspectives produce candidate
  steps; a 10-weight linear-softmax policy ranks them; top-1 is the prediction.

Both live in one repo and share the taxonomy and trace format. **mp_rl does not call agentrx and
does not inherit its performance.** Do not describe it as "agentrx plus RL".

## 2. Layout

```
agentrx/          attribution pipeline (BOTH systems)
  pipeline.py       diagnose_mp_rl, diagnose (joint/agentrx), validate_config
  perspectives.py   seed constraints, checks, blame matrix, encode_candidates
  constraints.py    guard_matches, check_constraint, prefix (no look-ahead)
  rl_policy.py      DualLoopRLJudge: select_action, update_policy, THETA0
  rl_cross_branch.py  MPRLRecoveryPolicy, train_and_evaluate_benchmark
  providers.py      HTTPProvider / OllamaProvider / CachedProvider
benchmark/        generation, corpus building, scoring
  generate.py       episode generation with identity lock + resume
  corpus.py         build(), splits, generation_dir() resolver
  env/llm.py        actor clients, CachedLLM, ACTION_SCHEMA
  env/workflow.py   Workflow, UserSimulator, system_prompt, AGENTS
  env/retail.py     tau-bench retail tools (TOOL_SCHEMAS)
  scoring/recovery.py  run_recovery (shared by training and frozen eval)
repair_strategies/, recovery_orchestrator/   repair side; MUST NOT import agentrx
data/benchmark/cases.json   73 external traces (29 tau, 44 magentic) + gold
```

## 3. The attribution chain

With `evidence: False` this is **pure Python, zero model calls**. Verified: 16,325 checks across
80 traces, 100% `origin: seed`, `check_type: python_check`.

```
constraints -> per-step verdicts (SAT/VIOL/SKIP)
  -> compute_blame_matrix:  viol / (viol + sat + EPS) per (perspective, agent)
  -> encode_candidates:     one candidate per violating agent-owned step
  -> select_action:         softmax(theta . h / tau), top-1 = prediction
```

Perspectives: `P1_INTENT_PLAN`, `P2_TOOL_EXECUTION`, `P3_HANDOFF_GROUNDING`.
Seed constraints: 2 / 1 / 2 respectively, all `python_check`.

Feature vector, all in [0,1], order fixed:

```
f0..f2  blame under P1, P2, P3
f3..f5  violation density per perspective, min(1, violations/2)
f6      share of evidence from python checks, min(1, n/2)
f7      temporal position (1 - offset/T when violations exist, else offset/T)
f8      perspectives that flagged it, /3
f9      max density
```

`THETA0 = (0.85, 0.95, 0.90, 1.15, 1.35, 1.25, 0.85, 0.45, 0.90, 1.20)`

## 4. Two config flags that are constantly misread

**`evidence`** switches LLM constraint synthesis. `False` = 5 hand-written Python rules only.
`True` = `synthesize_perspective_constraints` writes trace-specific constraints, and `nl_check`
ones cost one judge call per guarded step. This changes candidate recall, so it changes accuracy.

**`recovery_loop`** does NOT change the prediction. In `diagnose_mp_rl` the prediction is written
before the branch, with the comment *"Recovery is auxiliary to diagnosis: its failure is recorded
but never voids the prediction."* `live_dynamic_recover` retries up to 3 times with feedback, but
it is repairing the step, not revising the blame. It exists to produce the RL reward.

Any claim that "the loop iterates until attribution is right" is wrong.

## 5. Measured results

### Attribution, mp_rl, seeds-only (`evidence: False`), THETA0

| set | n | accuracy |
|---|---|---|
| external tau | 29 | **13.79%** |
| external magentic | 44 | **20.45%** |
| corpus-v2 calibration | 43 | **51.16%** |
| corpus-v2 test | 100 | **44.00%** |

The external/corpus gap is a generalization finding: corpus-v2 faults were injected with
operators the seed constraints were written around.

### Candidate recall, the hard ceiling on any reranker

| | top-1 | acc@3 | recall | mean pool |
|---|---|---|---|---|
| tau | 13.8% | 31.0% | **41.4%** | 4.5 |
| magentic | 20.5% | 20.5% | **20.5%** | 1.2 |

Random choice within the tau pool is ~9.2%. magentic's three figures are identical because when
gold is present it is the only candidate, so **theta cannot move magentic at all**.

### Training outcome: significantly negative

1 epoch, 138 episodes, no synthesis, qwen2.5:32b, `runs/train-v2/rl_state_v2.json`.

```
corpus-v2 test, n=100
  THETA0   44/100 = 44.00%
  trained  36/100 = 36.00%     -8.00 pts
  discordant: THETA0-only-right 8, trained-only-right 0
  McNemar p = 0.0078
```

Mechanism, from the epoch summary: `acceptance_rate 0.565`, `harness_recovery_rate 0.389`,
`gap +0.176`. The policy optimised an observable acceptance signal sitting 17.6 points above gold
recovery. Weight changes: `blame P1` -82%, `temporal` -42%.

**Use THETA0 for reporting. The trained checkpoint is kept as evidence of the degradation.**

### agentrx on a matched local model

Published 31.03% was on `gemini-3.1-flash-lite`. On `qwen2.5:32b`, 3 of 3 traces returned
`prediction: None` (`Inconclusive`), no errors. Small n, incomplete run. The honest framing is
model-dependence, not "we beat it".

## 6. Corpus

`benchmark/runs/corpus-v2`, `gold_hash 75e8bc3f07771bbf77ed9cb483105a9c1a76370c5f4e4dcc19b0448c7b0f5017`

733 episodes over 115/115 tau tasks, 35 eligible (30%): 322 failed, 252 tolerated, 44
not_activated, 80 natural_failure, 35 clean. Splits are disjoint by task template, held-out fault
variants confined to test: train 233 (113 faults), calibration 93 (43), test 165 (100).
`degraded: 0`, `lenient_response: 0`.

**It cannot be regenerated identically.** Ollama's continuous batching makes concurrent requests
non-deterministic, so `temperature 0, seed 0` does not give bit-reproducibility.

Eligibility requires the clean run to pass (`generate.py` line ~194, `any(r['reward'] != 1 ...)`),
so `pass^k` gets *stricter* with more seeds. `--seeds 0` is the most permissive setting.

## 7. Invariants that must not be broken

- Gold labels, ScoreLedger and `gold_state_hash` must never be read in any decision, attribution,
  repair or **RL update** path. This is why training uses an observable proxy reward.
- `repair_strategies/` and `recovery_orchestrator/` must not import `agentrx`.
- Do not weaken `tests/test_seed_injector_leakage.py` or `recovery_orchestrator/tests/test_leakage.py`.
- Seed constraints execute as source in a sandbox (`agentrx/worker.py`): no ClassDef, no private
  names, module whitelist. They see `prefix(trajectory, offset)` only, so **no look-ahead**.
- Do not modify the frozen agentrx diagnosis path (`protocol` joint / step_then_category / ranked).

## 8. Gotchas that have already cost hours

**Schema dialects.** `ACTION_SCHEMA` in `benchmark/env/llm.py` is standard JSON Schema for
Ollama's `format`. `GEMINI_ACTION_SCHEMA` is derived by `gemini_schema()` because Gemini wants
upper-case types. Sending the Gemini form to Ollama gives
`JSON schema error at #: unrecognized type OBJECT` on every agent turn.

**num_ctx.** agentrx's one-shot prompt overflows 16384 on long traces. Use **32768**
(qwen2.5's trained maximum). VRAM on a 45 GiB A40: `18.5 + 0.25 * (num_ctx/1024) * slots` GiB,
so 32768 allows `OLLAMA_NUM_PARALLEL=2`, not 4.

**Which Ollama answered.** A local Windows Ollama on `127.0.0.1:11434` can win the port against
an SSH tunnel. `prepare_for_run` reports "model not found" only *after* a successful connection,
so that error means the wrong server answered.

**Missing checkpoint paths fail open.** `DualLoopRLJudge.load` returns `cls()` (THETA0) when the
file is absent. A stored result citing `runs/rl_frozen_copy.json`, which does not exist, was
silently untrained. Always verify the path exists.

**`runs/` is gitignored** (`.gitignore` line 9), matching `benchmark/runs/` too. Committed data
needed `git add -f`.

## 9. Commands

```bash
# generation, 4 parallel slices, resumable
python3 -m benchmark generate --data-dir benchmark/data/tau_retail --backend ollama \
  --model qwen2.5:32b --num-ctx 16384 --seeds 0 --max-tasks 115 --task-slice K/4 \
  --out benchmark/runs/gen-v2-K

python3 -m benchmark build-corpus --gen benchmark/runs/gen-v2-{0,1,2,3} \
  --out benchmark/runs/corpus-v2

# training: per-episode telemetry, resumable, optional early stop
python3 -m benchmark train --corpus benchmark/runs/corpus-v2 --out runs/train-v2 \
  --checkpoint runs/train-v2/rl_state_v2.json --backend ollama --model qwen2.5:32b \
  --num-ctx 32768 --epochs 1 --val-split calibration [--use-llm-synthesis] \
  [--stop-above X --val-every N] [--control-sample 25]

# external benchmark, both systems
python3 -m agentrx run --config configs/<cfg>.json --out runs/<name>
python3 watch_run.py runs/<name> --follow      # live accuracy, read-only

python3 -m unittest discover -s tests -t . && \
python3 -m unittest discover -s benchmark/tests -t .     # 343 tests
```

`AGENTRX_CHECK_WORKERS` (default 4) controls concurrent per-step judge calls. Set it to match
`OLLAMA_NUM_PARALLEL`.

## 10. Cost model, measured

| operation | cost |
|---|---|
| attribution, `evidence: False` | **0 model calls** |
| attribution, `evidence: True` | ~87 calls/trace on external tau (median 36 steps, 27 agent-owned) |
| training episode, no synthesis | ~50 s (replay only; prefix is cache-served) |
| agentrx trace, one-shot | ~80 s at num_ctx 32768 |

Training throughput rises during a run as the cache warms: 176 s/ep at episode 1, 47 s/ep by
episode 138.

## 11. Open questions

1. **Does synthesis raise recall above 41.4% on tau?** Unmeasured. It is the only lever that can
   lift top-1 past the ceiling. ~87 calls/trace, so 29 traces is about 3 hours at 2 slots.
2. **agentrx on a matched model, full run.** 3 traces is not a result.
3. **Reward proxy.** The +0.176 gap is the core negative finding. Fixing it means either a better
   observable verifier or accepting gold in the update path, which the project's invariant forbids.
4. **magentic needs constraint work, not ranking work.** Pool 1.2, recall 20.5%.

## 12. Things previously claimed that were wrong

Stated so you do not rediscover them:

- "Training makes attribution worse" was first asserted from a checkpoint trained on **16
  episodes** of a different environment. That inference was invalid. The claim is now supported by
  a proper run with McNemar p=0.0078, but the first version of it was not.
- "Effective sample size is ~5, two episodes did all the work" was wrong. `|dtheta|` was 0.376 at
  episode 71 and 0.838 at 138; theta kept learning throughout.
- The 13.79% figure was repeatedly quoted as "mp_rl" when it is mp_rl with synthesis **off**.
  Always state the `evidence` flag alongside any accuracy number.
