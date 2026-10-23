# Real-trace evaluation: `agentrx` vs `mp_rl` on the AgentRx release

Date: 2026-10-23. Branch: `mp_rl-real-trace-eval` (from `mp_rl` at `e96aed0`).
Result files: [`checkpoints/results/agentrx73/`](../checkpoints/results/agentrx73/).

## 1. Purpose and scope

The team's own benchmark (`benchmark/`) is generated, and its failures are injected.
This document records a test of both fault-attribution systems on a benchmark we did not
build: the human-annotated failures released with the AgentRx paper.

* Question: does `mp_rl` localise real failures, and how does it compare with `agentrx`
  on exactly the same traces?
* Scope: **attribution only** (which step, which agent, which category). The traces are
  recordings and cannot be replayed, so repair, rollback and recovery are not tested here.

## 2. Setup

### 2.1 Data

| Item | Value |
|---|---|
| Source | AgentRx release (Microsoft), imported with `python3 -m agentrx prepare --fetch` |
| Traces | 73: 29 tau-bench retail and 44 Magentic-One |
| Labels | Human-annotated critical failure step, agent and category |
| Not available | Flash (42) and RelWork (45); the paper's tau set has 39 traces, the public release 29 |
| Licence | Dataset CC BY 4.0; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) |

The 73 traces are used for testing only. Nothing in this run was trained or tuned on them.

### 2.2 Systems

| System | Configuration |
|---|---|
| `mp_rl` | Seed checks only (`evidence: false`, LLM synthesis off). Weights frozen from [`checkpoints/rl_state.json`](../checkpoints/rl_state.json) (76 training episodes, see [evaluation-and-hld.md](evaluation-and-hld.md#5-training-and-validation-procedure)). No LLM completions are requested. |
| `agentrx` | Our reconstruction: full pipeline, `joint` protocol, **one-shot** constraint generation, `evidence` and `checklist` on. LLM: `gemini-3.1-flash-lite` (Gemini API, free tier), temperature 0, `max_tokens` 16000, 6 retries. |

Configs are saved in [`configs/`](../checkpoints/results/agentrx73/configs/).

### 2.3 Metrics

Computed by `agentrx/evaluation.py`, as in the paper's Table 7.

* **Exact step**: predicted step equals the annotated critical step.
* **Acc@k** (k = 1, 3, 5): predicted step is **within k steps** of the critical step.
* **Average step distance**: mean absolute distance, over valid predictions only.
* **Agent accuracy**: predicted agent equals the annotated agent.
* **Critical category accuracy**: predicted category equals the critical failure's category.

Failed or missing jobs stay in the denominator and count as wrong, except for average
step distance (computed over valid predictions).

## 3. Results

Same 73 traces for both systems. Full tables:
[`agentrx`](../checkpoints/results/agentrx73/agentrx_gemini-3.1-flash-lite/table7.md),
[`mp_rl`](../checkpoints/results/agentrx73/mp_rl_seeds_frozen_checkpoint/table7.md).

| Metric | tau: `agentrx` | tau: `mp_rl` | Magentic: `agentrx` | Magentic: `mp_rl` |
|---|---:|---:|---:|---:|
| Traces (N) | 29 | 29 | 44 | 44 |
| Exact step | **31.0%** (9/29) | 6.9% (2/29) | 18.2% (8/44) | 2.3% (1/44) |
| Acc@1 | 34.5% | 6.9% | 27.3% | 2.3% |
| Acc@3 | 44.8% | 27.6% | 34.1% | 22.7% |
| Acc@5 | 51.7% | 34.5% | 34.1% | 29.5% |
| Average step distance | 7.0 | 12.4 | 30.2 | 18.2 |
| Agent accuracy | 65.5% | 72.4% | 54.5% | 47.7% |
| Critical category accuracy | 10.3% | 3.5% | 25.0% | 18.2% |

95% Wilson intervals for exact step: tau `agentrx` 17-49% and `mp_rl` 2-22%;
Magentic `agentrx` 10-32% and `mp_rl` 0-12%.

### 3.1 Trivial baseline

An answer of "the last step of the trace" needs no analysis. The annotated critical step is
the final step in 1 of 29 tau traces and 9 of 44 Magentic traces, so that baseline scores
3.4% on tau and **20.5% on Magentic**. The Magentic exact-step scores of both systems are at
or below it (see section 4.3).

### 3.2 Reference: the paper

AgentRx paper (Table 3 and 7, `references/paper_results.json`): exact step 42.7% (tau, 39
traces) and 36.4% (Magentic-One, 44 traces); agent accuracy 66.7% and 71.2%; critical
category accuracy 47.0% and 46.2%. These came from a stronger LLM and a larger tau set, so
they are context only and not a like-for-like target.

## 4. Analysis

### 4.1 Why `mp_rl` scores low

1. **The seed rules do not fit these traces.** They are deterministic checks authored
   against the team's own benchmark (repeated messages, refusal phrases, tool-error
   status, claims of writes), and the real failures here are mostly of other kinds, such
   as a wrong tool argument or a misread tool output.
2. **A fallback to the earliest step.** When no check finds a violation (7 of 29 tau
   traces, 29 of 44 Magentic traces), candidates are ranked by earliness alone, so the
   earliest agent action wins. `mp_rl` answered step 2 or 3 in 7 of 29 tau and **30 of 44**
   Magentic traces. The annotated step has median 21 (tau) and 10.5 (Magentic).
3. **False alarms on harmless signs.** A tool error that the agent immediately recovers
   from still triggers the tool-error check on the calling step.
4. **No schema check.** There is no rule that compares a tool call's arguments with the
   tool's schema, which is where many tau failures start.

Where the rules do fire on tau, the answer is on average 12.4 steps from the annotated one,
against 7.0 for `agentrx`.

### 4.2 Worked example: tau trace 34 (30 steps)

The user asks to remove some items from a pending order.

* Step 5: the agent looks up the user by email; step 6 returns "user not found"; step 7
  retries by name and zip and succeeds. The error was recovered.
* **Step 17 (annotated critical step, `InvalidInvocation`)**: the agent calls
  `modify_pending_order_items` with argument names that do not match the tool's schema.
  Step 18 returns an error and the rest of the run flounders.

| System | Answer | Correct step? |
|---|---|---|
| `agentrx` | step 17, `InvalidInvocation`; rationale cites the wrong argument names against the schema | yes |
| `mp_rl` | step 5, `InvalidInvocation`; the tool-error check fired on the recovered error | no |

Both gave the right category, but exact-step scoring counts only step 17.

### 4.3 What the `agentrx` scores show

* **tau**: 9 correct steps, of which 1 involves the last step. Against 3.4% for the trivial
  baseline, 31.0% reflects real localisation.
* **Magentic**: `agentrx` answered "the last step" in **25 of 44** traces, and **6 of its 8**
  exact hits are traces whose annotated step is the last one. Its 18.2% is below the 20.5%
  trivial baseline. Its average step distance (30.2) is also larger than `mp_rl`'s (18.2),
  because on long traces a last-step guess lands further from the annotated step than an
  early-step guess does. We do not read this score as evidence of localisation.

## 5. Checkpoint used

`checkpoints/rl_state.json` holds the 10 ranker weights and the prompt memory after 76
training episodes. The run copies it before use, because evaluation runs write back to the
file they load. Procedure and validation: see
[evaluation-and-hld.md, section 5](evaluation-and-hld.md#5-training-and-validation-procedure).

## 6. Limitations

* Small samples and wide intervals (section 3); one run, one LLM, one repetition.
* The Magentic results are at or below a trivial baseline for both systems.
* The LLM is a lite model, and constraint generation is one-shot, so `agentrx` is weaker
  than in the paper. Its numbers should not be compared with the paper's.
* 5 of 73 `agentrx` jobs failed and count as wrong: tau 32 and 38 ("Inconclusive
  prediction needs custom_category"), and Magentic `1f975693...`, `2ddae3b7...`,
  `a3fbeb63...` (two Gemini HTTP 503 errors and one "agent absent from trajectory").
* Check quality is low: of the generated checks, 81% on tau and 87% on Magentic were
  skipped as inapplicable, and 453 Magentic checks errored with `KeyError` (generated code
  indexing by step number).
* Attribution only: no repair or recovery evidence from this benchmark.
* 73 of the paper's 170 traces; the tau subset has 29 instead of 39.

## 7. How to reproduce

From the repository root, with Python 3.10+.

```bash
# 1. Data (73 traces)
python3 -m agentrx prepare --fetch --out data/benchmark
python3 -m agentrx audit --corpus data/benchmark

# 2. mp_rl, seed-only, frozen checkpoint
#    An Ollama server with the model in the config must be reachable (it is only used
#    to record the model's identity; no completions are requested).
cp checkpoints/rl_state.json runs/rl_frozen_copy.json
python3 -m agentrx run --corpus data/benchmark \
  --config checkpoints/results/agentrx73/configs/mp_rl_seeds_frozen_checkpoint.json \
  --out runs/agentrx73_seeds_ckpt

# 3. agentrx with an LLM (set the key in your shell first; do not commit it)
export GEMINI_API_KEY=...        # your own key
python3 -m agentrx run --corpus data/benchmark \
  --config checkpoints/results/agentrx73/configs/agentrx_gemini-3.1-flash-lite.json \
  --out runs/gem_agentrx73
```

Runs resume if repeated with the same command and output directory. Results land in
`predictions.csv`, `table4.md`, `table7.md` and `summary.json` in the output directory.
`runs/` is git-ignored; the files kept for this report are in
`checkpoints/results/agentrx73/`.
