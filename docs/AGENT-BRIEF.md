# AgentReroute: brief for an AI assistant

Read this before changing anything. It states what the system does now, which design questions are
already settled by measurement, and which invariants will fail the test suite if you break them.

## What the project is

Given the trace of a failed agent run, name the step that caused the failure and the agent
responsible. Scoring is exact step match, no partial credit. The reference method is AgentRx, which
hands one model the whole trajectory and asks for a verdict in a single pass.

The benchmark is the AgentRx authors' published subset at `data/benchmark` (corpus hash
`a88bb715…`): 29 tau retail traces and 44 magentic web traces. We also generate our own corpus at
`benchmark/runs/corpus-v2`, used only for development. Never report a corpus-v2 number as a result;
see "Two corpora" below.

## Where the project stands

| domain | agentrx | ours, fixed checks | ours, written checks |
|---|---|---|---|
| tau, n=29 | 31.0% | 34.5% | **37.9%** |
| magentic, n=44 | 18.2% | **25.0%** | not reportable |

Both agentrx rows reproduce the published 31.03% and 18.18% to within 0.1 points, which is the
control that makes everything else meaningful. No difference against agentrx reaches significance:
one tau trace is 3.4 points. The one significant result is the ablation, where putting a model in
the decision path beats the linear policy 9 traces to 1, McNemar p = 0.0215.

Full evidence is in `results/`. Regenerate every table with:

```
python3 analyze_run.py results/final-evidence.csv
```

## The pipeline

`agentrx/pipeline.py:diagnose_mp_rl` is the entry point. In order:

1. **Three perspectives** (`agentrx/perspectives.py`). `P1_INTENT_PLAN`, `P2_TOOL_EXECUTION` and
   `P3_HANDOFF_GROUNDING` each own a slice of the taxonomy. Each contributes constraints: fixed
   seed checks always, plus model-written ones when `evidence` is true.
2. **Checking**. Every constraint is evaluated at every step against the observed prefix only.
   `constraints.prefix(trajectory, offset)` enforces that; a check can never see the future.
3. **Blame matrix** (`compute_blame_matrix`). `B[perspective][agent] = V / (V + S + eps)`. Note it
   is per agent, not per step, so on a single-agent trace it is a 3x1 matrix carrying no
   localisation signal at all.
4. **Candidates** (`encode_candidates`). One per (agent, step) carrying a violation, each with ten
   features in [0, 1].
5. **Decision**. Either the linear policy (`rl_policy.select_action`, softmax over `theta . h`) or,
   when `llm_selector` is set, one model call (`agentrx/selector.py:llm_select`).

## Settled by measurement: do not redo these

Each of these cost real time. The evidence is in `results/README.md`.

- **Training theta hurts.** Held-out n=100: untrained 44%, trained 36%, McNemar p = 0.0078, eight
  traces broken and none fixed. The REINFORCE reward ("violations fell after the repair") sat 17.6
  points above real recovery, so the policy learned to prefer easily repaired steps. The checkpoint
  in use is the untrained `THETA0`.
- **The repair-and-verify cascade changes nothing.** Published tau: 13.8% before and after. It
  moved six answers, all to another wrong step, and in the five traces where the true step sat at
  rank two or three it accepted that rank zero times. A model asked to rewrite any step produces
  one that satisfies more constraints, so the verifier's signal carries no information about
  whether the step was the fault. The code is kept because the negative result is reported;
  `configs/gemini_cascade_tau.json` reproduces it.
- **Accuracy is governed by pool recall, not by ranking.** Split by whether the true step is in the
  candidate pool: tau converts at 70% in-pool against 16% out, magentic 89% against 9%. Pool recall
  is 41.4% and 20.5%. Any further work on ranking is capped by those numbers.
- **The benchmark is single-agent where our corpus is not.** Published tau roles are
  `assistant`/`tool`/`user`/`system`; corpus-v2 uses `planner`/`returns_agent`/`order_agent`. The
  routing and handoff seeds therefore fire 268 and 104 times on corpus-v2 and 0 and 8 times on
  published. Every published tau candidate carries exactly one violation (130 of 130), against
  41.5% of corpus-v2 candidates carrying two or more. The features the policy ranks on are
  degenerate constants on the benchmark it is judged by.

The open direction follows from the third point: make constraint writing the action and pool recall
the reward. Recall is a hard observable rather than a proxy, and it carries about five times the
headroom of anything we tried.

## Invariants

Breaking any of these fails the test suite, and the first two are project rules rather than style.

- **Never read gold** in any decision, attribution, repair or update path. Not `ScoreLedger`, not
  `gold_state_hash`. `tests/test_seed_injector_leakage.py` and
  `recovery_orchestrator/tests/test_leakage.py` guard this; do not weaken or delete them.
- **Seed constraints must not key on injector wording.** They may read structure: counts, ordering,
  whether a value appeared in the prefix. A check that recognises our generator's phrasing scores
  on our corpus and stays silent on real traces.
- **Do not modify the baseline agentrx path** (`protocol` of `joint`, `step_then_category` or
  `ranked`) or its `complete()` flow. It is the control.
- `recovery_orchestrator` and `repair_strategies` must never import `agentrx`.
- API keys come from the environment only, never from a config file. `CLIENT_KEYS` in
  `agentrx/experiments.py` enforces it.
- `llm_selector` and `repair_cascade` both force `recovery_loop` to false, so theta is read and
  never written during inference.

## Running an experiment

```
python3 -m agentrx.cli run --corpus data/benchmark --config configs/gemini_final.json --out runs/final --preflight
python3 -m agentrx.cli run --corpus data/benchmark --config configs/gemini_final.json --out runs/final
python3 export_evidence.py runs/final results/final-evidence.csv
python3 analyze_run.py results/final-evidence.csv
```

Four configs survive, one per reported result: `gemini_final.json` (the headline),
`gemini_selector.json` (the ablations), `gemini_cascade_tau.json` (the cascade negative result) and
`gemini_magentic_synth.json` (the open item). The rest belong to the base project and are
referenced by `cli.py` and the tests.

Watch a run with `python3 watch_run.py runs/final --follow`.

## Traps that have already caught us

- **Rate limiting corrupts arms silently.** Gemini limits on input tokens per minute, so a run can
  sit at 2% of its request allowance and still be refused. When a selector call fails, the code
  falls back to the top-ranked candidate, which keeps the trace but makes the arm report theta's
  number instead of the selector's. One arm reported exactly 13.8% that way. `analyze_run.py`
  prints `DO NOT REPORT` for any arm whose selector failed on 20% or more of traces. Configs set
  `tpm_limit` so the client paces itself.
- **Synthesis fails silently too.** When synthesis calls fail, `synthesize_perspective_constraints`
  keeps the seed constraints, so the arm looks like a valid seeds-only row. The giveaway is pool
  size: roughly 1.2 candidates on magentic and 4.5 on tau for seeds, against 5 to 12 with synthesis
  working.
- **Experiment identity hashes the config, the prompts and every `.py` under `agentrx/`.** Change
  any of them and a resumed run is refused with "Experiment identity changed". That is the lock
  working; use a new output directory rather than editing stored identities.
- **Job artifacts are large.** Each `result.json` embeds every check row at roughly 800 KB, so a
  four-variant run over 73 cases is about 230 MB. `runs/` is gitignored for that reason; commit the
  distilled CSV from `export_evidence.py` instead.

## Two corpora

`data/benchmark` is the published subset and the only source of reportable numbers.
`benchmark/runs/corpus-v2` is ours, generated with the fault injector in `benchmark/faults/`, split
113 train / 43 calibration / 100 test for tau. The same code and weights score 44% on corpus-v2 and
13.8% on published, because our seed constraints detect the faults our own injector writes. Use
corpus-v2 for development and never quote it as a result.
