# Results

Evidence for the reported numbers. The per-job artifacts under `runs/` are ~800 KB each
(`result.json` embeds every check row), so a 4-variant run over 73 cases is ~230 MB and is not
committed. These CSVs carry exactly the fields the analysis uses, so every table below is
reproducible from this directory alone:

```
python3 analyze_run.py results/final-evidence.csv
```

Regenerate a CSV from a run directory with `python3 export_evidence.py runs/final results/final-evidence.csv`.

## What is reportable

Benchmark: the AgentRx authors' published subset (`data/benchmark`, corpus_hash `a88bb715…`),
29 tau and 44 magentic cases. Model: `gemini-3.1-flash-lite`, temperature 0.

| domain | variant | step accuracy | 95% CI | vs agentrx |
|---|---|---|---|---|
| tau (n=29) | agentrx | 31.0% | [17.3, 49.2] | reference, reproduces published 31.03% |
| tau | seeds_select | 34.5% | [19.9, 52.7] | +3.4, p=1.0000 |
| tau | **synth_select_p10** | **37.9%** | [22.7, 56.0] | **+6.9, p=0.6875** |
| magentic (n=44) | agentrx | 18.2% | [9.5, 32.0] | reference, reproduces published 18.18% |
| magentic | **seeds_select** | **25.0%** | [14.6, 39.4] | **+6.8, p=0.2500** |

Both controls reproduce the published figures to two decimals, which is the check that the
measurement is sound. No difference reaches significance: at n=29 one tau trace is 3.4 points.

## What is NOT reportable

Two arms were corrupted by provider rate limiting (HTTP 429 on tokens-per-minute). The selector
falls back to the theta top-1 ranking when its call fails, which keeps a prediction at runtime but
means a rate-limited arm reports a plausible wrong number instead of failing loudly:

| arm | selector calls failed | what the row actually measures |
|---|---|---|
| `synth_select_p5` tau | 29/29 | theta ranking, hence exactly 13.8% |
| `synth_select_p5` magentic | 19/19 (of 44 attempted) | theta ranking |
| `synth_select_p10` magentic | 24/44 | a 55/45 mixture of theta and the selector |

`analyze_run.py` now flags any arm whose selector failed on 20% or more of traces. Do not quote
those rows. A clean magentic synthesis number needs a rerun with `tpm_limit` set.

## Ablations (`selector-evidence.csv`, same 73 cases, all clean)

Isolates what the LLM selector contributes over the linear policy, and what the candidate pool
costs when the selector is confined to it.

| domain | variant | accuracy | what it isolates |
|---|---|---|---|
| tau | mp_rl_top1 | 13.8% | theta ranking alone, zero model calls |
| tau | mp_rl_select_pool | 24.1% | selector confined to the candidate pool |
| tau | mp_rl_select_trace | 34.5% | selector free to pick any step |
| magentic | mp_rl_top1 | 20.5% | theta ranking alone |
| magentic | mp_rl_select_pool | 20.5% | pool size is 1.23, so there is nothing to choose between |
| magentic | mp_rl_select_trace | 25.0% | selector free to pick any step |

Putting a model in the decision path beats theta ranking significantly: 9 wins to 1 across both
domains, McNemar p=0.0215. Confining it to the pool costs 10.4 points on tau, which is the pool
recall ceiling (41.4%) binding.

## The mechanism behind the numbers

Accuracy tracks candidate-pool recall, not selection quality. The selector is strong when the true
fault is in the pool and weak when it is not (tau 67-71% against 7-12%; magentic 89% against 9%).

- Published tau is **single-agent** (`assistant`/`tool`/`user`), while the training corpus is
  multi-agent (`planner`/`returns_agent`/`order_agent`). Two of five seed constraints are
  structurally inapplicable there: routing alignment fires 268 times on corpus-v2 train and 0 on
  published; handoff grounding 104 and 8.
- So every published tau candidate carries exactly one violation (130/130), against 41.5% of
  corpus-v2 candidates carrying two or more. The features theta ranks on are degenerate constants.
- Synthesis raises tau recall 41.4% -> 48.3% while conversion holds at ~71%, which is the +3.4
  points of `synth_select_p10` over `seeds_select`. On magentic it raises recall 20.5% -> 38.6% but
  conversion falls to 53%, and that arm is tainted anyway.

## Negative results worth reporting

- **theta training hurt.** Test split n=100: THETA0 44%, trained 36%, McNemar p=0.0078, 8
  discordant pairs to 0. Reward hacking measured at +17.6 points of acceptance over real recovery.
- **The repair-and-verify cascade did nothing.** Published tau n=29, 13.8% -> 13.8%. It moved 6
  answers, all to another wrong step, and accepted the gold rank in 0 of the 5 traces where gold
  sat at rank 2-3. "Violations fell after repairing step X" is not evidence that X was the fault.
- **Lesson-bank training was not run to completion** and is not claimed. The bank built from 113
  training episodes produced 45 lessons with no support above the category level.
