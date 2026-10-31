# Usage and implementation details

A runnable Python reconstruction of **AgentRx: Diagnosing AI Agent Failures from
Execution Trajectories** (arXiv:2602.02475v2). It diagnoses an already failed agent
run: global constraints → prefix-based dynamic constraints → Python/semantic
checks → evidence log → LLM attribution of the first **unrecovered** critical
failure, responsible agent, and category.

This project is an independent reconstruction, not Microsoft's original code.
See [attribution](../THIRD_PARTY_NOTICES.md).

## Start here

Python 3.10+; the runtime has **no third-party dependencies**. Commands work from
this project directory without installation:

```bash
python3 -m agentrx smoke --out runs/smoke
python3 -m unittest discover -v
```

The smoke test uses an explicitly synthetic count-mismatch example, fixed model
responses, real Python/semantic checker plumbing, two judging configurations,
and three repetitions. Its reports say **OFFLINE FIXTURE TEST — NOT BENCHMARK
RESULTS**. The fixture judge response is predetermined; the smoke test verifies
integration, not an LLM's diagnostic ability.

For the console command, optionally install in a local environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/agentrx --help
```

## Original benchmark data

```bash
python3 -m agentrx prepare --fetch --out data/benchmark
python3 -m agentrx audit --corpus data/benchmark
```

`--fetch` obtains the authors' repository at pinned revision
`7a18c79708e7671be15124460f4f7296107c2a55` into `data/reference`, then imports the
original data. Alternatively use `prepare --source /path/to/AgentRx --out data/benchmark`.
Fetch never overwrites an existing checkout with another revision.

| Domain | Paper trajectories | Paired traces in pinned GitHub release |
| --- | ---: | ---: |
| tau-bench retail | 39 | 29 |
| Magentic-One | 44 | 44 |
| Flash | 42 | 0 |
| RelWork | 45 | 0 |
| Total | 170 | 73 |

All 73 pairs were audited for root references, taxonomy spellings, and valid
step indices. Equal domain counts do not prove identical paper corpus membership.
The Hugging Face release requires authenticated access acceptance and currently
describes two splits; missing domains are not assumed to be there.

`manifest.json` records input hashes, source revision, corpus identity, all IDs,
excluded records, and domain coverage. The retail import explicitly uses
`tau_ground_truth.json` (29 annotations), not the separate 24-row file.
`cases.json` stores observable IR and gold labels separately. Model requests
receive allowlisted observable fields and the supplied task instruction, never
failure annotations, rewards, reference answers, or expected action sequences.

The source file `data/tool_schemas/tools_structure.json` contains example events,
not schemas. This project packages the actual tool definitions from the authors'
domain registry, with source attribution. Magentic receives its tool definitions
and no added domain policy in the paper setting.

## Model-backed pilot and benchmark runs

Configure credentials in your shell; `.env.example` lists the supported names.
The package does not automatically source `.env`. Keep keys out of JSON config
files and Git. Neither ChatGPT nor Codex sign-in supplies this package with an
API credential.

```bash
export OPENAI_API_KEY='your-local-api-key'
python3 -m agentrx run --corpus data/benchmark --config configs/pilot_gpt5.json --out runs/pilot --preflight
python3 -m agentrx run --corpus data/benchmark --config configs/pilot_gpt5.json --out runs/pilot
python3 -m agentrx run --corpus data/benchmark --config configs/paper_gpt5.json --out runs/gpt5
```

Preflight checks local inputs/configuration/credential presence and makes zero
network calls. It does not confirm model access or quota. The pilot selects one
retail trajectory and compares baseline with AgentRx. Full runs can make many
model calls, particularly prefix generation and semantic checking on long traces.

The paper matrix has six columns: baseline, step-then-category, baseline with
violations, step-then-category with violations, checklist, and checklist with
violations. Evidence-bearing columns run with both stepwise and one-shot
synthesis, giving nine distinct configurations. Judge-only columns are computed
once per repetition rather than billed again for each generation mode. Default
benchmark configurations request three independent repetitions.

Other configurations:

- `configs/dev_ollama.json`: local Ollama without credentials (`--model ollama`).
- `configs/dev_gemini.json`: Google Gemini via OpenAI-compatible endpoint (`--model gemini`).
- `configs/dev_openai.json`: OpenAI GPT-4o (`--model openai`).
- `configs/dev_multi.json`: unified configuration with `models` dictionary for quick switching.
- `configs/azure_gpt5.json`: Azure API key, endpoint, deployment, API version.
- `configs/paper_o3.json`: o3 ablation.
- `configs/paper_deepseek.json`: set `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, and
  optionally `DEEPSEEK_MODEL` for an endpoint actually serving **DeepSeek-V3.2**.
  A changing `deepseek-chat` alias is deliberately not assumed to be V3.2.

Model/deployment identity, actual returned model, timestamps, API settings,
usage, raw responses, and prompts are retained. Reconstruction token caps are
12,000 completion tokens (GPT-5/o3) and 8,192 (DeepSeek), with provider-default
sampling unless explicitly configured. These are reconstruction choices; the
paper does not fully specify sampling, prompt, or deployment snapshots. Context
and output overflows are errors, never silent truncation.

The GPT-5 token cap includes reasoning tokens. Increase it in a new configuration
if an endpoint returns `finish_reason=length`; failed calls are retained for audit.

## Resume, artifacts, and metrics

Rerun the same command with the same output directory to resume. Successful
requests and completed jobs are reused within that experiment. Repetitions have
separate cache namespaces, so one repetition cannot recycle another's stochastic
outputs. Changes to corpus, code, prompts, model settings, or endpoint require a
new output directory. Stored provider failures are retried; semantically invalid
successful JSON remains in its request record for inspection. To regenerate one,
remove that specific request JSON and rerun its failed job, or choose a new output
directory. Never edit cached responses and call them model measurements.

```bash
python3 -m agentrx evaluate --corpus data/benchmark --run runs/gpt5
```

A run contains:

- `experiment.json`: model/config/code/prompt/corpus identity and selected IDs.
- `requests/`: full prompt messages, raw responses, usage, timestamps, and errors.
- `jobs/<variant>/rep-<n>/<domain>/<id-hash>/`: normalized trace, constraints,
  all checks, violation-only log, localization, and final result.
- `summary.json`, `predictions.csv`, `confusion_matrices.json`.
- `table4.{md,csv}`: step/category accuracy and step distance by ablation.
- `table7.{md,csv}`: exact and ±1/±3/±5 step accuracy, agent accuracy, and
  critical/any/earliest/terminal category accuracy.

Accuracy denominators are the fixed intended case counts. Missing jobs, errors,
invalid predictions, and abstentions cannot improve accuracy by disappearing.
Mean absolute step distance uses valid localized predictions only and includes
its coverage denominator. Mean and sample standard deviation are computed across
repetitions. At tied earliest/terminal failure steps, any category at that boundary
matches. JSON retains numerators, coverage, errors, and abstentions; CSV accuracy
values are fractions, while Markdown displays percentages.

`references/paper_results.json` contains selected **published reference values**,
not measured predictions. This project does not claim the paper's accuracies were
reproduced. Original full data, accessible model endpoints, and completed live
experiments are required for scientific comparison.

## Diagnose one trajectory

```bash
python3 -m agentrx diagnose --input examples/count_mismatch.json --domain tau \
  --replay examples/replay.json --out runs/example
```

Remove `--replay` to use a model; optionally provide `--config`, `--variant`,
`--policy`, and `--tools`. Raw retail and Magentic logs and explicit normalized
IR are accepted. Flash/RelWork require normalized IR until representative native
files are supplied. Tool arguments preserve malformed strings as diagnostic evidence.

Canonical IR:

```json
{"trajectory_id":"example","domain":"flash","instruction":"Diagnose incident",
 "steps":[{"index":4,"substeps":[{"sub_index":1,"role":"KustoAgent","content":"timeout"}]}]}
```

Step indices may be sparse and are preserved; Python checks receive a zero-based
offset into the supplied prefix. Guards support integer/list/inclusive `a-b`
step selectors and regular expressions for role/tool/content, matched within
the same substep.

To add missing-domain original data, place `flash_cases.json` or
`relwork_cases.json` in the source data root. Each is a list of objects with
`trajectory` (normalized IR), `annotation` (the original failure list/root-cause
record), and optional `policy`/`tools`. Re-prepare into a new corpus directory.
Preserve the paper's top-level Flash troubleshooting steps and substeps.

## External baselines and MAST stress testing

Score an independently generated modified Who&When or other baseline with:

```bash
python3 -m agentrx import-predictions --corpus data/benchmark \
  --input baseline_predictions.json --source-name whowhen-modified --out runs/whowhen_metrics.json
```

Prediction format: a list of `{"domain":"tau","trajectory_id":"2","status":"ok",
"prediction":{"step_index":7,"agent":"assistant","category":"MisinterpretationOfToolOutput",
"rationale":"...","evidence_refs":[{"step_index":7,"substep_index":1}]}}`.
A step/agent-only baseline may omit category; category coverage is then explicit.
This imports predictions; it does not reconstruct or run the original W&W system.

MAST's 1,022-trace generalization experiment has no gold critical-failure labels.
Supply its original traces converted to a list of normalized IR objects:

```bash
python3 -m agentrx stress --input mast_ir.json --config configs/paper_gpt5.json --out runs/mast
```

The stress report contains category counts and diagnoses, with `gold_accuracy`
set to null. The original MAST files/native adapters are not bundled. An offline
fixture stress invocation is covered by the CLI tests.

## Checker behavior and reconstruction deviations

Applicable constraints return SAT or VIOL; inapplicable guards return SKIP.
Malformed checks, exceptions, timeouts, or invalid model output return ERROR.
Semantic rubric entries use CLEAR_PASS/CLEAR_FAIL/UNCLEAR: only a clear failure
creates a violation, and uncertainty remains visible. The root-cause judge sees
full context and may override noisy violation evidence or skip recovered errors.
Inconclusive is an abstention outcome in addition to the nine substantive labels.

Generated Python runs with a timeout, bounded output, restricted imports and
builtins, no model credentials, and OS limits where supported. This is isolated
execution **without a claim of a complete security sandbox**. Its approved
standard-library interface is narrower than unrestricted reference execution;
unsupported generated code is recorded as ERROR rather than repaired invisibly.
Imports expose explicitly approved exports through module facades. Raw module
internals, dynamic attribute helpers, frame/traceback introspection, and
`str.format`/`format_map` are rejected; generated checks can use f-strings.
No optional reflection/pruning or hidden examples are added to paper baselines.

Prompts are explicit reconstructions of the paper's procedures and schemas, using
the released taxonomy/checklist wording and tool descriptions. They are not
claimed byte-identical to the unpublished full experimental prompt/settings
snapshot. Global constraints are generated per domain/configuration/repetition;
stepwise dynamic constraints apply at their generation step, while one-shot
constraints retain their event guards.

## Ollama runtime notes

Use `--config configs/dev_ollama.json --model YOUR_INSTALLED_MODEL` to override the
model while preserving the native Ollama provider and its settings. A bare
`--model ollama` selects the default coding-model preset. Optional `model_env`
and `base_url_env` name environment overrides; missing explicit endpoint
variables fail instead of silently falling back. Actual runs resolve model
weights and server version before cache lookup; preflight makes no requests.

Native requests disable input truncation and context shifting. Context rejection
is an error requiring a larger `num_ctx` and a new run directory. This behavior
was tested with Ollama 0.40.2; compatibility with older servers is unverified.
All models in config must use `api_key_env` rather than storing literal keys.
See [the live verification record](verification.md#ollama-integration-verification--2026-10-18)
for the tested model and the distinction between pipeline completion and
diagnostic accuracy.
