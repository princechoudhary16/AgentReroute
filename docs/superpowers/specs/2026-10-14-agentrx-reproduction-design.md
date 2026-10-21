# AgentRx benchmark reproduction design

## Objective and status

Rebuild the fault diagnosis implementation in *AgentRx: Diagnosing AI Agent
Failures from Execution Trajectories*, arXiv:2602.02475v2, as provided in
`/Users/badraldhalaan/Downloads/agentrx.pdf`. The user explicitly wants reproduction
of the paper's benchmarks. The target is diagnosis of **already failed** recorded
agent runs, rather than estimating whether a live agent will succeed.

The user approved implementation on 2026-10-14. The implementation and offline
verification are complete; live model-backed benchmark measurements remain
pending credentials and missing original data. The workspace was initially empty;
implementation is preserved on the `implement/agentrx-reproduction` Git branch.

## Evidence and reproducibility limits

Reference code: https://github.com/microsoft/AgentRx, inspected at commit
`7a18c79708e7671be15124460f4f7296107c2a55` in a temporary checkout.
Dataset: https://huggingface.co/datasets/microsoft/AgentRx.

The paper specifies 170 annotated trajectories: 39 tau-bench retail, 42 Flash,
44 Magentic-One, and 45 RelWork. Its default model is GPT-5; each experimental
configuration is run three times. Additional experiments use o3 and DeepSeek-V3.2,
and a MAST generalization experiment uses 1,022 traces without this benchmark's
critical-failure gold labels.

The inspected code release contains:

| Domain | Paper count | Usable labeled pairs found | Limitation |
| --- | ---: | ---: | --- |
| tau-bench | 39 | 29 | `tau_ground_truth.json` pairs with 29 failed traces; another annotation file has only 24 records |
| Magentic-One | 44 | 44 | All 44 annotated IDs have matching trace files |
| Flash | 42 | 0 | No benchmark trace/annotation files found |
| RelWork | 45 | 0 | No benchmark trace/annotation files found |

These 73 pairs are an available release subset, not the paper's full benchmark.
The Hugging Face page requires authenticated acceptance of access conditions and
currently describes only retail and Magentic-One splits. Do not assume gaining
access supplies the missing domains. Never invent examples or silently substitute
newly generated trajectories for the original benchmark.

No OpenAI, Azure model, or Hugging Face credential variables were present in the
environment during inspection. Live runs require a configured endpoint; access
acceptance must be handled by the account owner. Secrets belong in local
environment configuration, never committed files or experiment artifacts.

## Implementation choices proposed by the assistant

Use a fresh, testable Python package in this workspace, taking the paper as the
method specification and the pinned authors' implementation as a reference.
Retain authorship/license notices for any reused prompts, policies, schemas,
or data. Do not present this rebuild as the authors' original code.

Provide a CLI, JSON experiment configurations, and JSON/CSV/Markdown reports.
Use environment-configured OpenAI-compatible and Azure endpoints, with GPT-5 as
the paper default. A deterministic fake provider supports offline integration
tests; its outputs must be visibly marked test fixtures and never benchmark
measurements. No web app or training pipeline is needed.

## Data ingestion and separation of labels

1. Import data from a pinned reference release or user-supplied benchmark root.
   Create a manifest recording source revision, file hashes, domain, trajectory
   IDs, annotation source, and expected versus available counts.
2. Explicitly select `tau_ground_truth.json` for the available retail subset;
   never merge conflicting annotation files automatically.
3. Normalize retail and Magentic logs into the paper's step/substep IR; support
   explicitly supplied Flash and RelWork IR through validated imports. Flash
   top-level troubleshooting steps retain their substeps. Native RelWork formats
   cannot be claimed supported before representative files are available.
4. Preserve original benchmark step numbers and agent names. Record mappings
   between raw positions, benchmark indices, and zero-based Python offsets.
   Validate every gold step against the normalized trajectory before scoring.
5. Join root causes by `root_cause.failure_id`, not by the first annotated event.
   Canonicalize known category and agent spelling variants with explicit maps;
   reject unknown or ambiguous labels rather than guessing.
6. Keep failure annotations, summaries, rewards, reference answers, and expected
   action sequences outside all model inputs. Retail `info.task.actions`,
   `info.task.outputs`, and reward metadata are evaluation information. Preserve
   the reference implementation's supplied task instruction I where present,
   recording its source explicitly; its extractor can use the retail task's
   `instruction` field, which is distinct from the ground-truth actions/answers.
   Do not pass the whole task object to a model. A trace-only task-context variant
   may be useful but must be labeled as a deviation from that reference behavior.

## Fault diagnosis pipeline

### Global constraints

Synthesize global constraints once per domain, model configuration, and repetition
from tool schemas and optional domain policy. Use the retail policy; Flash and
Magentic have no external policy in the paper's main setting. Do not add the
repository's optional Magentic policy to that setting. Store the generated
constraints and their provenance for reuse within that repetition.

### Dynamic constraints

Implement both modes in Table 4:

- Step-by-step: synthesize local constraints from the task, global store, and
  prefix through the current step only. Neither synthesis nor checking may read
  future steps in this mode. At step k, evaluate the global constraints and the
  local constraints generated for k; do not accumulate all prior local stores
  unless an explicitly named experimental variant requests it.
- One-shot: synthesize trajectory-specific constraints from the full trace and
  retain their triggers. Label this mode explicitly because it sees future events.

Use the Appendix F schema: assertion name, taxonomy targets, constraint type,
step/substep/role/tool/content guards, check hint, examples, and either a generated
Python check or a semantic check with rubric and prompt templates. Accept documented
reference aliases, then validate a single internal schema. Support SCHEMA,
PROTOCOL, RELATIONAL_POST, PROVENANCE, TEMPORAL, CAPABILITY, and ANY types.

### Guarded evaluation and evidence

Guards yield SKIP when not applicable. Applicable checks yield SAT or VIOL,
with exact event references and evidence. Record checker/parser/time-out failures
separately as ERROR; they must never become invented constraint violations.

Run generated Python in a time-limited subprocess with bounded output, a small
approved standard-library interface, and no model credentials in its environment.
Reject unsupported code explicitly. This is execution isolation, not a claim of
a complete security sandbox; document compatibility restrictions as deviations.
Pass the correct zero-based offset into generated functions while preserving the
original step index in evidence.

For semantic checks, preserve the Appendix F rubric behavior: CLEAR_PASS,
CLEAR_FAIL, and UNCLEAR. Discard UNCLEAR criteria; only definitive CLEAR_FAIL
criteria produce a violation. Retain uncertainty in telemetry even when the
paper's aggregate semantic verdict is pass.

Persist all check outcomes, a violation-only validation log, raw model responses,
prompt versions, usage counts, latency, and error records. Never hide schema
failures, truncate traces silently, or treat missing results as successful checks.

### Root-cause judge

The judge sees the full observable trajectory, task, selected taxonomy/checklist,
and the validation log when enabled. It selects the earliest **unrecovered**
causally decisive failure, not mechanically the earliest logged violation.
It may reject false-positive or downstream violation evidence.

Use the nine substantive categories in Section 2.2 and Appendix A, plus
Inconclusive as an abstention outcome. Return the original step index, failed
agent, category, evidence references, and rationale; an inconclusive result must
explain why evidence is insufficient. Implement both joint All-at-Once and
Step-then-Category protocols from Appendix B. Do not add few-shot examples to a
baseline unless the configuration and report explicitly identify that variant.

## Experiment matrix

Provide configurations for the six Table 4 columns: Baseline, Step-then-Category,
Baseline+Violations, Step-then-Category+Violations, Taxonomy Checklist, and
Checklist+Violations. Cross evidence-bearing configurations with step-by-step
and one-shot generation; duplicate judge-only configurations need not be billed
twice but must have explicit shared provenance in the reports.

The primary run is GPT-5 on all available original benchmark data, three independent
repetitions per configuration. Request caching and resumable checkpoints are
scoped by corpus hash, prompt/config hash, stage, trajectory, and repetition.
Repetitions cannot reuse stochastic generation or judging outputs from another
repetition. Record actual model/deployment identity, sampling settings, provider,
date, and SDK/API versions. Unspecified paper hyperparameters must be documented
as reconstruction choices, never described as recovered experimental settings.

Support separate named configurations for o3 and DeepSeek-V3.2 ablations. Include
import interfaces for the modified Who&When baseline and the MAST stress test;
their comparison is only valid after original inputs and baseline configuration
are available and checked. The initial available subset cannot reproduce the
full Table 3 comparison or MAST figures. Optional reflection/pruning is off by
default and must be reported as a separate variant if enabled.

## Metrics and reporting

Implement exact critical-step accuracy, Acc@+/-1, +/-3, +/-5, average absolute
step distance, failed-agent accuracy, and critical/any/earliest/terminal category
accuracy. Evaluate earliest and terminal categories by temporal step order,
with explicitly documented handling of ties, rather than assuming failure IDs
encode chronology.

Compute metrics separately per domain and repetition, then mean and sample
standard deviation across repetitions. Record this standard-deviation convention
because the paper does not specify its estimator. Report numerator, denominator,
corpus IDs/hash, coverage, invalid predictions, abstentions, and failed jobs.
Accuracy uses the fixed intended corpus denominator; abstentions and failed jobs
cannot improve accuracy by dropping out. Average distance is calculated only for
valid step predictions and always accompanied by their coverage count.

Export paper-style Table 4 and Table 7 reports, per-trajectory predictions,
category confusion matrices, and machine-readable metrics. Keep paper-reported
numbers in a separate reference file. A comparison must distinguish exact-corpus
measurements, available-subset results, offline fixture tests, and blocked rows.
For Table 3, identify the chosen protocol/configuration for each result instead
of silently cherry-picking the best measured run. Never hard-code reported
accuracies as generated predictions or claim exact numeric replication in advance.

## Verification and delivery

Tests must verify index preservation, root-cause joins, label separation from
model prompts, prefix-only generation/checking, guard behavior, semantic ambiguity,
Python timeouts/errors, recovered early failures, metric denominators, independent
repetitions, and resume invalidation when prompts/configuration change.

Deliver an offline end-to-end smoke test, an audited manifest of the available
73 pairs, explicit status for unavailable domains, and documented commands for
data audit, single-trajectory diagnosis, repeated experiments, and evaluation.
Then run a small model-backed pilot once credentials are configured, followed by
the approved benchmark matrix using resumable jobs. Do not label fixture testing
as successful reproduction of the paper's results.

Completion of the full scientific reproduction requires the missing original
data, working model endpoints, finished repeated experiments, and reports of the
measured results and deviations. Implementation completion is a separate milestone.

## Decisions left to implementation

The builder may choose the package layout, validation library, test framework,
retry policy, atomic checkpoint representation, and report rendering library.
These choices must preserve the contracts above. Keep the core pipeline usable
without Azure-specific infrastructure. Do not install or invoke the original
repository's internal Microsoft TRAPI service as a requirement.

## Builder check

A fresh software-engineering reviewer checked the design and reported
"nothing important left". Subsequent source inspection clarified that supplied
task instructions should follow the reference extraction behavior, while expected
actions and answers remain excluded from model inputs.
