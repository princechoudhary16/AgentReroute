# Implementation verification — 2026-10-14

The reconstruction is implemented. This records software verification, not
reproduction of the paper's measured model accuracies.

## Evidence

- `python3 -m unittest discover -v`: 41 passing tests, including CLI subprocesses,
  real localhost HTTP transport/retries, Python worker processes, prefix isolation,
  attribution/abstention validation, fixed denominators, and resume identities.
- `python3 -m compileall -q agentrx` and `git diff --check`: passed.
- Editable installation and standalone wheel build: passed. The wheel was
  installed into a separate target and its smoke test ran from `/tmp`, confirming
  packaged resources and worker paths work outside the checkout.
- `prepare --fetch`: fetched the pinned authors' commit and audited 73 pairs with
  no excluded records. The original corpus first-prefix prompt boundary audit
  also passed for all 73 cases.
- Gold-label oracle consistency check: all 73 original roots can be accepted and
  scored correctly, including two whose responsible agent differs from the
  selected event speaker. This is a scorer check by construction, not prediction
  accuracy.
- `smoke --out runs/verified-smoke`: baseline and AgentRx plumbing across three
  repetitions each; six jobs finished without errors. Reports visibly identify
  offline fixture testing. Standalone replay diagnosis found two count-mismatch
  violations and attributed the fixture failure to step 3.
- Model preflight: correctly refused to start without `OPENAI_API_KEY`, exited 2,
  and did not create a model-backed experiment. No paid model calls were made.

## Independent review and fixes

A fresh reviewer inspected the complete implementation and ran the original 36
tests. Four material findings were verified with failing regression tests, then
fixed; the full 41-test suite passed afterward:

1. Raw module exports exposed file access through `statistics.sys`; constrained
   export facades now reject this and dynamic attribute/frame variants.
2. Responsible-agent validation incorrectly required the agent to speak at the
   selected step; validation now checks known agents across the trajectory.
3. Invalid list/range selectors silently matched or skipped; malformed selectors
   now become explicit errors, excluding booleans and negative/reversed ranges.
4. Localization abstention required a category while its prompt deferred category;
   null localization with a rationale/custom category now becomes Inconclusive.

No minor findings were deferred. The reviewer could not assess live endpoint
compatibility/diagnostic quality, exact numeric replication/full-corpus membership,
or unavailable historical prompt/model snapshots. Those remain unverified and
are disclosed as external prerequisites, not treated as passing checks.

## Implementation decisions and limits

- Execution continued inline following the user's approval, in the initially
  empty project on a dedicated branch. There was no existing base branch or
  remote to merge or publish; all project work remains local and reviewable.
- Actual domain-registry tool definitions replace the source's misleading
  example-event `tools_structure.json`; MIT notices and revision are retained.
- DeepSeek requires a configured endpoint serving V3.2; a changing API alias is
  not silently substituted. Incorrect endpoint configuration can still change
  experimental meaning, so returned model identity is recorded.
- Python execution exposes a deliberately narrower interface than unrestricted
  reference execution. Unsupported generated constructs become auditable errors;
  this may reduce checker coverage. It is not a complete adversarial sandbox.
- Paper sampling/deployment snapshots are unspecified; completion caps, prompt
  reconstruction, and sample SD are explicit reconstruction choices.

The available original data consists of 29 retail and 44 Magentic-One traces.
Full 170-trajectory reproduction still lacks 10 retail, 42 Flash, and 45 RelWork
traces, working model credentials, and completed independent live repetitions.
W&W predictions and normalized MAST data are import interfaces, not bundled
reproductions of those external systems.

## Repository integration

The implementation was subsequently merged into the existing `faultattribution`
branch of `princechoudhary16/AgentReroute`, preserving its initial commit, project
description, and original MIT license. Microsoft's reused-material license notice
is retained separately in `LICENSE-Microsoft-AgentRx.txt`. Generated data, local
environments, and run outputs remain excluded by `.gitignore`.

## Ollama integration verification — 2026-10-18

Pulled collaborator commit `963180c` from `faultattribution`. All 47 incoming tests
passed. Regression fixes brought the suite to 57 passing tests:

- Preserve custom endpoint/context/thinking/timeout options when choosing the
  Ollama preset; do not carry OpenAI model names/options into Ollama.
- Honor explicit model/endpoint environment overrides and reject missing explicit
  endpoint variables rather than silently choosing localhost.
- Send native `truncate:false` and `shift:false`, require completed responses,
  and validate usage counters. Server-side context errors name `num_ctx`.
- Resolve and record model digest, quantization metadata, and Ollama version before
  experiment/cache lookup. Changing weights under the same tag invalidates reuse.
  Preflight remains a no-network configuration check.
- Reject plaintext secrets in persisted experiment/model configuration.

Actual local verification used Ollama **0.40.2** and the already installed model
`hf.co/noctrex/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF:Q4_K_M`, digest
`f227e448f8bd23fb9dfd506422315fc22be723fed48721bcc9a6361a290bfb09`,
with 16,384 context tokens and temperature 0. No cloud LLM was called.

The configured default `qwen2.5-coder:7b` was not installed. A slow download was
stopped; its default-model inference path remains unverified on this machine.

Live checks performed:

- Full three-step diagnosis with actual global/dynamic generation and judging;
  selected step 3 correctly but selected a different category than the fixture.
- Original retail trajectory **2**, **32 steps**: baseline and AgentRx completed
  in both one-shot and stepwise modes and produced JSON/CSV/Markdown reports.
  AgentRx selected the gold critical step **7**; the baseline selected **16**.
  Both category predictions differed from the critical gold category. This is a
  single-case engineering pilot, not a benchmark accuracy estimate.
- Generated assertions were all inapplicable: 64 SKIP outcomes in one-shot and
  42 SKIP outcomes in stepwise. Thus these runs verify orchestration/artifacts,
  not useful generated constraint evidence. No generated assertion was secretly
  rewritten to force a match.
- A separate controlled count-mismatch test executed both a real Python check
  and a live Ollama semantic rubric; both returned VIOL. The semantic response
  correctly identified two available variants versus the claimed three.
- A real 12,009-token prompt with a 2,048-token context and truncation disabled
  was rejected by Ollama with HTTP 400 (`exceed_context_size_error`).

Local run artifacts are under `runs/ollama-live-retail-20261018`,
`runs/ollama-live-stepwise-20261018`, `runs/ollama-live-checkers-20261018`, and
`runs/ollama-final-digest-e2e-20261018`. They remain excluded from Git.

These controls follow Ollama's [native request schema](https://raw.githubusercontent.com/ollama/ollama/main/api/types.go).
Keep the selected local model unchanged during a run; weights are resolved at
run start, not locked against another process updating its tag mid-generation.

## Tool prevention verification — 2026-10-18

Added a separate `tool_prevention` package with a persistent SQLite retail backend,
explicit policy compiler, protected registry/dispatcher, scoped approval tokens,
conditional writes, durable operation results, trusted reconciliation, finite host
completion obligations, and redacted audit/AgentRx trace export. See
[setup and trust boundary](prevention.md).

Verification:

- **89 tests pass**, including the existing 57 diagnosis/provider tests and 32
  prevention tests, with Python warnings treated as errors.
- CLI request → preview → trusted approval → execution → audit/trace works across
  separate processes using persistent databases.
- Concurrent gateway instances cannot consume one grant twice. Restart preserves
  results. A committed write followed by a timeout remains uncertain until backend
  evidence confirms success; a late timeout cannot undo confirmed success.
- Journals reject another backend instance. Approval changes, expiration,
  revocation, ownership/status/product/payment errors, stale versions and explicit
  omissions are tested. All seven mutation tools have valid controls.
- Composed item change → cancellation/payment replacement reverses the recorded
  payment allocations, including mixed card/gift-card funding.
- Built a wheel, installed into an isolated environment, and ran both diagnosis
  and prevention smoke commands from outside the source checkout. Both packaged
  resource sets load successfully.

Controlled local evaluation uses 22 scenarios per condition, including five valid
controls. The selected run produced:

| Condition | Unsafe writes | Duplicate writes | Valid business outcomes | False rejections |
| --- | ---: | ---: | ---: | ---: |
| Unprotected reference | 18 | 4 | 5/5 | 0 |
| Schema-only reference | 17 | 4 | 5/5 | 0 |
| Full prevention | 0 | 0 | 5/5 | 0 |

The full condition also blocked completion with an unfinished explicit obligation.
“Valid outcome” checks actual order state and a single refund to the original
payment method, separately from dispatch success. Conditions share simulated
settlement semantics; baseline policy/approval checks are deliberately absent.
Concurrent baseline write counts can vary with scheduling. These are engineering
controls against an intentionally unchecked local writer, **not** an evaluation of
original tau-bench backend vulnerabilities, paper benchmark recovery, or universal
agent reliability.

Historical screening covers all **73** available labeled-corpus traces without
using their labels: 29 retail, of which 24 contain supported mutation calls, and
44 Magentic traces without structured calls. One retail prefix has an incompatible
mutation argument schema; its suffix is not evaluated. Approval and authoritative
state evidence remain unknown. No prevented-failure or recovered-task claim follows
from this screening.

Artifacts: `runs/prevention-evaluation-20261018/evaluation.json`,
`runs/prevention-smoke-20261018/{report,audit,trace}.json`,
`runs/prevention-screen-20261018.json`, and
`runs/prevention-package-verification/`. Generated artifacts remain excluded from Git.

Independent review identified and verified fixes for backend journal identity,
late-timeout state races, payment composition, and evaluation postconditions. The
remaining integration limit is explicit: the collaborator's orchestrator branch
has no implemented live tau backend to connect. The supplied proposed-action
adapter and executable local reference do not substitute for that live task study.
