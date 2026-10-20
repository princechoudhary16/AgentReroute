# Ranked fault attribution (Track B)

Track B adds a divergence gate and a ranked recovery handoff to AgentRx. It is an
opt-in extension, separate from the paper's single-answer protocols and the tool
prevention middleware.

## Run

Offline plumbing demonstration, using prerecorded responses:

```bash
python3 -m agentrx smoke-track-b --out runs/track-b-demo
```

With a running Ollama server and an installed model:

```bash
python3 -m agentrx run \
  --config configs/track_b_ollama.json \
  --model YOUR_INSTALLED_MODEL \
  --out runs/track-b-local
```

Prepare the available benchmark first with `python3 -m agentrx prepare --fetch` if
`data/benchmark` is absent. The config runs one retail trace, one repetition.
It defaults to `qwen2.5-coder:7b`; use the override above for another installed model.
Use a new output directory after changing code, data, settings or model weights.
For a single saved trace, use `diagnose --input TRACE.json --domain tau` with the
same config, model and output options. The existing `dev_ollama.json` behavior is
unchanged.

## What happens

1. Generate and evaluate constraints using the existing AgentRx pipeline.
2. Group mechanically identical checks at the same step. Python formatting,
   comments and the generated function name do not create different rules. This
   does not prove arbitrary semantic equivalence between rules.
3. Count conclusive violations and satisfactions. Divide violations by their sum.
4. If that rate is **strictly greater** than the configured threshold, ask the
   model for ranked critical-failure hypotheses. Otherwise skip attribution.
5. Validate the hypotheses, sort by confidence (earlier step breaks ties), save
   the handoff, and expose the first candidate through the existing prediction.

The example threshold is **0.30**, an explicit demonstration setting, not a tuned
operating point. Configure `divergence_threshold` in [0,1] and `max_candidates` as
2 or 3. Ranked mode requires evidence generation.

The denominator is **unique conclusive rule/step evaluations**, not all generated
rules. Skipped checks, errors and unresolved judgments are excluded. A semantic
check with any definite failure counts as violated; a check with uncertainty and
no definite failure is unresolved. Conflicting SAT/VIOL results for an identical
rule/step are unresolved. A duplicate error does not erase an otherwise conclusive
measurement; raw errors remain visible in the pipeline's partial status.

No conclusive checks means `inconclusive`, with a null rate. At or below the
threshold means `within_boundary`; this does not establish success or creative
validity. The generated rules are incomplete, fallible and unweighted. More
applicable low-value rules can still dilute the rate. Evaluate threshold choices
on representative traces before treating the gate as an operational decision.

## Recovery payload

Every ranked job saves `recovery_handoff.json` alongside `result.json`,
`trajectory_ir.json`, `checks.json` and the request log. The same payload appears
in `result.recovery_handoff`.

| Field | Meaning |
| --- | --- |
| `schema_version` | `agentrx-recovery-v1` |
| `decision` | `repair_required`, `within_boundary`, or `inconclusive` |
| `boundary` | Threshold, strict comparison, rate, raw/unique/duplicate counts and excluded-check counts |
| `status` | `ok`, `partial_checker_errors`, or `error` |
| `attribution_status` | `ranked`, `abstained`, `not_requested`, or `error` |
| `taxonomy_version` | `agentrx-9-v1` |
| `confidence_kind` | `uncalibrated_model_estimate` |
| `candidates` | Up to three ranked, distinct observed steps |
| `abstention_reason` | Explanation when no ranking is available; otherwise null |

Each candidate has `rank`, `step_index`, `agent`, `category`, `confidence`,
`rationale`, and specific `evidence_refs`. Unsupported locations, evidence, types,
duplicate steps and invalid scores are rejected. The model may return one grounded
candidate or abstain rather than fabricate alternatives. The existing nine
failure types are used; adopting a team-defined 19-mode vocabulary requires its
explicit definitions and mapping.

**Confidence is not a calibrated probability or a probability distribution.**
Candidate scores need not sum to one. A divergence rate measures different
things from a model's confidence in which step caused failure.

If the ranker abstains or fails, an existing `repair_required` alarm remains in the
handoff. Recovery should act on candidates only when decision is
`repair_required`, attribution is `ranked`, and status is not `error`.

A consumer validates the payload against the saved trajectory and original checks:

```python
from agentrx.handoff import load_handoff
from agentrx.io import read_json

handoff = load_handoff(
    job / "recovery_handoff.json",
    read_json(job / "trajectory_ir.json"),
    checks=read_json(job / "checks.json"),
)
if (handoff["decision"] == "repair_required"
        and handoff["attribution_status"] == "ranked"
        and handoff["status"] != "error"):
    candidates_for_recovery = handoff["candidates"]
```

This validates internal consistency, not authenticity or the truth of model
reasoning. Keep the artifacts from a trusted run together.

The other branch's legacy RCA payload uses repeated-run **votes/share**. This
implementation does not write `rca_payload.json` or convert confidence into votes.
That orchestrator needs an explicit adapter for this versioned contract; no
recovery actions are executed here.

## Reports

`summary.json` and the existing tables retain the normal top-one metrics.
`handoff_summary.json` adds top-1/2/3 critical-step hit rates, decision/attribution
counts, missing/failed/invalid counts and exploratory confidence bins. All intended
case/repetition jobs remain in the accuracy denominator, including missing runs,
errors, abstentions and traces that did not pass the gate. Confidence bins include
emitted candidates only and report their observed critical-step correctness;
empty bins are null. They do not demonstrate calibration.

The offline smoke is a fixture test, not benchmark accuracy. See
[verification](verification.md) for actual local checks.
