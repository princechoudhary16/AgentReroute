# Repair strategies design (KAN-6)

Status: implemented in `repair_strategies/`. Owner: Repair team. Consumers:
Recovery Orchestrator (KAN-7) and the benchmark recovery harness (KAN-4).
Input producer: Fault Attribution (KAN-5).

## 1. Problem

Fault attribution tells us *where* a run went wrong and *what kind* of fault it
was. The orchestrator decides *whether* and *when* to intervene. This component
answers the remaining question: **given an attributed fault, what exactly do we
change so that the replayed run succeeds?**

The earlier proposal answered with one guidance sentence per fault type
("re-present the output", "re-inject the checklist"). Three observations from
the team's strategy review make that insufficient:

1. **A critique alone tends to be miscorrected.** Replacing the faulty input
   with a verified one works better than telling the agent it was wrong.
2. **Corrections are taken more seriously when they arrive under an external
   role** (tool or system message) than inside the agent's own reasoning
   (arXiv 2606.05976).
3. **Text reminders decay during re-execution.** A rule that is enforced before
   each tool call cannot be ignored; a reminder can.

So a repair is modelled as a **typed plan**, not a string.

## 2. Contracts

### 2.1 Input

| Input | Source | Shape |
|---|---|---|
| Trace | FA `trajectory_ir.json`, benchmark public trace, orchestrator `Trace`, OpenAI messages | `steps[].substeps[]` with `role`, `content`, `tool_calls`, `name` |
| Hypothesis | FA `prediction`, orchestrator `Candidate`, benchmark submission hypothesis | `step_index`, `category` or `mode`, `agent`, `confidence`, `rationale` |
| Task spec | Instruction, policy, tool schemas, write tools, rules, compensations | `TaskSpec` (a retail preset ships with the package) |
| History | Plans already tried for this run and their outcome | `AttemptRecord` list |

Categories are normalized to the AgentRx taxonomy plus `Cycle`. The benchmark's
seven modes map onto it; the mode is kept as a **subtype** so a strategy can
specialize (e.g. `context_propagation` is the *handoff* subtype of
MisinterpretationOfToolOutput, `missing_details` is the *omission* subtype of
InventionOfNewInformation).

### 2.2 Output: `RepairPlan`

```
RepairPlan
  strategy_id, category, rollback_step
  operations     ordered, typed (below)
  acceptance     deterministic post-conditions on the repaired rollout
  terminal       true for escalate / abstain plans (nothing to replay)
  estimated_tokens, expected_success, rationale, evidence
```

| Operation | Meaning |
|---|---|
| `InjectMessage(role, content, persistent)` | Insert a message at the rollback point. `role` is `tool`, `system` or `user`; `persistent` re-applies it on every turn of the replay |
| `ReplaceEvent(step, sub, content)` | Patch a kept event before the rollback point (e.g. complete a handoff) |
| `RequeryTool(tool, arguments)` | Re-execute a read-only tool and insert its fresh result |
| `GuardCalls(rules, schema, contract)` | Pre-execution gate applied to every proposed call during the replay |
| `BanCalls(signatures)` | Refuse exact repeats of a looping call |
| `SwitchTool(from, to)` | Route to an equivalent tool |
| `Resample(fresh_context, seed_offset)` | Regenerate from the rollback point |
| `AskUser(question, fields)` | Ask only for the missing fields |
| `Retry(backoff_s)` | Re-issue a call after a transient failure |
| `Compensate(calls, irreversible)` | Undo executed writes where an inverse exists |
| `Escalate(reason, summary)` / `Abstain(reason)` | Terminal outcomes |

The plan is backend-neutral. **Lowering** (section 6) turns it into what a given
backend can execute and records every operation it had to degrade.

## 3. Architecture

```
            trace + hypothesis + task spec + history
                              │
   ┌──────────────────────────▼──────────────────────────┐
   │ evidence/   claims & provenance · tool-output parse │  deterministic,
   │             schema validation · cycle detection     │  no LLM needed
   │ policy/     rule gate · plan state · intent contract│
   │ memory/     tool outcome posteriors · lessons       │  persistent JSON
   └──────────────────────────┬──────────────────────────┘
                              │
   ┌──────────────────────────▼──────────────────────────┐
   │ strategies/  one family per fault, ordered ladder   │
   │ engine.py    select next untried rung, explain why  │
   └──────────────────────────┬──────────────────────────┘
                              │ RepairPlan
   ┌──────────────────────────▼──────────────────────────┐
   │ lowering.py  capabilities → directive + degradations│
   │ acceptance.py post-conditions on the rollout        │
   │ adapters/    orchestrator · benchmark               │
   └─────────────────────────────────────────────────────┘
```

The package is standard-library only and never imports `agentrx`, `benchmark`
or `recovery_orchestrator`. Integration is by data contract and duck typing, so
the branch merges independently of the others.

## 4. Strategy catalogue

Rungs are tried in order; a later rung is offered only after the earlier
applicable rungs for the same fault have failed.

| Fault | Rung | Strategy | What it does |
|---|---|---|---|
| Invention of new information | 1 | `grounded_replacement` | Provenance check on every claim (IDs, amounts, counts) against user turns and tool outputs. Contradicted claims are replaced by the verified value, delivered as a tool message; ungrounded claims trigger a re-query of the tool that can verify them |
| | 1 | `complete_omitted_details` (omission) | Finds requested facts that were retrieved but left out of the final answer and supplies them as verified data |
| | 2 | `abstain_or_ask` | No source exists: ask the user if the fact is theirs to give, otherwise forbid the claim and state it as unavailable |
| Misinterpretation of tool output / handoff | 1 | `repair_handoff` (handoff) | Fields the receiving agent's tools require but the handoff omitted are recovered from earlier evidence and patched into the handoff; the receiver re-runs |
| | 1 | `structured_reading` | Parses the consumed output into fields, computes counts and sums in code, and presents the authoritative reading as a tool message |
| | 2 | `independent_reading` | A second reading in a fresh context (LLM), compared with the deterministic parse |
| | 3 | `cross_source_check` | Re-queries the source (and an equivalent tool) to catch silent corruption |
| Intent-plan misalignment | 1 | `intent_contract_replan` | Extracts an intent contract (goal, entities, required and forbidden actions), rolls back to before the plan was formed, regenerates in a fresh context with a lesson from the abandoned branch, and gates every action against the contract |
| | 2 | `confirm_ambiguous_writes` | Asks the user to confirm when the target entity is ambiguous |
| Instruction / plan adherence | 1 | `policy_gate` | Compiles the policy into pre-execution rules, gates every call, and carries a persistent plan-state checklist (done / pending) through re-execution |
| | 2 | `policy_gate_fresh` | Same gate after a fresh resample |
| Invalid invocation | 1 | `hard_call_repair` | Blocks invalid calls, returns the exact schema violations as a tool error and asks for the corrected call only |
| | 2 | `switch_on_schema_drift` | Switches to an equivalent tool when the error indicates drift |
| Cycle | 1–4 | `loop_nudge`, `loop_force_change`, `loop_replan`, `loop_rollback` | Escalation ladder: nudge, ban the repeated call, replan from before the loop, roll back to the last user turn |
| System failure | 1 | `tool_outcome_routing` | Classifies the error (transient / persistent) and uses per-tool outcome memory to retry, switch to an equivalent tool, or abstain |
| Underspecified user intent | 1 | `targeted_clarification` | Asks only for required fields that no evidence supplies; fills low-risk fields with declared safe defaults |
| Guardrails / intent not supported | 1 | `compensate_and_escalate` | Compensates executed writes that have an inverse, lists the irreversible ones, then escalates with a partial-completion summary |
| Inconclusive | 1 | `verified_unmodified_replay` | One bounded resample, accepted only through the verifier |

Verified lessons (memory) are attached to any plan whose fault and tools match
a lesson that previously led to a verified recovery.

## 5. Acceptance

Each plan carries deterministic post-conditions evaluated on the replayed part
of the rollout: claims grounded, no repeated signature, calls schema-valid,
rules satisfied, obligations completed, contract respected, the user was asked
for the right fields. These are cheap pre-checks. They **do not replace** the
orchestrator's fresh-context verifier, which still decides whether a repair
counts as recovered.

## 6. Capabilities and lowering

| Backend | Can execute |
|---|---|
| Orchestrator (`RepairDirective`) | text guidance (`inject_guidance`), `retry`, `none` |
| Benchmark (`Directive`) | `guidance` (supervisor note), `resample`, `none` |
| Full | every operation |

Lowering keeps what the backend supports and turns the rest into explicit
instructions inside the guidance text (e.g. a gate becomes the rule text, a
re-query becomes "call X again before answering"). Every such step is listed in
`degraded`, so evaluation can separate strategy quality from backend limits.

## 7. Memory

* **Tool outcome memory.** Beta posterior of success per tool plus counts of
  transient and persistent errors. Drives retry / switch / abstain.
* **Lesson memory.** A lesson is stored only after a *verified* recovery and
  is retrieved by fault category and overlapping tools/agents.

Both persist to JSON with atomic writes.

## 8. Integration

* **Orchestrator.** `adapters.orchestrator.export_registry()` writes the
  registry in the orchestrator's schema (v1). `OrchestratorRepairer`
  implements its `Repairer.build(strategy, candidate, trace)` protocol.
* **Benchmark.** `adapters.benchmark.RepairStrategyPolicy` implements the
  harness's `recover(env)` protocol: plan, lower, replay, check acceptance,
  escalate the ladder, deliver the accepted rollout.

## 9. Evaluation

Use the benchmark harness with paired runs on the same failed episodes:

* `fixed_rules` (the earlier proposal) vs `repair_strategies`, both with oracle
  and with predicted attribution;
* per fault mode recovery, harm on controls, tokens per recovery;
* the `degraded` counts per backend.

## 10. Limits

* Claim extraction covers identifiers, money, and quantified counts. Free-text
  facts without such anchors are left to the verifier.
* The retail preset encodes the tau-bench retail policy that can be checked from
  a trace. Other domains need their own rules and compensations.
* `independent_reading` and the LLM contract extractor need a local Ollama
  model; the remaining strategies are deterministic.
