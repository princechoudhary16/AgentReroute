# Repair strategies

Part 3 of AgentReroute. Given a failed agent run and an attributed fault
(*which step, which category*), it produces the **repair plan**: where to roll
back, what to change, and how to tell whether the change took effect.

```
fault attribution ──► repair strategies ──► recovery orchestrator / benchmark harness
 (step, category)      (RepairPlan)          (whether and when to apply it, replay, verify)
```

Standard library only (Python 3.10+). It reads the other parts' outputs as
data and never imports their packages (enforced by `tests/test_boundaries.py`),
so this branch merges independently. Design notes: [docs/repair-strategies-design.md](../docs/repair-strategies-design.md).

## Why plans, not guidance strings

The earlier proposal mapped each fault to one guidance sentence. Three findings
from the team's strategy review shape a different design:

1. **Replace, don't critique.** A critique tends to be miscorrected; a verified
   value is not. Strategies prove a claim wrong from the trace and supply the
   correct value.
2. **External role.** Corrections are delivered as tool messages rather than
   inside the agent's own reasoning (arXiv 2606.05976).
3. **Gates, not reminders.** Policy is enforced before each call during
   re-execution; a reminder in the prompt decays.

A `RepairPlan` is therefore a list of typed operations (inject a tool message,
patch a handoff, re-query, gate calls, ban a call, switch tool, resample, ask,
retry, compensate, escalate, abstain) plus deterministic **acceptance checks**.
Plans are backend-neutral and **lowered** to what a backend can execute.

## Quick start

From the repository root:

```bash
python3 -m unittest discover -s repair_strategies/tests -t .
python3 -m repair_strategies catalog
python3 -m repair_strategies plan --trace trace.json \
    --hypothesis '{"step_index": 7, "category": "InventionOfNewInformation"}' --target benchmark
```

`--hypothesis` also accepts a fault-attribution `result.json` (its `prediction`
is used) or a benchmark submission row (its first hypothesis). `--explain`
shows every rung of the ladder and why it applies or not.

| Command | Purpose |
|---|---|
| `catalog [--category C]` | strategies, ladder rung, cost and prior |
| `registry [--out F]` | registry for the orchestrator (its schema v1) |
| `plan --trace T --hypothesis H [--history H] [--target benchmark\|orchestrator\|full] [--explain]` | next repair plan |
| `audit --trace T` | rule violations, loops and unsupported claims |
| `check --plan P --rollout R` | acceptance checks on a replayed run (exit 1 if they fail) |

`--preset retail` (default) loads the tau-bench retail tools, the policy rules
that can be checked from a trace, compensations and safe defaults. Use
`--task spec.json` for another domain.

## Strategy catalogue

Rungs are tried in order. A later rung is offered once the earlier applicable
ones failed on the same run.

| Fault | Rung | Strategy |
|---|---|---|
| Invention of new information | 1 | `grounded_replacement`: replace contradicted claims with verified values; re-query the source of ungrounded ones |
| | 1 | `complete_omitted_details` (omission): supply requested facts the answer left out |
| | 2 | `abstain_or_ask`: ask the user, or state the information is unavailable |
| Misinterpretation of tool output / handoff | 1 | `repair_handoff`: patch the handoff with fields recovered from evidence, re-run the receiver |
| | 1 | `structured_reading`: parse the output and compute counts and sums in code |
| | 2 | `independent_reading` (LLM): second reading in a fresh context, confirmed against the output |
| | 3 | `cross_source_check`: re-query to catch stale or corrupted data |
| Intent-plan misalignment | 1 | `intent_contract_replan`: fresh replan against an explicit contract, with a lesson from the abandoned branch |
| | 2 | `confirm_ambiguous_writes`: confirm the target of a write with the user |
| Instruction / plan adherence | 1 | `policy_gate`: pre-execution rules plus a persistent done/pending checklist |
| | 2 | `policy_gate_fresh`: the same after a fresh resample |
| Invalid invocation | 1 | `hard_call_repair`: block the call, return exact schema violations, request the corrected call |
| | 2 | `switch_on_schema_drift`: route to an equivalent tool |
| Cycle | 1–4 | `loop_nudge`, `loop_force_change`, `loop_replan`, `loop_rollback` |
| System failure | 1 | `tool_outcome_routing`: retry, switch or abstain from per-tool outcome memory |
| Underspecified user intent | 1 | `targeted_clarification`: ask only for missing fields, apply safe defaults |
| Guardrails / intent not supported | 1 | `compensate_and_escalate`: undo writes that have an inverse, then escalate |
| Inconclusive | 1 | `verified_unmodified_replay`: one bounded resample, accepted only through verification |
| **Any fault (fallback)** | last | `rollback_compensate_escalate`: see below |

### Fallback: rollback, compensate, escalate

When every applicable strategy for a fault has failed, when none applies, or
when the budget runs out, the engine issues one last plan for any category:

1. **Roll back side effects, not the run.** Writes made from the fault onward
   are compensated with their inverse actions from the task spec (e.g. restore
   the address read before the change). Writes before the fault are the user's
   completed work and are kept.
2. **List what cannot be undone** (no inverse, previous value never read,
   arguments not logged).
3. **Escalate** with the fault, every strategy tried and how it ended, and what
   was reverted.

The fallback sits outside the ladders, is issued once per run, and is not
exported to the orchestrator registry (its EV policy should never choose it as a
repair). In the benchmark adapter, a run where replays happened but none passed
acceptance still delivers the last replay; the fallback applies when no replay
could run. `deliver_unaccepted=False` makes it escalate in both cases.

Benchmark modes map onto these categories; the mode is kept as a subtype
(`context_propagation` → handoff, `missing_details` → omission, `drift`).

## Layout

```
repair_strategies/
├── taxonomy.py, trace.py, context.py     inputs: fault types, trace model, hypothesis, task spec
├── evidence/                             deterministic analysis
│   ├── claims.py                         claim provenance (ids, amounts, counts) with entity anchoring
│   ├── tool_output.py                    parsing, error classes, aggregates
│   ├── schema.py                         tool-call validation
│   └── cycles.py                         loop detection
├── policy/                               rules.py (pre-execution gate), plan_state.py, intent.py
├── memory/                               tool outcome posteriors, verified lessons, atomic JSON store
├── strategies/                           one module per fault family, plus base.py
├── plan.py, acceptance.py                the RepairPlan model and its post-conditions
├── engine.py                             ladder selection, lessons, learning from outcomes
├── lowering.py                           capabilities and lowering per backend
├── adapters/                             orchestrator and benchmark integration
├── llm.py, cli.py
└── resources/retail.json                 retail preset
```

## Integration

**Orchestrator** (`recovery_orchestrator`): write the registry, then pass the
repairer to `Orchestrator`.

```python
from recovery_orchestrator.orchestrator import Orchestrator
from recovery_orchestrator.policy import EVPolicy
from recovery_orchestrator.registry import StrategyRegistry
from recovery_orchestrator.replay import RepairDirective
from repair_strategies.adapters.orchestrator import OrchestratorRepairer, export_registry
from repair_strategies.context import TaskSpec
from repair_strategies.engine import RepairEngine

registry = StrategyRegistry.from_dict(export_registry())
repairer = OrchestratorRepairer(RepairEngine(TaskSpec.preset('retail')), directive_factory=RepairDirective)
orchestrator = Orchestrator(env, verifier, EVPolicy(registry), repairer=repairer)
```

Each repeated `build` for a trajectory counts the previous plan as failed, so the
next rung is used. `repairer.record_outcome(trajectory_id, verified)` feeds the
verifier's verdict back for lessons and tool memory.

**Benchmark harness** (`benchmark.scoring.recovery`):

```python
from benchmark.env.workflow import Directive
from benchmark.scoring.recovery import run_recovery
from repair_strategies.adapters.benchmark import RepairStrategyPolicy

report = run_recovery(corpus, 'test', RepairStrategyPolicy(directive_cls=Directive), llm_factory, oracle=True)
```

`policy.log[trajectory_id]` records each attempt: strategy, rollback step,
acceptance failures and degraded operations.

**Fault attribution**: no import is needed. A `result.json` prediction (or an
orchestrator `top_k` item) converts with `Hypothesis.from_mapping`.

## LLM use (Ollama)

Twenty of the 22 strategies are deterministic and need no model. The other
two, `independent_reading` and the optional intent-contract extractor, use a
local model through Ollama's native chat API (`OllamaClient`). No API keys or
cloud endpoints are involved.

```bash
ollama serve
ollama pull qwen2.5:7b
python3 -m repair_strategies plan ... --llm qwen2.5:7b
python3 -m repair_strategies plan ... --llm qwen2.5:7b --ollama-url http://gpu-box:11434 --num-ctx 32768
```

The server defaults to `$OLLAMA_HOST`, then `http://127.0.0.1:11434`. Requests
ask for JSON output and disable prompt truncation and context shifting, so an
overlong prompt or a reply cut at the token limit is an error rather than a
silently shortened answer. Without `--llm` those two rungs are skipped, with the
reason shown by `--explain`. In Python, pass `llm=OllamaClient('qwen2.5:7b')`
to `RepairEngine`.

## Extending

* **A strategy**: subclass `strategies.base.Strategy`, declare a `StrategySpec`
  (categories, rung, registry kind, cost, prior), return a `RepairPlan` or a
  `Skip(reason)`, add it to `strategies.CATALOGUE`, and test both outcomes.
* **A domain**: copy `resources/retail.json`. Rules are declarative
  (`requires_prior_call`, `requires_confirmation`, `argument_enum`,
  `at_most_once`, `argument_grounded`, `forbidden_tool`); compensations name an
  inverse tool and where its previous values are read from.
* **A backend**: describe it with a `lowering.Capabilities`; operations it
  cannot execute are turned into guidance and listed in `degraded`.

## Verification

* 163 unit and integration tests (`python3 -X dev -W error -m unittest ...`),
  including a local HTTP server standing in for the Ollama API and a subprocess run of the CLI.
* A wheel built from `pyproject.toml` installs and runs outside the checkout,
  resources included.
* Planning a repair on a 300-step trace with 6 KB tool outputs takes about one
  second (evidence is parsed once per trace and shared across steps).
* Run against the real orchestrator branch (scripted environment and verifier):
  the exported registry loads, `EVPolicy` selects from it, and a misread count
  in fault attribution's smoke trace is recovered on the second rung.
* Run in the benchmark harness on its **offline fixture** (scripted agents,
  8 retail tasks, 23 failed test episodes):

| Policy | Oracle attribution | Heuristic attribution | Harm on controls |
|---|---|---|---|
| `fixed_rules` (earlier proposal) | 0.65 | 0.30 | 0.00 |
| `repair_strategies` (default) | 0.65 | 0.48 | 0.00 |
| `repair_strategies`, `deliver_unaccepted=False` | 0.65 | 0.48 | 0.00 |

These numbers verify the pipeline, not model behaviour: the scripted agents
recover from any directive applied at or before the faulty turn, and the cycle
and tool-failure faults in the fixture are persistent for every policy. With
the strict setting every delivered replay passed its acceptance checks, and the
recovery rate is unchanged. Real
results need model-backed episodes. The policy used about 1.4–1.7× the tokens of
`fixed_rules`, mostly by climbing the cycle ladder on unrecoverable episodes;
how much of that budget to spend is the orchestrator's decision.

## Limits

* Claim checks cover identifiers, emails, amounts and quantified counts.
  Free-text facts without such anchors are left to the verifier.
* Acceptance checks are cheap pre-filters. They never declare a recovery; the
  orchestrator's fresh-context verifier does.
* Priors in `StrategySpec` are starting values to be tuned from evaluation.
* Backends that only accept guidance text cannot run re-queries, gates or event
  patches natively; `degraded` records each case.
