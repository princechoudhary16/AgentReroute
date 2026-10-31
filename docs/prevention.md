# Generic tool prerequisite middleware

`tool_middleware` wraps an existing enterprise tool registry. An LLM examines its
catalog and proposes simple prerequisites: for example, validate a shipment number
before calling the shipment-status API. At runtime, the middleware automatically
runs the explicitly designated read-only validator. If it succeeds, the original
tool receives its original arguments and returns its original result. A failed or
unverifiable check blocks the target call with a structured error.

There are no new endpoint parameters, approval tokens, operation IDs, retail
backends, or endpoint implementations. The earlier retail-specific `tool_prevention` package is a separate component.

## Try it

Requires Python 3.10+; no third-party runtime dependencies.

```bash
python3 -m tool_middleware smoke
python3 -m tool_middleware evaluate
```

These commands use **fixture plans**, not LLM inference. Smoke shows:

```text
get_shipment_status("SHIP-123")
  → validate_shipment_number("SHIP-123") → valid
  → original get_shipment_status("SHIP-123") → unchanged result

get_shipment_status("BAD")
  → validate_shipment_number("BAD") → invalid
  → blocked; original status API is never called
```

Outputs include `catalog.json`, `plan.json`, `report.json`, and metadata-only
`audit.json` under `runs/middleware-smoke/`. Evaluation compares original and
wrapped tools for shipping and an unrelated employee/device catalog. These are
small engineering controls with intentionally permissive original endpoints,
not paper benchmark results or an estimate of enterprise-wide accuracy.

## Discover prerequisites with Ollama

The LLM is used during **setup** to discover dependencies. Normal calls run the
resulting inspectable checks without asking an LLM to make a fresh judgment.
Native Ollama discovery uses a catalog-specific [structured output schema](https://docs.ollama.com/capabilities/structured-outputs)
to constrain tool names, binding keys and output predicates; the compiler still
checks the proposed rules after generation.

```bash
# Generate the example catalog, without invoking an LLM.
python3 -m tool_middleware smoke

# Supply an installed model name; the config's default may not be installed.
python3 -m tool_middleware discover \
  --catalog runs/middleware-smoke/catalog.json \
  --validators validate_shipment_number \
  --config configs/dev_ollama.json \
  --model YOUR_INSTALLED_MODEL \
  --out runs/middleware-discovery

# Inspect coverage before applying the saved plan.
python3 -m tool_middleware inspect \
  --catalog runs/middleware-smoke/catalog.json \
  --plan runs/middleware-discovery/plan.json \
  --validators validate_shipment_number

# Apply that plan to the local example.
python3 -m tool_middleware smoke \
  --plan runs/middleware-discovery/plan.json \
  --out runs/middleware-with-discovered-plan
```

Discovery sends metadata to the configured local Ollama server, never invokes
enterprise tools, and saves the plan, original model response, model digest,
telemetry, coverage, and abstentions. A plan with zero rules leaves every tool
unguarded; inspect coverage instead of treating successful compilation as protection.

The application chooses which validators are safe to automate. The model cannot
designate its own tools as read-only. Both a host `read_only` declaration and the
explicit `validators` allowlist are required. The declaration is a host contract;
the middleware cannot prove that the endpoint truly has no side effects.

## Connect existing tools

A small neutral adapter describes existing callables:

```python
from tool_middleware import Tool, rewire, Blocked
from tool_middleware.discovery import discover
from agentrx.providers import OllamaProvider

# Existing functions remain unchanged.
# input_schema and output_schema describe their real contracts.
tools = [
    Tool("validate_shipment_number", "Check existence; valid=true means valid.",
         validator_input_schema, original_validate, validator_output_schema,
         effect="read_only"),
    Tool("get_shipment_status", "Requires a validated shipment number.",
         status_input_schema, original_get_status, status_output_schema),
]

model = OllamaProvider("YOUR_INSTALLED_MODEL")
model.prepare_for_run()
plan = discover(tools, model, validators=["validate_shipment_number"])
# Inspect/save the plan before activating it. Later, reload that same saved plan.
layer = rewire(tools, plan, validators=["validate_shipment_number"])

# Replace the agent-visible callable registry, keeping its original definitions.
agent_tools = layer.wrapped_tools()
try:
    result = agent_tools["get_shipment_status"](shipment_number="SHIP-123")
except Blocked as error:
    feedback_to_agent = error.to_dict()
```

The plan must have been discovered from this exact catalog, including descriptions,
schemas and effect declarations. Changed metadata invalidates its catalog hash.
The host must rediscover/review it. Handler implementations and their correspondence
to the declared schemas remain host responsibilities.

For an existing dispatcher instead of individual functions:

```python
from tool_middleware.adapters import from_dispatcher

tools = from_dispatcher(
    existing_tool_definitions,
    execute=original_dispatch,            # (tool_name, arguments) -> original result
    effects={"validate_shipment_number": "read_only"},
    output_schemas=known_output_schemas,  # needed when definitions omit them
)
layer = rewire(tools, saved_plan, validators=["validate_shipment_number"])
result = layer.dispatch("get_shipment_status", original_arguments)
```

The adapter accepts neutral, OpenAI-style, and MCP-style **metadata**. It does not
implement those network transports. Your executor supplies the actual dispatch and
must return the declared JSON response shape; transport envelopes need a host
adapter or schemas describing that envelope. Keep your original framework tool
definitions in place. The middleware changes dispatch behavior, not schemas.

Both synchronous and asynchronous handlers work. Use
`await layer.adispatch(name, arguments)` with async dispatchers. Wrapped handlers
become async when the target or a prerequisite needs it. Sync handlers, target
exceptions, and successful result objects retain their normal behavior.

## What a rule can express

```json
{
  "id": "shipment_is_valid",
  "target": "get_shipment_status",
  "validator": "validate_shipment_number",
  "bindings": {"shipment_number": {"from": "/shipment_number"}},
  "success": [{"pointer": "/valid", "equals": true}],
  "confidence": "high",
  "rationale": "Status requires an existing shipment number."
}
```

A full plan also includes `version: 1`, its `catalog_hash`, and an `abstentions`
list. Bindings copy target argument values or provide explicit schema-valid
literals; they do not repair/coerce the target's arguments. Predicates compare
literal JSON values on declared output fields. Boolean `true` is different from
numeric `1`. Shared declared fields such as `tenant_id`, including optional
validator fields, must retain matching bindings. Validator arguments are copied;
a validator that mutates those copies is rejected before its result becomes
evidence. Protected values are checked again immediately before forwarding, so
changes while an async validator is awaiting block the target.

Rules can chain validators. Cycles and chains deeper than eight tools are rejected.
A dispatch permits at most 16 automatic validation calls by default
(`max_validations` is configurable). Identical checks can share evidence within
one dispatch. Every new top-level call validates afresh; there is no cross-call
or cross-session evidence cache.

The compiler supports explicit JSON schema types, object properties/required
fields, arrays/items, enum/const, numeric bounds and string/array length bounds.
It does not interpret arbitrary JSON Schema extensions, `$ref`, regex patterns,
or generated code. Missing output schemas, unclear success semantics, unsupported
checks and low-confidence suggestions are reported as gaps/abstentions. Bound
source paths and predicate paths must be required in their declared schemas; optional
target paths are currently unsupported. Tools
without applicable rules continue unchanged and are listed as unguarded.

## Conservative rollout and limits

Use `mode="shadow"` to log where guards would apply while forwarding the original
call without additional validator traffic or blocking. In `enforce` mode, a false
predicate, missing/malformed validator result, validator failure, or exhausted
budget blocks the target. No target retry or compensation is attempted.

Catalog descriptions and model output are untrusted input. Compilation checks
structure, references, schema compatibility, scope bindings and automatic-tool
permission. It cannot prove an LLM's proposed dependency is semantically correct.
Review the concrete plan and test representative valid calls before deployment;
a wrong dependency can overblock an otherwise legitimate request.

This layer provides conservative preflight checks, not complete correctness.
State can change between a validator and a remote target call. Reasoning mistakes,
missing business rules, infrastructure errors and unguarded tools remain possible.
All protected calls must pass through this dispatcher; keep raw privileged access
outside the agent's interface. This is not an OS sandbox or a replacement for an
endpoint's own authorization and consistency checks.
