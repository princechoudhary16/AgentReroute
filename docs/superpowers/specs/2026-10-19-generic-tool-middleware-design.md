# Generic conservative tool middleware

## Corrected intent and authorization

The user clarified that the requested product is a generic middleware that organizes
existing enterprise tools for agent use. Retail was an example, not the architecture.
The earlier retail policy/capability system did not satisfy that central requirement.
Keep it as a separately labeled experiment; make the new generic middleware the main
prevention entry point. The original conditional build authorization remains active.
The user explicitly selected automatic execution of designated read-only validators.

The middleware receives a tool catalog and callable/dispatch adapters, uses an LLM
to propose straightforward prerequisite relationships, and rewires the exposed tool
registry. It preserves endpoint names, signatures, original argument values and
successful return values. No new approval tokens, operation IDs, backend transactions,
retail database or modified endpoint contracts are required.

## Architecture

1. A `Tool` contains name, description, original input JSON schema, optional output
   schema, a callable handler and host-declared effect (`read_only`, `write`, `unknown`).
2. LLM discovery consumes metadata only and a host-selected validator allowlist.
   It emits an inspectable, versioned dependency plan pinned to the catalog hash,
   plus abstentions. It never invokes a tool, supplies executable code, changes an
   API schema, or grants itself permission to designate a validator.
3. A compiler accepts high-confidence, structurally supported rules. Each names a
   target, a designated read-only prerequisite, bindings from target argument JSON
   pointers to prerequisite argument names (or schema-validated literal constants),
   explicit literal-equality predicates on declared output fields, and a rationale.
4. Rewiring replaces only exposed callables. Before a protected target executes,
   the middleware obtains its prerequisite arguments, executes that prerequisite,
   and checks the declared success predicates. False/missing/malformed evidence or
   validator exceptions block the target with a structured `Blocked` exception.
   A successful target receives unchanged arguments and its exact result is returned.

The main example is validate_shipment_number(shipment_number) -> {valid: bool}
preceding get_shipment_status(shipment_number). Demonstrate a second unrelated IT
catalog (verify_employee(employee_id) before retrieve_employee_devices) using the
same engine. No tool-name or domain-specific runtime logic.

## Conservative behavior and explicit choices

These are assistant implementation choices under the user's corrected scope:

- Discovery produces a plan first; applying that concrete plan is a separate API
  operation. No model inference occurs during a normal protected tool call.
- Rules must be grounded in available metadata; insufficient output semantics or
  ambiguous dependencies are reported as abstentions, leaving original tools as
  pass-through. Structural compilation cannot prove an LLM's semantic suggestion.
- Explicit host allowlisting AND `read_only` declaration are both required before
  any automatic validator invocation. Other tools are never auto-executed.
- No evidence cache across top-level calls. Run validators fresh, avoiding stale
  evidence and cross-session assumptions. Identical validator calls may be reused
  within one dispatch only. No blanket mutation invalidation/idempotency claims.
- Support prerequisite chains but reject cycles, malformed/missing tool references,
  invalid field mappings, unsupported evidence predicates and excessive depth/cost.
  A per-dispatch limit prevents an unexpectedly large validator fan-out.
- Both sync and async callable registries are supported. An opaque executor adapter
  can forward names/arguments to an existing RPC/MCP/framework dispatcher; the
  middleware itself needs no framework or enterprise credentials.
- Provide `enforce` and `shadow` modes. Shadow logs proposed checks without invoking
  additional validators, blocking, or changing the original call. This avoids
  surprising extra traffic during deployment inspection.
- An unsupported/unknown tool with no applicable rule remains unchanged and is
  visible as unguarded coverage. Unknown original input schema features need not
  prevent pass-through; strict schema/type/path checks apply to activated rules.
- Output schemas are required for machine-checkable success fields; original tools
  lacking them are surfaced as discovery/coverage gaps rather than guessed.
- Runtime supports declarative equality and field binding only. No LLM-generated
  Python, regex checks, argument repair/coercion, prompt-based permission, general
  semantic judge, or automatic target retry/compensation.
- Host must replace the agent-visible registry and keep raw dispatch access private.
  Read-only declarations, output meanings and tool implementations are trusted host
  contracts. This wrapper is neither an OS sandbox nor a proof of business correctness.
- Validation can become stale between two remote API calls. The layer provides
  conservative preflight checks, not transactional correctness or complete failure
  prevention. It cannot infer all enterprise policies from sparse descriptions.

## Inputs and rule shape

Tool input/output schemas remain unmodified. Initial binding compilation supports
JSON objects/properties, required paths and scalar/array typed values. Clearly report
unsupported constructs; never treat an unsupported check as a successful validation.
Rules must cover every required prerequisite argument; shared declared top-level
input fields, including optional validator fields, must retain matching bindings so tenant/customer scope is not silently
lost. Runtime arguments must satisfy the fields used in bindings. Output predicates
must refer to declared typed fields; exact JSON equality distinguishes true from 1.

Plan example (catalog hash omitted here):

    {"version": 1, "rules": [{"id": "shipment_is_valid",
      "target": "get_shipment_status", "validator": "validate_shipment_number",
      "bindings": {"shipment_number": {"from": "/shipment_number"}},
      "success": [{"pointer": "/valid", "equals": true}],
      "confidence": "high", "rationale": "Status requires a validated shipment number."}],
      "abstentions": []}

The builder may settle exact class/function names, packaging and schema subset,
provided unsupported cases are explicit and API preservation holds.

## Deliverables and verification

- `tool_middleware` framework-independent package: catalog/plan compiler, LLM
  discovery, sync/async interceptor and original-shape callable registry.
- Native local Ollama discovery using the project's existing verified client;
  preserve all existing diagnosis prompts and behavior. A small generic JSON-chat
  method may be extracted from the provider to avoid coupling to diagnosis stages.
- CLI: metadata-only discover, inspect/compile plan and executable offline shipment
  smoke/evaluation. Fixture discovery is explicitly labeled; verify actual local
  LLM discovery separately when the running endpoint/model is available.
- Tests: exact original arguments/result/schema preservation, failed validator means
  zero target calls, validator exceptions and malformed results, correct field and
  tenant binding, low-confidence/unsupported suggestions, no unauthorized automatic
  tool calls, unrelated tools pass through, fresh checks, chaining/cycle/fan-out
  limits, sync/async parity, provider regression and CLI/package install.
- Controlled comparisons: original vs wrapped tools on valid and invalid arguments
  in unrelated catalogs. Report target attempts, backend-observed invalid invocations,
  valid completions, extra validator calls and false blocks. Do not claim paper
  benchmark results or universal prevention.
- Rewrite docs/prevention.md and README around the generic middleware, clearly
  distinguishing the earlier retail prototype and the remaining semantic limits.

## Review focus

Incorrect LLM dependencies can overblock legitimate calls. Misdeclared validator
side effects can cause unintended actions. Missing tenant bindings can validate the
wrong entity. Schema/path uncertainty must never become evidence. Runtime errors,
mutable data, async execution and shared instances must preserve the original API
behavior and isolate one dispatch's evidence from another.
