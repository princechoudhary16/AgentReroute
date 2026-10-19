# Generic Tool Middleware Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline task-by-task. Steps use checkbox tracking. Original conditional build authorization and the user's clarified automatic-validator behavior apply; do not repeat permission requests.

**Goal:** Rewire arbitrary declared tool registries with conservative, LLM-discovered, read-only prerequisite checks while preserving existing APIs.
**Architecture:** Metadata produces an inspectable catalog-pinned dependency plan. A strict compiler checks field bindings, output predicates and explicit validator designation. A generic sync/async dispatcher applies prerequisites before forwarding original target calls.
**Tech Stack:** Python >=3.10, stdlib, existing native Ollama client; no new runtime dependencies.
**Spec:** docs/superpowers/specs/2026-10-19-generic-tool-middleware-design.md

## Global Constraints

- Preserve tool names, original input schemas, original target arguments and successful return values.
- No automatic non-designated or non-read-only tool invocation, injected tokens, backend rewrites or domain-specific runtime names.
- Model suggestions cannot execute code or designate their own validators. Apply a concrete plan separately.
- Check validators fresh per top-level call; no shared/cross-call evidence cache.
- Unsupported metadata/low-confidence relationships abstain; invalid activated plans fail explicitly.
- Reuse the existing diagnosis client without changing diagnosis prompts or stage behavior.

## Review Focus

- Same-name identifiers from different tenants must not share validation evidence.
- A validator that raises, returns wrong types, omits the success field, or mutates its inputs must not allow the target or alter the caller's arguments.
- Async handlers must not be accidentally called by sync dispatch and leave unawaited work.
- Model-selected names, bindings, predicate values and catalog changes must not bypass compilation.
- Chains, repeated prerequisites and concurrent top-level dispatches must respect budgets and keep evidence local.

### Task 1: Generic catalog compiler and conservative interceptor

**Files:** tool_middleware/{__init__,catalog,runtime}.py, tests/test_tool_middleware.py.
**Interfaces:** Tool(name,description,input_schema,handler,output_schema=None,effect='unknown'); compile_plan(tools,plan,validators); Middleware(tools,plan,validators,mode='enforce'); dispatch/adispatch and wrapped_tools preserving original definitions.

- [x] Write and run failing behavior tests for the shipment prerequisite and a second unrelated catalog, unchanged API/results, invalid values, original exceptions and unrelated pass-through.
- [x] Implement catalog fingerprints, supported JSON-pointer/type binding checks, exact literal predicates, host designation, cycles/scope checks and clear coverage.
- [x] Implement fresh bounded prerequisite dispatch, sync/async parity, local deduplication, immutable validator input copies, shadow without extra calls, structured blocking and metadata-only audits.
- [x] Add/run red-green boundary tests for tenant matching, malformed evidence, unauthorized automatic writes, stale catalogs, ambiguous confidence, cycles, budgets and concurrent calls.
- [x] Run task tests, record result and commit.

### Task 2: LLM-assisted metadata discovery

**Files:** tool_middleware/discovery.py, agentrx/providers.py, tests/test_tool_discovery.py, tests/test_providers.py.
**Interfaces:** discover(tools,model,validators)->versioned catalog-pinned plan with abstentions. Model supplies complete_messages(messages)->JSON object. OllamaProvider.complete continues existing stages unchanged and delegates to complete_messages.

- [x] Write/run failing discovery tests for actual request metadata, conservative prompt, schema validation, invented/low-confidence rules, no handler execution and model errors.
- [x] Extract generic JSON chat from the native Ollama client, preserving all transport/response checks and existing stage behavior.
- [x] Implement typed proposal filtering/compilation and coverage/abstention reports; no model output is executable code or runtime authority.
- [x] Run discovery and existing provider tests, record result and commit.

### Task 3: Runnable examples, evaluation, documentation and package verification

**Files:** tool_middleware/{examples,cli,__main__}.py, tests/test_tool_middleware_cli.py, pyproject.toml, README.md, docs/{prevention,verification}.md.
**Interfaces:** python -m tool_middleware smoke/evaluate/discover/inspect; synthetic shipment and unrelated IT tools; examples explicitly distinguish fixture vs actual Ollama discovery.

- [ ] Write/run failing CLI tests for complete automatic validation and target-call counters, metadata-only discovery and inspectable plan artifacts.
- [ ] Implement end-to-end examples and original-vs-wrapped comparisons with valid completion, invalid backend invocation, validator calls and false-block counts.
- [ ] Verify actual local Ollama metadata discovery if available; record failures/abstentions honestly instead of changing constraints to force results.
- [ ] Rewrite primary prevention docs around generic middleware and label retail as a separate experiment.
- [ ] Run full suite, isolated wheel install/smoke, fresh whole-change review, red-green fixes, final checks and authorized branch push.
