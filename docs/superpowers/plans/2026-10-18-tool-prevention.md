# Tool Prevention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task; the user explicitly delegated the feasibility/build decision and selected this project's agent system.

**Goal:** Compile a declared retail tool system into an enforced registry with persistent scoped approvals and auditable execution.
**Architecture:** A trusted adapter supplies schemas, authoritative snapshots, conditional mutations, and idempotency lookup. A policy compiler, SQLite grant journal, and dispatcher enforce preconditions before tool effects. A local persistent retail backend makes the entire system executable while the collaborator's live backend remains absent.
**Tech Stack:** Python >=3.10, sqlite3, unittest; no runtime dependencies.
**Spec:** docs/superpowers/specs/2026-10-18-tool-prevention-design.md

## Global Constraints

- Never infer authority from an LLM, a tool name, or a recorded transcript affirmation.
- Agent-visible tools cannot grant their own approvals or access raw mutation methods.
- Reject unsupported schemas/policies/backend guarantees before exposing tools.
- Persist capability claims and execution identity; ambiguous results remain indeterminate.
- Preserve the existing diagnosis pipeline and distinguish controlled execution from trace screening.

## Review Focus

- Concurrent/restarted gateways must not duplicate effects or resurrect used grants.
- Mutations after approval must fail conditional version checks.
- A returned timeout may hide a committed side effect; never blindly retry it.
- Adapter/schema mismatch must be reported rather than guessed from AgentRx metadata.
- Historical missing approval records cannot be counted as proven violations or recovery.

### Task 1: Policy contracts and local retail backend

**Files:** tool_prevention/{__init__,contracts,policy,retail}.py, resources/retail_policy.json, tests/test_prevention_policy.py.
**Interfaces:** trusted Context, ToolSpec, ActionPlan; compile_policy(system, manifest) -> compiled policy/coverage; RetailSystem snapshot/read/execute_atomic/lookup_execution.

- [x] Write tests for unsupported adapters/schemas/policy tools, strict argument types, correct ownership/status/product/payment rules, and stale conditional mutation.
- [x] Observe missing-behavior failures.
- [x] Implement explicit contracts, policy planning, backend schemas and durable conditional/idempotent local mutations.
- [x] Run policy/backend tests to green.

### Task 2: Persistent grants and protected tool registry

**Files:** tool_prevention/{store,gateway}.py, tests/test_prevention_gateway.py.
**Interfaces:** rewire(system, policy, store_path) -> ProtectedSystem; tools(context), call(context, tool, args, operation_id, approval_token); trusted approve/deny/reconcile; optional explicit completion obligations.

- [x] Write failing tests for absent/forged/wrong-session/expired/changed/used grants, successful exact approval, idempotent retry, concurrent claims, restart, indeterminate result, and reconciliation.
- [x] Implement SQLite journal, immutable bound requests, hashed tokens, guarded reads/writes, protected callable registry, and redacted audit/trace export.
- [x] Verify all denials cause zero effects and successful retries do not duplicate effects.

### Task 3: Runnable demonstrations, evaluation, and project adapter

**Files:** tool_prevention/{cli,__main__,evaluation,adapters}.py, tests/test_prevention_cli.py, docs/prevention.md, pyproject.toml, README.md.
**Interfaces:** CLI smoke/compile/evaluate/screen/approve/audit; adapter rejects replay-only environments and offers dispatch hook for future live tool backend.

- [x] Write failing end-to-end CLI and evaluation tests; assert valid controls, blocked side effects, no duplicate effects, and historical unknowns.
- [x] Implement executable local scenarios and separate trace coverage screening; add AgentRx IR export.
- [x] Document trusted deployment boundary, exact setup, original backend limitation, and prior art.
- [x] Run full existing/new suite, package build, smoke/evaluation, independent review, and regression fixes.
