# Enforced tool prevention for AgentReroute

## Decision and purpose

The user requested simultaneous feasibility evaluation and conditional construction
of a complete prevention system, delegating the go/no-go decision to the assistant.
They selected **this project's existing agent system** as the first target.
Assessment: a bounded deterministic tool gateway is feasible, appropriate, and a
useful project contribution. Proceed with it under that authorization. The choices
below are implementation decisions, not claims that the user selected every detail.

This system mechanically replaces a declared tool registry after validating its
explicit policy/backend contract. It does not infer all business rules from
arbitrary source code or promise correct reasoning. AgentRx remains the diagnostic
component for residual mistakes.

## Existing-system boundary

`origin/orchestrator` currently exposes `ReplayEnvironment.rollout`,
`ScriptedEnvironment`, and a `TauBenchEnvironment` whose constructor raises
`NotImplementedError`. There is no live retail tool dispatcher to patch yet.
Intercepting completed replay steps cannot prevent effects that already happened.

Deliver a `tool_prevention` package with an adapter/dispatcher interface suitable
for that future backend and a complete persistent local retail implementation.
Do not merge the orchestrator branch or silently change the collaborator's recovery
logic. A replay-only environment supplied to `rewire` must produce an explicit
compatibility error identifying the missing pre-execution interface.

The packaged AgentRx tool descriptions are not executable schemas and differ from
actual retail trace signatures. Prevention accepts authoritative adapter schemas,
not those descriptive resources. Unsupported tools, fields, policies, or backend
guarantees must fail compilation rather than guess.

## Input and public API

`rewire(system, policy, store_path) -> ProtectedSystem` accepts:

- A trusted system adapter exposing `backend_identity`, `tool_specs`, `read`, `snapshot`,
  `execute_atomic`, and `lookup_execution`.
- A versioned JSON manifest declaring read tools and protected mutation rules.
- A SQLite path for persistent approval requests, capability hashes, operations,
  session obligations, and audit events.

The adapter declares conditional writes and durable idempotency support. Its stable
backend identity binds the persistent grant journal to one backend instance. It must
execute validated actions atomically against resource versions and must not expose
raw mutation functions/credentials to the agent. A Python wrapper alone is not an
OS security boundary against arbitrary agent code or shell/network access.

`protected.tools(context)` returns replacement JSON tool definitions and callables
bound to a trusted context. `protected.call(context, tool, arguments, operation_id,
approval_token)` dispatches calls through validation and policy enforcement.
Context includes a host-established customer identity and session ID; these are
not taken from agent-controlled tool arguments. Credentials/authentication remain
host responsibilities. Explicit completion obligations may be registered by the
host; `complete_task` refuses completion while they are unmet.

An additional agent-visible `request_action_approval(tool_name, arguments)` tool
creates an immutable pending request; it does not grant itself approval.
A separate trusted administrator interface approves or denies a request after
showing the exact action, customer, resources, amounts, and payment destination.
The grant is returned through that trusted interface, not included in audit logs.

## Enforced policy and action planning

The initial manifest supports this project's retail scope: ownership, authentication,
allowed order states, required exact-action approval, matching item/product types,
valid payment destinations, inventory availability, and once-per-order restrictions.
Validate tool arguments against declared JSON schemas before touching the backend.
No executable policy expressions or LLM approval decisions.

Plans bind the operation name, canonical arguments, customer/session, policy
version, authoritative quote, and all relevant resource versions. Money uses integer
minor units. Item counts, identifiers, replacement products, and payment destinations
come from authoritative backend data, not the agent's natural-language claims.

The local backend exposes retail-style read and mutation tools, using actual field
names such as `order_id,item_ids,new_item_ids,payment_method_id` and flat address
fields. All declared exposed tools need a policy classification; unknown mutations
are not passed through. Supported operations and omissions are reported in a
compilation coverage report.

## Persistent approval and execution lifecycle

Capabilities are high-entropy opaque random tokens, stored only as cryptographic
hashes. Each binds exact canonical plan content, principal/session, policy version,
resource versions, and expiration. Changing any bound field requires new approval.
The agent can request a grant but cannot call the trusted approve function.

Before mutation, use a SQLite immediate transaction to claim a valid single-use
capability and insert a durable running operation. A same-session retry with the
same operation ID and same action returns a cached successful result. Reusing an
operation ID for different arguments, a token for another session, an expired or
consumed token, or stale resource versions is denied before execution.

The backend then applies a conditional/idempotent mutation. It must compare the
plan's complete resource-version map atomically with the write. A stale state is a
definitive rejection. A timeout/unknown execution outcome becomes **indeterminate**;
the capability is not released and automatic retries are blocked. A trusted
reconciliation call may query the backend's durable operation record and recover
a confirmed result. A local lock does not create generic distributed exactly-once
execution guarantees.

Concurrency must allow at most one side effect per capability/operation even when
multiple gateway instances share a store. Restarting the gateway must not reset
consumption or successful result caching. Audit calls, decisions, request state,
and execution outcomes without recording raw tokens.

## Demonstration and evaluation

Deliver runnable local CLI commands for a smoke test, policy compilation report,
trusted approval, controlled evaluation, and audit/AgentRx-compatible trace export.
Demo code may act as a trusted human to approve an exact action, clearly labeled;
the agent-visible registry never includes that privilege. A full local walkthrough
must show rejected calls leaving state unchanged and an approved action executing
once, including an idempotent retry.

Evaluate executable controlled cases under unprotected, schema-only, and full
prevention conditions with equivalent starting state. Include valid controls and
invalid/forged/mismatched/expired/replayed approval attempts, cross-customer access,
stale state, concurrency, restart, same-ID/different-arguments, product/payment
mismatch, already-processed orders, omissions against explicit host obligations,
and ambiguous backend completion with reconciliation. Report side effects,
valid completion, false rejections, duplicate effects, and latency separately.

The available corpus has 29 retail traces (24 with mutation calls) and 44 Magentic
traces without structured tool calls. Retrospective screening can report tool
coverage and known/unknown applicability; missing cryptographic approval metadata
in a historical transcript is **unknown**, not a proven unauthorized action.
Never equate a blocked historical call with a recovered task or a successful live
counterfactual. Original live tau task evaluation remains blocked by the missing
execution backend. Labels stay outside runtime policy and approval logic.

## Contribution and prior art

Capability/security and runtime policy enforcement are established approaches.
AgentSpec already specifies and enforces agent runtime constraints:
https://arxiv.org/abs/2503.18666 . OPA separates policy decisions and enforcement:
https://www.openpolicyagent.org/docs . Cedar validates application/policy schemas:
https://docs.cedarpolicy.com/policies/validation.html . No general novelty claim.

The project contribution is an explicit adapter-to-enforced-registry transformation,
persistent scoped capabilities and execution semantics, an executable evaluation,
and a clear account of which failures remain for AgentRx. Deterministic protection
of declared operations and post-hoc diagnosis are complementary outcomes.

## Non-goals and limits

No guarantee of correct interpretation, correct authorized order selection,
arbitrary task completion, or prevention of infrastructure failures. No unrestricted
LLM policy synthesis, AST rewriting of arbitrary projects, or unmediated shell
execution. Raw privileged backends and administrator approval APIs are trusted
host-side objects and must be isolated from agent-visible interfaces.

## Implementation discretion

Use Python standard library and SQLite; no cloud service required. The builder
may choose module layout, schema validator subset, policy JSON layout, concrete
adapter dataclasses, fixture catalog, and CLI command names, provided the contracts
and evaluation distinctions above hold. Keep prevention independent of AgentRx
model providers and do not modify measured diagnostic prompts to improve scores.
