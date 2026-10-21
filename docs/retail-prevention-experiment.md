# Earlier retail prevention experiment

This retail-specific experiment is separate from the [generic middleware](prevention.md).
It is preserved for comparison and is not the generic product.

The `tool_prevention` package checks proposed tool calls **before** they change
state. AgentRx diagnoses saved runs; this component prevents declared tool-policy
violations during execution. It runs offline without an LLM or API key.

```bash
python3 -m tool_prevention smoke --out runs/prevention-smoke
python3 -m tool_prevention evaluate --out runs/prevention-evaluation
python3 -m tool_prevention screen --corpus data/benchmark
python3 -m tool_prevention compile
```

`smoke` rejects an unapproved cancellation, obtains a simulated customer approval
through the trusted host interface, executes once, retries without duplicating the
write, and finishes an explicit obligation. It writes `report.json`, `audit.json`,
and an AgentRx-compatible `trace.json`. Each demo/evaluation uses a fresh database.

## What gets rewired

```python
from tool_prevention import Context, rewire
from tool_prevention.retail import RetailSystem
from tool_prevention.adapters import ProtectedDispatcher

# Trusted application startup; these privileged objects stay outside the agent.
backend = RetailSystem.create("runs/my-system/retail.db")
protected = rewire(backend, store_path="runs/my-system/grants.db")
context = Context(session_id="session-1", customer_id="alice")

# Give only these definitions/callables to the tool-calling agent.
definitions = protected.tool_definitions()
tools = protected.tools(context)
dispatcher = ProtectedDispatcher(protected, context)
# dispatcher.step(Action(name="...", kwargs={...}))
```

The compiler validates the adapter's actual schemas against the explicit retail
policy. The replacement registry contains six read tools, seven protected mutation
tools, `request_action_approval`, and `complete_task`. Unknown tools, incompatible
schemas, unclassified tools, and unsupported backend guarantees fail compilation.
The current compiler supports the retail policy family; it does not discover
business rules automatically from arbitrary application code.

A compatible backend supplies:

| Interface | Required guarantee |
| --- | --- |
| `backend_identity()` | Persistent identity of this backend instance |
| `tool_specs()` | Authoritative name/schema/effect declarations |
| `read(context, name, arguments)` | Ownership-safe reads without side effects |
| `snapshot()` | Authoritative resources and versions |
| `execute_atomic(plan, operation_id)` | Atomic version checks, effects and durable idempotency record |
| `lookup_execution(operation_id)` | Confirmed durable execution evidence or unknown |

It must declare `conditional_writes` and `durable_idempotency`. The grant journal
is bound to the backend identity, preventing accidental reuse against a fresh DB.
The local SQLite retail backend implements this contract, including a simulated
payment ledger. It does not connect to payment processors, shipping, or tau-bench.
Returns/exchanges are modeled locally as requests with immediate simulated ledger
entries; this is an executable reference model, not full tau-bench semantics.

## Approval flow

```python
arguments = {"order_id": "pending-alice", "reason": "no longer needed"}
request = tools["request_action_approval"](
    tool_name="cancel_pending_order", arguments=arguments)

# Separate trusted customer/host channel: show the exact quote and action.
preview = protected.admin.preview(request["request_id"])
# After the authenticated customer approves that exact action:
grant = protected.admin.approve(request["request_id"])

result = tools["cancel_pending_order"](
    **arguments, operation_id="cancel-1", approval_token=grant["approval_token"])
```

`request_action_approval` creates a pending request; it cannot approve itself.
Approving/denying/reconciling are trusted host operations absent from the agent's
registry. In an application, deliver the grant through that trusted channel only
after real approval. Demo code explicitly simulates this external decision.

The opaque token binds the customer, session, exact arguments, policy version,
quote/payment destination, and resource versions. It expires and can be consumed
once. Changing the amount, order, session, or current resource state requires a
new request. Tokens are stored as hashes and redacted from structured audit data.
Sensitive logs and databases still require normal host access controls.

A stable `operation_id` identifies one action. Repeating it returns the stored
result; changing its arguments is rejected. Concurrent calls and restarts share
the persistent journal. If a backend might have committed before a timeout, the
operation stays `indeterminate`; retries cannot execute it again. The host can use
`protected.admin.reconcile(context, operation_id)` to recover confirmed success.
Success remains terminal even if a delayed caller later reports a timeout.

CLI equivalents use the same persistent databases:

```bash
python3 -m tool_prevention request cancel_pending_order --session session-1 --customer alice \
  --arguments '{"order_id":"pending-alice","reason":"no longer needed"}'
python3 -m tool_prevention approve REQUEST_ID              # preview
python3 -m tool_prevention approve REQUEST_ID --decision approve
# The last command outputs a sensitive token to the trusted operator.
# Supply it through PREVENTION_APPROVAL_TOKEN in your local environment.
python3 -m tool_prevention call cancel_pending_order --session session-1 --customer alice \
  --operation-id cancel-1 --arguments '{"order_id":"pending-alice","reason":"no longer needed"}'
python3 -m tool_prevention audit
python3 -m tool_prevention audit --trace --session session-1 --customer alice
```

Use `--state`, `--store`, and optionally `--policy` consistently to select another
system. The CLI's `--customer` is a trusted operator input, not authentication.
Do not expose these commands or the admin object directly to the LLM.

## Enforced rules and limits

The initial policy covers host authentication, ownership, exact-action approval,
order status, item quantities, same-product replacements, inventory availability,
owned payment methods, refund destinations, gift-card funds, and once-per-order
item modifications/returns/exchanges. Money uses integer cents. Per-order funding
allocations preserve mixed payment sources: cancellation refunds those sources, and
payment changes refund all prior allocations and charge the new method atomically.

The host can register `protected.require_action(context, tool, arguments)`;
`complete_task` then refuses completion until that exact action succeeds. This
catches explicit finite omissions. It cannot infer every obligation from a user's
natural-language goal or prove the quality of an answer.

An approved but mistaken order choice can still be valid under all these rules.
Misinterpretation, unsupported reasoning, incomplete research, and infrastructure
failures remain possible. AgentRx can still analyze the resulting audit trace.

All side effects must go through the protected dispatcher. Keep raw backend tools,
credentials, the journal, and approval authority outside the agent's accessible
process/interface. This Python registry is not an OS sandbox against arbitrary
agent code, shell access, or direct network access. Adapter guarantees and trusted
host identity are prerequisites, not properties the compiler can prove by itself.

## This project's integration boundary

On `origin/orchestrator`, `TauBenchEnvironment.__init__` currently raises
`NotImplementedError`; `ScriptedEnvironment.rollout` produces already executed
steps. Neither offers a live pre-execution tool boundary that can be safely patched.
`rewire` explicitly rejects a replay-only object. `ProtectedDispatcher.step` accepts
the planned `Action(name, kwargs)` shape and enforces calls before the backend.
Install it where a future live agent dispatches proposed tool actions. The
collaborator's orchestrator branch has not been merged or altered.

The diagnosis package's descriptive tool registry is unsuitable as executable
schemas. This component uses its adapter contract and actual retail field names.
Original live tau-bench task evaluation still needs the missing execution adapter.

## Evaluation and research contribution

`evaluate` executes a controlled local comparison of **unprotected**, **schema-only**,
and **full prevention** conditions from equivalent initial states. The first two
use a deliberately unchecked reference writer for cancellation/item modification;
these baselines do not claim that the original tau-bench API has those weaknesses.
All conditions retain SQLite conditional writes and share simulated settlement
semantics. Full policy enforcement is the variable being tested. Valid completion
checks the resulting order and refund state separately from dispatch success.
Successful controls, unsafe/duplicate writes, completion
omissions and local dispatch latency are reported separately. Timing excludes
user approval and LLM time; concurrent baseline write counts can vary with scheduling.
Other retail mutations have valid-control tests in the unit suite.

`screen` inspects the available 73 historical traces without loading failure labels
into policy decisions. Missing capability evidence is **unknown**, not a proven
violation. Screening stops at the first definitive schema rejection and does not
invent the remaining execution. Magentic traces without structured calls are
reported as unsupported for this form of enforcement. Neither a recovered-task
rate nor a paper-benchmark prevention score can be obtained from these logs alone.

This is a useful engineering contribution and an evaluable project architecture,
not a claim to invent runtime enforcement. Related prior work includes
[AgentSpec](https://arxiv.org/abs/2503.18666),
[OPA's policy/enforcement separation](https://www.openpolicyagent.org/docs), and
[Cedar schema validation](https://docs.cedarpolicy.com/policies/validation.html).
The project's research question is how much observed failure can be covered by
explicit deterministic policies, with what false rejections and integration cost,
and which failures still need diagnosis. Live tasks, matched controls, additional
policy families and model comparisons are needed to answer that empirically.
