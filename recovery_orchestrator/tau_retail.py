"""tau-bench retail: decision-safe task loading and the (not yet wired) replay backend.

FA's ``trajectory_id`` equals the tau ``task_id``. The raw file
``tau_dataset_failed.json`` also holds gold ``actions``, ``outputs`` and
``reward``; this module returns only the instruction, the policy (system
prompt) and the message history. Gold data is loaded only in ``evaluation``.
"""
from __future__ import annotations

import json
from pathlib import Path

from .replay import TaskView
from .trace import Trace

DOMAIN = 'tau'

# Tools that change the database. In a live deployment these steps cannot be
# rolled back, so they are non-rollback points (see ``EVConfig.respect_irreversible``).
WRITE_TOOLS = frozenset({
    'cancel_pending_order', 'modify_pending_order_address', 'modify_pending_order_items',
    'modify_pending_order_payment', 'modify_user_address', 'return_delivered_order_items',
    'exchange_delivered_order_items', 'transfer_to_human_agents',
})


def _records(raw_path) -> list[dict]:
    return json.loads(Path(raw_path).read_text(encoding='utf-8-sig'))


def task_view(record: dict) -> TaskView:
    traj = record.get('traj') or []
    policy = traj[0].get('content', '') if traj and traj[0].get('role') == 'system' else ''
    return TaskView(str(record['task_id']), DOMAIN,
                    instruction=record.get('info', {}).get('task', {}).get('instruction', ''),
                    policy=policy)


def trace_from_record(record: dict) -> Trace:
    """Message history only; step_index == 1-based message position, as in FA's IR."""
    view = task_view(record)
    messages = [{k: v for k, v in m.items() if k in ('role', 'content', 'tool_calls', 'name')}
                for m in record.get('traj') or []]
    return Trace.from_messages(view.trajectory_id, DOMAIN, messages, view.instruction)


def load_tasks(raw_path) -> dict[str, TaskView]:
    return {str(r['task_id']): task_view(r) for r in _records(raw_path)}


def load_traces(raw_path) -> dict[str, Trace]:
    return {str(r['task_id']): trace_from_record(r) for r in _records(raw_path)}


def task_index(trajectory_id) -> int:
    return int(trajectory_id)


class TauBenchEnvironment:
    """Live tau-bench retail replay. Not implemented until the team agrees on the dependency.

    Planned implementation of ``rollout(trace, prefix, directive, monitor, max_tokens)``:

    1. ``env = tau_bench.envs.get_env('retail', user_strategy='llm', user_model=...,
       task_split='test', task_index=task_index(trajectory_id))`` and ``env.reset(index)``.
    2. Rebuild environment state by re-executing ``prefix.tool_calls`` with
       ``env.step(Action(name, kwargs))``; check each output matches the original.
    3. Restore the user simulator's history from the prefix's user/assistant turns
       (needs a small subclass of tau's LLM user simulator).
    4. Agent messages = prefix messages + ``directive.as_step(...)`` (if any); continue
       the tool-calling agent loop, feeding each new step to ``monitor``.
    5. Write ``env``'s final reward to ``self.ledger`` only; return a ``Rollout`` without it.

    The user simulator is an LLM and the replay agent may differ from the original
    one, so always run the unmodified-replay control alongside.
    """
    write_tools = WRITE_TOOLS

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('tau-bench replay is not wired yet; use ScriptedEnvironment '
                                  'offline. See the class docstring for the plan.')
