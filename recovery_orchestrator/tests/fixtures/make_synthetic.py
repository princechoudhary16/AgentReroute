"""Regenerate the synthetic fixtures (FA-format run, gold cases, tau raw file).

    python3 recovery_orchestrator/tests/fixtures/make_synthetic.py

Synthetic, hand-written data in the exact file formats FA writes. Gold-only
values carry the marker GOLDSENTINEL so leakage tests can search for it.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
TID = 'syn-1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


ORDER = {'order_id': '#W1', 'status': 'delivered',
         'items': [{'item_id': '111', 'name': 'Headphones'}, {'item_id': '222', 'name': 'Watch'}]}
MESSAGES = [
    {'role': 'system', 'content': '# Retail agent policy\nAuthenticate the user first. Confirm before writes.'},
    {'role': 'user', 'content': 'I want to return my headphones from order #W1. My email is a@b.c.'},
    {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'c1', 'type': 'function', 'function': {'name': 'find_user_id_by_email', 'arguments': '{"email": "a@b.c"}'}}]},
    {'role': 'tool', 'tool_call_id': 'c1', 'name': 'find_user_id_by_email', 'content': 'u1'},
    {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'c2', 'type': 'function', 'function': {'name': 'get_order_details', 'arguments': '{"order_id": "#W1"}'}}]},
    {'role': 'tool', 'tool_call_id': 'c2', 'name': 'get_order_details', 'content': json.dumps(ORDER)},
    {'role': 'assistant', 'content': 'Order #W1 only contains a Watch. Shall I return the Watch?'},
    {'role': 'user', 'content': 'Oh. Ok, return the watch then.'},
    {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'c3', 'type': 'function', 'function': {'name': 'return_delivered_order_items',
                                                      'arguments': '{"order_id": "#W1", "item_ids": ["222"]}'}}]},
    {'role': 'tool', 'tool_call_id': 'c3', 'name': 'return_delivered_order_items', 'content': '{"status": "return requested"}'},
    {'role': 'assistant', 'content': 'Done, the watch return is requested.'},
]


def ir():
    steps, index_map = [], []
    for i, m in enumerate(MESSAGES, start=1):
        sub = {'sub_index': 1, 'role': m['role'], 'content': m['content']}
        if m['role'] == 'tool':
            sub['name'] = m['name']
        if m.get('tool_calls'):
            sub['tool_calls'] = [{'id': c['id'], 'name': c['function']['name'],
                                  'arguments': json.loads(c['function']['arguments'])} for c in m['tool_calls']]
        steps.append({'index': i, 'substeps': [sub]})
        index_map.append({'offset': i - 1, 'step_index': i, 'raw_position': i - 1})
    return {'trajectory_id': TID, 'domain': 'tau',
            'instruction': 'You are user u1 (a@b.c). Return the headphones from order #W1.',
            'instruction_source': 'info.task.instruction', 'steps': steps, 'index_map': index_map}


FULL = {'evidence': True, 'checklist': True, 'generation': 'stepwise', 'protocol': 'joint',
        'python_timeout': 3, 'name': 'agentrx'}
BASE = {'evidence': False, 'checklist': False, 'generation': 'stepwise', 'protocol': 'joint',
        'python_timeout': 3, 'name': 'baseline'}


def prediction(step, category, agent='Assistant', rationale='r'):
    return {'step_index': step, 'agent': agent, 'category': category, 'rationale': rationale,
            'evidence_refs': [{'step_index': 6, 'substep_index': 1}]}


def violation(step, target, name):
    return {'step_index': step, 'assertion_name': name, 'verdict': 'VIOL', 'taxonomy_targets': [target],
            'constraint_type': 'RELATIONAL_POST', 'check_type': 'nl_check', 'check_hint': '...',
            'evidence': [], 'rubric_results': [], 'unclear_count': 0, 'origin': 'dynamic'}


RUNS = [  # (variant, rep, config, status, prediction, violations)
    ('agentrx', 0, FULL, 'ok', prediction(7, 'MisinterpretationOfToolOutput',
                                          rationale='Step 6 lists Headphones (111), but step 7 says only a Watch.'),
     [violation(7, 'MisinterpretationOfToolOutput', 'items_grounded')]),
    ('agentrx', 1, FULL, 'partial_checker_errors', prediction(7, 'Misinterpretation of Tool Output / Handoff Failure',
                                                              agent='assistant', rationale='Misread order items.'),
     [violation(7, 'MisinterpretationOfToolOutput', 'items_grounded'),
      violation(9, 'Instruction/PlanAdherenceFailure', 'confirm_before_write')]),
    ('agentrx', 2, FULL, 'ok', prediction(9, 'Instruction/Plan Adherence Failure',
                                          rationale='Returned the wrong item.'),
     [violation(9, 'Instruction/PlanAdherenceFailure', 'confirm_before_write')]),
    ('agentrx', 3, FULL, 'error', None, []),
    ('baseline', 0, BASE, 'ok', prediction(3, 'InvalidInvocation'), []),
]


def make_fa_run():
    root = HERE / 'synthetic_run' / 'jobs'
    for variant, rep, config, status, pred, viols in RUNS:
        d = root / variant / f'rep-{rep}' / 'tau' / digest(TID)
        write(d / 'trajectory_ir.json', ir())
        write(d / 'validation_log.json', viols)
        write(d / 'result.json', {'trajectory_id': TID, 'domain': 'tau', 'measurement_kind': 'offline_fixture',
                                  'config': config, 'status': status, 'prediction': pred, 'checks': viols,
                                  'violations': viols, 'errors': [],
                                  'experiment_identity': 'synthetic-identity'})


def make_gold_and_raw():
    gold = {'trajectory_id': TID,
            'failures': [{'failure_id': '1', 'step_index': 7, 'category': 'MisinterpretationOfToolOutput', 'agent': 'Assistant'},
                         {'failure_id': '2', 'step_index': 9, 'category': 'Instruction/PlanAdherenceFailure', 'agent': 'Assistant'}],
            'critical': {'failure_id': '1', 'step_index': 7, 'category': 'MisinterpretationOfToolOutput', 'agent': 'Assistant'}}
    write(HERE / 'cases.json', [{'trajectory': ir(), 'gold': gold, 'policy': MESSAGES[0]['content'], 'tools': {}}])
    gold_actions = [{'name': 'return_delivered_order_items',
                     'kwargs': {'order_id': '#W1', 'item_ids': ['111'], 'note': 'GOLDSENTINEL-action'}}]
    write(HERE / 'tau_raw.json', [{
        'task_id': TID, 'reward': 0.0, 'trial': 0, 'records': {},
        'info': {'task': {'user_id': 'u1', 'actions': gold_actions,
                          'instruction': 'You are user u1 (a@b.c). Return the headphones from order #W1.',
                          'outputs': ['GOLDSENTINEL-output']},
                 'reward_info': {'reward': 0.0, 'info': {'note': 'GOLDSENTINEL-reward'}, 'actions': gold_actions}},
        'traj': [{**m, 'index': i} for i, m in enumerate(MESSAGES, start=1)]}])


if __name__ == '__main__':
    make_fa_run()
    make_gold_and_raw()
