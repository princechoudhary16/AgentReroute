"""Multi-agent retail workflow with fault hooks, rollback directives and trace export.

Agents
  planner        talks to the user, authenticates, routes work to specialists
  order_agent    pending-order changes (cancel, address, payment, items)
  product_agent  catalogue questions
  returns_agent  returns and exchanges of delivered orders

Every agent turn is one LLM call that returns a JSON action:
  {"thought": "...", "action": "tool",    "tool": "<name>", "arguments": {...}}
  {"thought": "...", "action": "handoff", "to": "<agent>", "message": "...", "context": {...}}   (planner only)
  {"thought": "...", "action": "reply",   "content": "..."}
A planner reply goes to the user; a specialist reply returns to the planner.

Specialists start a fresh conversation on every handoff, so whatever the
planner does not pass in the handoff is unavailable to them (this is what makes
context-propagation faults observable).

Fault operators (``benchmark.faults``) attach through three hooks:
``on_prompt`` (before an agent's LLM call), ``on_action`` (after its action is
parsed) and ``on_tool_result`` (after a tool runs). They never see gold data.
"""
import copy
import json
from dataclasses import dataclass, field

from ..io import digest
from .llm import parse_json, request
from .retail import TOOL_SCHEMAS, WRITE_TOOLS, RetailDB

STOP = '###STOP###'
# Bump when agent roles, tools or prompt text change: it is part of a generation's settings lock.
WORKFLOW_VERSION = 2

SPECIALIST_NOTE = ('Division of work: the planner has already authenticated the user and obtained the user\'s '
                   'explicit confirmation for the request in the handoff, so the policy steps about '
                   'authenticating and confirming with the user are already satisfied. Do not ask for them again; '
                   'you cannot talk to the user. If details are missing or ambiguous, or the request belongs to '
                   'another specialist, reply to the planner saying exactly what is missing or who should handle it.')

AGENTS = {
    'planner': {
        'role': 'You are the planner of a retail customer-service team. You talk to the user, authenticate '
                'them, look up orders and products to work out exactly what they want, obtain explicit '
                'confirmation before any database change, and delegate the change to a specialist: changes to '
                'PENDING orders and user addresses go to order_agent, returns and exchanges of DELIVERED orders go '
                'to returns_agent, catalogue questions go to product_agent. Pass every detail the specialist needs '
                '(user id, order id, item ids, new item ids, payment method, the confirmed request) in the '
                'handoff, because specialists cannot see this conversation.',
        'tools': ['find_user_id_by_email', 'find_user_id_by_name_zip', 'get_user_details', 'get_order_details',
                  'get_product_details', 'list_all_product_types', 'transfer_to_human_agents'],
        'handoff': ['order_agent', 'product_agent', 'returns_agent'],
    },
    'order_agent': {
        'role': 'You are the order specialist. You handle PENDING orders: cancel, change address, change '
                'payment method, modify items, and user address changes. The user has already confirmed the '
                'request given to you. Do the work with your tools, then reply with the result.',
        'tools': ['get_order_details', 'get_product_details', 'get_user_details', 'cancel_pending_order',
                  'modify_pending_order_address', 'modify_pending_order_payment', 'modify_pending_order_items',
                  'modify_user_address'],
    },
    'product_agent': {
        'role': 'You are the product specialist. You answer questions about products, variants, availability '
                'and prices using your tools, then reply with the answer.',
        'tools': ['list_all_product_types', 'get_product_details', 'calculate'],
    },
    'returns_agent': {
        'role': 'You are the returns specialist. You handle DELIVERED orders: returns and exchanges. The user has '
                'already confirmed the request given to you. Do the work with your tools, then reply with the '
                'result, including any refund amount.',
        'tools': ['get_order_details', 'get_product_details', 'get_user_details', 'return_delivered_order_items',
                  'exchange_delivered_order_items', 'calculate'],
    },
}

GREETING = 'Hi! How can I help you today?'


def system_prompt(agent, policy):
    spec = AGENTS[agent]
    tools = [TOOL_SCHEMAS[t] for t in spec['tools']]
    actions = ['{"thought": "...", "action": "tool", "tool": "<tool name>", "arguments": {...}}']
    if 'handoff' in spec:
        actions.append('{"thought": "...", "action": "handoff", "to": "<agent>", "message": "<request>", '
                       '"context": {"user_id": "...", "order_id": "...", ...}}')
        reply = 'reply to the user'
    else:
        reply = 'reply to the planner with the result'
    actions.append('{"thought": "...", "action": "reply", "content": "<' + reply + '>"}')
    role = spec['role'] if 'handoff' in spec else spec['role'] + '\n\n' + SPECIALIST_NOTE
    parts = [role, '', '# Policy', policy.strip(), '', '# Tools', json.dumps(tools, indent=1)]
    if 'handoff' in spec:
        parts += ['', '# Specialists', 'You can hand off to: ' + ', '.join(spec['handoff']) + '.']
    parts += ['', '# Output format', 'Respond with exactly one JSON object, one of:', *actions,
              'Use only the tools listed above. One action per turn.']
    return '\n'.join(parts)


@dataclass
class Directive:
    """A recovery intervention applied when a replay reaches ``rollback_step``.

    kind: ``guidance`` appends ``text`` as a supervisor note to the agent's context;
          ``resample`` re-draws that turn and every later one with a new seed;
          ``none`` replays unchanged (control).
    """
    rollback_step: int
    kind: str = 'guidance'
    text: str = ''
    seed_offset: int = 1

    def __post_init__(self):
        if self.kind not in ('guidance', 'resample', 'none'):
            raise ValueError(f'unknown directive kind {self.kind}')


@dataclass
class Step:
    index: int
    actor: str
    kind: str                      # user | agent | tool | system
    content: str
    action: dict | None = None
    tool: dict | None = None       # {name, arguments} for agent tool calls; {name} for tool results
    handoff: dict | None = None
    input: list | None = None      # messages the agent saw (full view only)
    input_digest: str | None = None
    state_version: int = 0
    tokens: int = 0
    replayed: bool = False         # produced from the replayed prefix (no new cost)
    faults: list = field(default_factory=list)   # gold-side injection records


@dataclass
class Run:
    task_id: str
    model: str
    seed: int
    steps: list
    final_state_hash: str
    user_replies: list
    termination: str
    tokens: int
    new_tokens: int
    faults_fired: list
    write_calls: list
    lenient_responses: int = 0     # completions where the actor's lenient fallback fired this run

    def to_dict(self):
        out = {k: v for k, v in self.__dict__.items() if k != 'steps'}
        out['steps'] = [s.__dict__ for s in self.steps]
        return out


class UserSimulator:
    """tau-bench-style simulated user driven by an LLM (plain text replies)."""

    def __init__(self, llm, model, instruction, seed=0, temperature=0.0):
        self.llm, self.model, self.seed, self.temperature = llm, model, seed, temperature
        self.instruction = instruction
        self.messages = [{'role': 'system', 'content': (
            'You are a user talking to a customer-service agent.\n\nInstruction: ' + instruction + '\n\n'
            'Rules:\n- Write one short message at a time, as the user.\n- Do not reveal the whole instruction at '
            'once; share details when they are needed.\n- Do not invent information not in the instruction.\n'
            '- Confirm (say yes) when the agent asks to confirm an action that matches your instruction.\n'
            f'- When your goal is fully achieved, or cannot be achieved, reply with {STOP} only.')}]

    def respond(self, agent_text, step_index):
        self.messages.append({'role': 'user', 'content': agent_text})
        completion = self.llm.complete(request(self.model, self.messages, self.temperature, self.seed,
                                               json_mode=False, agent='user', step_index=step_index))
        text = completion.text.strip()
        self.messages.append({'role': 'assistant', 'content': text})
        return text, completion


class Workflow:
    def __init__(self, domain, task, llm, model, seed=0, temperature=0.0, faults=(), directive=None,
                 user_llm=None, max_steps=60, max_specialist_turns=10, max_invalid=3, max_no_progress=3):
        self.domain, self.task, self.llm, self.model = domain, task, llm, model
        self.seed, self.temperature = seed, temperature
        self.faults = list(faults)
        self.directive = directive
        self.user = UserSimulator(user_llm or llm, model, task['instruction'], seed, temperature)
        self.max_steps, self.max_specialist_turns, self.max_invalid = max_steps, max_specialist_turns, max_invalid
        self.max_no_progress = max_no_progress
        self.db = RetailDB(domain.data)
        self.steps, self.user_replies, self.write_calls = [], [], []
        self.lenient_responses = 0
        self.state_version = 0
        self.resampling = False
        self._applied = False
        self.guided = {}            # agent -> step at which guidance entered its context
        self.prompts = {a: system_prompt(a, domain.policy) for a in AGENTS}

    # ------------------------------------------------------------------ bookkeeping
    def _add(self, **kwargs):
        index = len(self.steps) + 1
        step = Step(index=index, state_version=self.state_version, **kwargs)
        step.replayed = self.directive is not None and index < self.directive.rollback_step
        self.steps.append(step)
        return step

    def _budget_left(self):
        return len(self.steps) < self.max_steps

    def _seed(self):
        return self.seed + (self.directive.seed_offset * 1000 if self.resampling else 0)

    # ------------------------------------------------------------------ agent turn
    def _agent_turn(self, agent, history):
        """One LLM call for ``agent``; returns (action dict or None, step)."""
        index = len(self.steps) + 1
        directive = self.directive
        if directive and directive.kind != 'none' and index >= directive.rollback_step and not self._applied:
            self._applied = True
            if directive.kind == 'guidance':
                history.append({'role': 'user', 'content': 'SUPERVISOR NOTE: ' + directive.text})
                self.guided[agent] = index
            elif directive.kind == 'resample':
                self.resampling = True
        messages = [{'role': 'system', 'content': self.prompts[agent]}] + history
        ctx = {'agent': agent, 'step_index': index, 'input_digest': digest(messages), 'guided': dict(self.guided)}
        records = []
        for op in self.faults:
            messages, record = op.on_prompt(ctx, messages)
            if record:
                records.append(record)
        if records:  # the digest identifies the context the agent actually saw
            ctx['input_digest'] = digest(messages)
        completion = self.llm.complete(request(self.model, messages, self.temperature, self._seed(),
                                               agent=agent, step_index=index, task_id=self.task['task_id']))
        self.lenient_responses += int(completion.lenient)
        try:
            action = parse_json(completion.text)
            action = _validate_action(agent, action)
            error = None
        except ValueError as exc:
            action, error = None, str(exc)
        if action is not None:
            ctx['messages'] = messages
            for op in self.faults:
                action, record = op.on_action(ctx, action)
                if record:
                    records.append(record)
        content = json.dumps(action, ensure_ascii=False) if action is not None else completion.text
        step = self._add(actor=agent, kind='agent', content=content, action=action, input=messages,
                         input_digest=ctx['input_digest'], tokens=completion.tokens, faults=records)
        history.append({'role': 'assistant', 'content': completion.text if not records else content})
        if error:
            history.append({'role': 'user', 'content': f'SYSTEM: invalid action ({error}). Reply with one JSON '
                                                       'object in the required format.'})
        return action, step

    def _tool(self, agent, action, history):
        name, args = action['tool'], action.get('arguments', {})
        allowed = AGENTS[agent]['tools']
        if name not in allowed:
            result = f'Error: tool {name} is not available to {agent}'
        else:
            before = self.db.state_hash() if name in WRITE_TOOLS else None
            result = self.db.call(name, args)
            if before is not None and self.db.state_hash() != before:
                self.state_version += 1
            if name in WRITE_TOOLS:
                self.write_calls.append({'agent': agent, 'tool': name, 'arguments': args,
                                         'step_index': len(self.steps), 'ok': not result.startswith('Error')})
        records = []
        ctx = {'agent': agent, 'step_index': len(self.steps) + 1, 'tool': name, 'arguments': args}
        for op in self.faults:
            result, record = op.on_tool_result(ctx, name, args, result)
            if record:
                records.append(record)
        self._add(actor='tool', kind='tool', content=result, tool={'name': name, 'called_by': agent},
                  faults=records)
        history.append({'role': 'user', 'content': f'TOOL_RESULT {name}: {result}'})
        return result

    # ------------------------------------------------------------------ specialist conversation
    def _specialist(self, agent, handoff):
        try:
            return self._specialist_turns(agent, handoff)
        finally:
            self.guided.pop(agent, None)   # its conversation (and any guidance in it) ends here

    def _specialist_turns(self, agent, handoff):
        history = [{'role': 'user', 'content': 'HANDOFF from planner: ' + handoff.get('message', '') +
                    '\nCONTEXT: ' + json.dumps(handoff.get('context', {}), ensure_ascii=False)}]
        invalid = 0
        for _ in range(self.max_specialist_turns):
            if not self._budget_left():
                return 'SYSTEM: step budget exhausted'
            action, _ = self._agent_turn(agent, history)
            if action is None:
                invalid += 1
                if invalid >= self.max_invalid:
                    return f'SYSTEM: {agent} produced no valid action'
                continue
            invalid = 0
            if action['action'] == 'tool':
                self._tool(agent, action, history)
            elif action['action'] == 'reply':
                return action['content']
            else:
                history.append({'role': 'user', 'content': 'SYSTEM: specialists cannot hand off; reply instead.'})
        return f'SYSTEM: {agent} reached its turn limit without replying'

    # ------------------------------------------------------------------ main loop
    def run(self):
        planner_history = []
        termination = 'max_steps'
        agent_text = GREETING
        invalid = 0
        last_fp = last_state = None
        repeats = 0
        user_text, completion = self.user.respond(agent_text, 1)
        self.lenient_responses += int(completion.lenient)
        self._add(actor='user', kind='user', content=user_text, tokens=completion.tokens)
        planner_history.append({'role': 'user', 'content': 'USER: ' + user_text})
        if STOP in user_text:
            termination = 'user_stop'
        while termination == 'max_steps' and self._budget_left():
            action, step = self._agent_turn('planner', planner_history)
            if action is None:
                invalid += 1
                if invalid >= self.max_invalid:
                    termination = 'invalid_actions'
                    break
                continue
            invalid = 0
            kind = action['action']
            if kind in ('tool', 'handoff'):
                # No-progress guard: the planner emitting the same tool/handoff with the same args while the
                # DB is unchanged is a deterministic stuck loop (temperature 0 reproduces it forever). Replies
                # to the user between repeats are not progress, so they neither count nor reset. Specialist
                # loops stay bounded by max_specialist_turns (the cycles fault relies on them).
                fp = ((action.get('tool'), json.dumps(action.get('arguments', {}), sort_keys=True)) if kind == 'tool'
                      else (action['to'], action.get('message', ''), json.dumps(action.get('context', {}), sort_keys=True)))
                state = self.db.state_hash()
                repeats = repeats + 1 if (fp == last_fp and state == last_state) else 1
                last_fp, last_state = fp, state
            if kind == 'tool':
                result = self._tool('planner', action, planner_history)
                if action['tool'] == 'transfer_to_human_agents' and not result.startswith('Error'):
                    termination = 'transfer'
            elif kind == 'handoff':
                target = action['to']
                step.handoff = {'to': target, 'message': action.get('message', ''),
                                'context': action.get('context', {})}
                if target not in AGENTS['planner']['handoff']:
                    result = f'SYSTEM: unknown agent {target}'
                else:
                    result = self._specialist(target, step.handoff)
                planner_history.append({'role': 'user', 'content': f'AGENT_RESULT from {target}: {result}'})
            else:
                self.user_replies.append(action['content'])
                if not self._budget_left():
                    break
                user_text, completion = self.user.respond(action['content'], len(self.steps) + 1)
                self.lenient_responses += int(completion.lenient)
                self._add(actor='user', kind='user', content=user_text, tokens=completion.tokens)
                if STOP in user_text:
                    termination = 'user_stop'
                    break
                planner_history.append({'role': 'user', 'content': 'USER: ' + user_text})
            if termination == 'max_steps' and repeats >= self.max_no_progress:
                termination = 'no_progress'
        rollback = self.directive.rollback_step if self.directive else 1
        return Run(self.task['task_id'], self.model, self.seed, self.steps, self.db.state_hash(),
                   self.user_replies, termination, sum(s.tokens for s in self.steps),
                   sum(s.tokens for s in self.steps if s.index >= rollback),
                   [r for s in self.steps for r in s.faults], self.write_calls,
                   lenient_responses=self.lenient_responses)


def _validate_action(agent, action):
    kind = action.get('action')
    if kind == 'tool':
        if not isinstance(action.get('tool'), str):
            raise ValueError('tool action needs a tool name')
        if not isinstance(action.get('arguments', {}), dict):
            raise ValueError('arguments must be an object')
        action.setdefault('arguments', {})
    elif kind == 'handoff':
        if 'handoff' not in AGENTS[agent]:
            raise ValueError(f'{agent} cannot hand off')
        if not isinstance(action.get('to'), str):
            raise ValueError('handoff needs a target agent')
        if not isinstance(action.get('context', {}), dict):
            action['context'] = {'note': str(action.get('context'))}
        action.setdefault('message', '')
        action.setdefault('context', {})
    elif kind == 'reply':
        if not isinstance(action.get('content'), str):
            action['content'] = json.dumps(action.get('content'), ensure_ascii=False)
    else:
        raise ValueError(f'unknown action {kind!r}')
    return action


# ---------------------------------------------------------------------- outcome (gold side)
def outcome(task, run, gold_hash):
    """tau-bench reward: final database equals the gold-action database, and every
    required output string appears in a reply to the user."""
    state_ok = run.final_state_hash == gold_hash
    text = ' '.join(run.user_replies).lower().replace(',', '')
    missing = [o for o in task.get('outputs', []) if o.lower() not in text]
    return {'reward': float(state_ok and not missing), 'state_ok': state_ok, 'missing_outputs': missing}


# ---------------------------------------------------------------------- trace export
def export_trace(run, trajectory_id, domain_name, instruction, view='full'):
    """Observable trace in the faultattribution IR (steps -> substeps), without gold data."""
    if view not in ('full', 'output_only'):
        raise ValueError(view)
    steps = []
    for s in run.steps:
        event = {'sub_index': 1, 'role': s.actor, 'content': s.content}
        if s.kind == 'tool':
            event['name'] = s.tool['name']
        if s.action and s.action.get('action') == 'tool':
            event['tool_calls'] = [{'id': f'call-{s.index}', 'name': s.action['tool'],
                                    'arguments': s.action.get('arguments', {})}]
        step = {'index': s.index, 'substeps': [event]}
        if s.handoff:
            step['substeps'].append({'sub_index': 2, 'role': s.actor, 'content': json.dumps(
                {'handoff_to': s.handoff['to'], 'message': s.handoff['message'], 'context': s.handoff['context']},
                ensure_ascii=False), 'name': 'handoff'})
        if view == 'full':
            step['state_version'] = s.state_version
            if s.input is not None:
                step['input'] = copy.deepcopy(s.input[1:])          # system prompt is shared, see agents
                step['input_digest'] = s.input_digest
        steps.append(step)
    trace = {'trajectory_id': trajectory_id, 'domain': domain_name, 'instruction': instruction,
             'instruction_source': 'first_user_message', 'view': view, 'steps': steps,
             'agents': sorted(AGENTS)}
    if view == 'full':
        trace['system_prompts'] = {a: system_prompt(a, '<policy>') for a in AGENTS}
    return trace
