"""Phase 2 multi-perspective tracing: [B1] split, [B2] seeded+synthesized checks, [B3] blame, [B4] features."""
import builtins
import copy
import json
from collections import Counter
from functools import lru_cache
from .constraints import _python, check_constraint, constraint_fingerprint, guard_matches, prefix, validate_constraint
from .io import digest
from .taxonomy import CATEGORIES, agent, taxonomy
from .worker import SAFE_BUILTINS, check_source, safe_import

PERSPECTIVES = {
    'P1_INTENT_PLAN': {
        'name': 'P1 Intent & Plan Lens',
        'categories': ('Instruction/PlanAdherenceFailure', 'IntentPlanMisalignment',
                       'UnderspecifiedUserIntent', 'IntentNotSupported'),
        'focus': 'User intent alignment, domain policy adherence, plan completeness, and abstention when intent is unsupported or underspecified.'},
    'P2_TOOL_EXECUTION': {
        'name': 'P2 Tool & Execution Lens',
        'categories': ('InvalidInvocation', 'GuardrailsTriggered', 'SystemFailure'),
        'focus': 'Tool call schema validity, required parameter presence, argument types, guardrail triggers, and runtime/system execution errors.'},
    'P3_HANDOFF_GROUNDING': {
        'name': 'P3 Handoff & Grounding Lens',
        'categories': ('InventionOfNewInformation', 'MisinterpretationOfToolOutput'),
        'focus': 'Factual grounding in prior tool/agent outputs, absence of fabricated IDs/numbers/claims, and accurate cross-agent handoffs.'},
}
ORDER = tuple(PERSPECTIVES)
FEATURES = 10
PASSIVE = ('user', 'system', 'tool')  # canonical roles that never own a fault candidate's action
NON_AGENTS = PASSIVE + ('unknown',)  # never blamed, never repaired (e.g. a static system policy prompt)
PLACEHOLDERS = {'assertion_name': 'unique_snake_case', 'judge_rubric': 'objective criterion'}  # CONSTRAINT_SCHEMA literals
DRY_RUN_HINT = ("Use exact IR keys: step['substeps'][i]['tool_calls'][j]['name'] and ['arguments'], never "
                "step['tool_calls'], call['tool_name'], call['args'], or trajectory['tools'].")
EPS = 1e-6
ANY_EVENT = {'step_index': '*', 'substep_index': '*', 'role_name': '*', 'tool_name': '*', 'content_regex': '*'}
ERROR_MARKERS = r'error:|exception|\binvalid\b'
# ponytail: keyword/ID heuristics, negations ("not successful") can false-positive; LLM constraints cover nuance.
SUCCESS_CLAIM = r'\b(success(ful(ly)?)?|confirmed|has been (processed|completed|cancell?ed|updated|placed|modified|exchanged|returned|refunded))\b'
ID_TOKEN = r'#?\b[A-Z0-9]*[0-9][A-Z0-9]{4,}\b'
WRITE_CLAIM = (  # every completion claim in benchmark/faults/operators.py:CLAIMS
    r'\b(?:'
    r'has been (?:cancell?ed|modified|exchanged|returned|refunded|requested|submitted|updated)'
    r'|return submitted for order'
    r'|exchange requested for order'
    r'|i (?:cancell?ed|modified|updated) (?:order|the items|the user\'s default address)'
    r'|shipping address of order \S+ is now'
    r'|will ship to the new address'
    r'|is now paid with'
    r'|payment method for order \S+ updated'
    r'|default address for \S+ is now'
    r')\b')
WRITE_TOOLS = ('cancel_', 'modify_', 'exchange_', 'return_')
ROUTING_DRIFT = r'cannot handle|cannot proceed|wrong specialist'
STUCK_TOOL = r'"status"\s*:\s*"processing"'
HANDOFF_ID = r'#W\d+|\buser_\w+'
# Misroute rules per handoff target (retail MAS specialists): work the target cannot own.
PRODUCT_MISROUTE = r'#w\d+|cancel|return|exchange|modify|address|payment|refund'
ORDER_ID_FIELD = r'"order_id"\s*:'
RETURNS_WORK = r'\b(return|exchange|refund)\b'
ORDER_WORK = r'\b(cancel|modify|address|payment)\b'
PRODUCT_QUESTION = r'\bvariants?\b|\bavailab|\bproduct types?\b'
MONEY_FIELD = r'"[a-z_]*(total|price|refund|amount)[a-z_]*"\s*:\s*-?\d'
NUMBER = r'\d+(?:\.\d+)?'
DOLLARS = r'\$\s?(\d+(?:\.\d+)?)'  # ponytail: "$1,234" reads as 1; add comma grouping if amounts reach thousands


def taxonomy_subset(perspective_id):
    categories = PERSPECTIVES[perspective_id]['categories']
    return [t for t in taxonomy(False) if t['category'] in categories]


def tool_names(tools):
    spec = tools.get('tools', tools) if isinstance(tools, dict) else tools
    if isinstance(spec, dict):
        return sorted(k for k, v in spec.items() if isinstance(v, dict))
    if isinstance(spec, list):
        names = [(t.get('function') or t).get('name') for t in spec if isinstance(t, dict)]
        return sorted(n for n in names if isinstance(n, str))
    return []


def _seed(name, targets, kind, hint, body, content_regex='*'):
    c = validate_constraint({
        'assertion_name': name, 'taxonomy_targets': list(targets), 'constraint_type': kind,
        'event_trigger': {**ANY_EVENT, 'content_regex': content_regex}, 'check_hint': hint, 'check_type': 'python_check',
        'python_check': {'function_name': name, 'args': ['trajectory', 'current_step_index'],
                         'code_lines': [f'def {name}(trajectory, current_step_index):'] + body}})
    c['origin'] = 'seed'
    return c


def seed_constraints_for_perspective(perspective_id, tools):
    """D-10: one wildcard-guarded floor seed per perspective, so it never collapses to all-SKIP, plus guarded
    specialist seeds that judge only the steps their rule applies to (vacuous SATs would dilute V/(V+S))."""
    if perspective_id == 'P1_INTENT_PLAN':
        return [_seed('p1_seed_nonempty_progress', ['Instruction/PlanAdherenceFailure'], 'PROTOCOL',
                      'Non-tool agent messages must be non-empty, must not repeat the same role\'s previous message '
                      'verbatim, and must not claim a write that no earlier write tool call performed.', [
            "    import re",
            "    passive = ('user', 'human', 'system', 'tool')",
            "    steps = trajectory['steps']",
            "    for event in steps[current_step_index]['substeps']:",
            "        role = str(event.get('role', ''))",
            "        if role.lower() in passive or event.get('tool_calls'):",
            "            continue",
            "        content = str(event.get('content', '')).strip()",
            "        if not content:",
            "            return False",
            "        handoff_here = any(e.get('name') == 'handoff' for e in steps[current_step_index]['substeps'])",
            f"        if not handoff_here and re.search({WRITE_CLAIM!r}, content, re.I):",
            "            # Only writes since the latest handoff excuse a specialist's claim; a planner relaying a",
            "            # specialist's result is excused by any write in the episode.",
            "            handoffs = [o for o, s in enumerate(steps[:current_step_index])",
            "                        if any(e.get('name') == 'handoff' for e in s['substeps'])]",
            "            since = 0 if role.lower() == 'planner' or not handoffs else handoffs[-1]",
            f"            if not any(str(call.get('name', '')).startswith({WRITE_TOOLS!r})",
            "                       for s in steps[since:current_step_index] for e in s['substeps']",
            "                       for call in e.get('tool_calls') or [] if isinstance(call, dict)):",
            "                return False",
            "        for earlier in reversed(steps[:current_step_index]):",
            "            same = [e for e in earlier['substeps'] if e.get('role') == role]",
            "            if same:",
            "                if str(same[-1].get('content', '')).strip() == content:",
            "                    return False",
            "                break",
            "    return True"]),
                _seed('p1_seed_specialist_routing_alignment', ['IntentPlanMisalignment'], 'PROTOCOL',
                      'Handoffs must go to a specialist that owns the work; agents must not drift into refusal/'
                      'wrong-specialist replies or repeat an identical tool call a 3rd time.', [
            "    import json",
            "    import re",
            "    steps = trajectory['steps']",
            "    earlier = ' '.join(str(e.get('content', '')) for s in steps[:current_step_index] for e in s['substeps'])",
            "    seen = [(e.get('role'), call.get('name'), json.dumps(call.get('arguments'), sort_keys=True))",
            "            for s in steps[:current_step_index + 1] for e in s['substeps']",
            "            for call in e.get('tool_calls') or [] if isinstance(call, dict)]",
            "    for event in steps[current_step_index]['substeps']:",
            "        if str(event.get('role', '')).lower() in ('user', 'human', 'system', 'tool'):",
            "            continue",
            "        if event.get('name') == 'handoff':",
            "            try:",
            "                data = json.loads(str(event.get('content', '')))",
            "            except ValueError:",
            "                continue",
            "            if not isinstance(data, dict):",
            "                continue",
            "            target = data.get('handoff_to')",
            "            msg = (str(data.get('message', '')) + ' ' + json.dumps(data.get('context', {}))).lower()",
            f"            if target == 'product_agent' and (re.search({PRODUCT_MISROUTE!r}, msg)",
            f"                                              or re.search({ORDER_ID_FIELD!r}, earlier)):",
            "                return False",
            f"            if target == 'order_agent' and 'delivered' in earlier.lower() and re.search({RETURNS_WORK!r}, msg):",
            "                return False",
            f"            if target == 'returns_agent' and 'pending' in earlier.lower() and re.search({ORDER_WORK!r}, msg):",
            "                return False",
            f"            if target in ('order_agent', 'returns_agent') and re.search({PRODUCT_QUESTION!r}, msg) \\",
            f"                    and not re.search({PRODUCT_MISROUTE!r}, msg):",
            "                return False",
            "            continue",
            "        # A specialist's refusal is the drift signal; the planner relaying it is an echo of that symptom.",
            f"        if str(event.get('role', '')).lower() != 'planner' and re.search({ROUTING_DRIFT!r},",
            "                                                                     str(event.get('content', '')), re.I):",
            "            return False",
            "        for call in event.get('tool_calls') or []:",
            "            if isinstance(call, dict) and seen.count((event.get('role'), call.get('name'),",
            "                                                      json.dumps(call.get('arguments'), sort_keys=True))) >= 3:",
            "                return False",
            "    return True"], content_regex=f'(?i){ROUTING_DRIFT}|"arguments"|"handoff_to"')]
    if perspective_id == 'P2_TOOL_EXECUTION':
        return [_seed('p2_seed_valid_tool_invocation', ['InvalidInvocation', 'SystemFailure'], 'SCHEMA',
                      'Tool calls need object arguments and known tool names; tool outputs must not report errors '
                      'or a stuck "processing" status.', [
            "    import re",
            f"    known = {tool_names(tools)!r}",
            "    for event in trajectory['steps'][current_step_index]['substeps']:",
            "        for call in event.get('tool_calls', []):",
            "            if not isinstance(call, dict) or not isinstance(call.get('arguments'), dict):",
            "                return False",
            "            if known and call.get('name') not in known:",
            "                return False",
            "        text = str(event.get('content', ''))",
            f"        if str(event.get('role', '')).lower() == 'tool' and (re.search({ERROR_MARKERS!r}, text, re.I)",
            f"                or re.search({STUCK_TOOL!r}, text) or text.lstrip().startswith('Error')):",
            "            return False",
            "    return True"])]
    if perspective_id == 'P3_HANDOFF_GROUNDING':
        return [_seed('p3_seed_no_contradicted_tool_output', ['MisinterpretationOfToolOutput', 'InventionOfNewInformation'],
                      'PROVENANCE', 'After an empty/error tool output, the next agent message must not claim success or cite unseen IDs.', [
            "    import re",
            "    if current_step_index == 0:",
            "        return True",
            "    steps = trajectory['steps']",
            "    failed = False",
            "    for event in steps[current_step_index - 1]['substeps']:",
            "        text = str(event.get('content', ''))",
            f"        if str(event.get('role', '')).lower() == 'tool' and (not text.strip() or re.search({ERROR_MARKERS!r}, text, re.I)):",
            "            failed = True",
            "    if not failed:",
            "        return True",
            "    seen = ' '.join(str(e.get('content', '')) for s in steps[:current_step_index] for e in s['substeps'])",
            "    for event in steps[current_step_index]['substeps']:",
            "        if str(event.get('role', '')).lower() in ('user', 'human', 'system', 'tool') or event.get('tool_calls'):",
            "            continue",
            "        text = str(event.get('content', ''))",
            f"        if re.search({SUCCESS_CLAIM!r}, text, re.I):",
            "            return False",
            f"        for found in re.findall({ID_TOKEN!r}, text):",
            "            if found.lstrip('#') not in seen:",
            "                return False",
            "    return True"]),
                _seed('p3_seed_handoff_and_fact_grounding', ['MisinterpretationOfToolOutput'], 'RELATIONAL_POST',
                      'Handoffs must carry the order/user ids seen so far; dollar amounts after a priced tool output '
                      'must appear in the observed prefix.', [
            "    import json",
            "    import re",
            "    steps = trajectory['steps']",
            "    earlier = ' '.join(str(e.get('content', '')) for s in steps[:current_step_index] for e in s['substeps'])",
            "    for event in steps[current_step_index]['substeps']:",
            "        if event.get('name') != 'handoff':",
            "            continue",
            "        try:",
            "            data = json.loads(str(event.get('content', '')))",
            "        except ValueError:",
            "            continue",
            "        if not isinstance(data, dict):",
            "            continue",
            "        context = data.get('context')",
            "        if not context or '[omitted]' in str(data.get('message', '')):",
            "            return False",
            f"        if isinstance(context, dict) and re.search({HANDOFF_ID!r}, earlier) and not any(",
            "                context.get(k) for k in ('order_id', 'user_id', 'item_ids')):",
            "            return False",
            "    if current_step_index == 0:",
            "        return True",
            "    tool_text = ' '.join(str(e.get('content', '')) for e in steps[current_step_index - 1]['substeps']",
            "                         if str(e.get('role', '')).lower() == 'tool')",
            f"    if not re.search({MONEY_FIELD!r}, tool_text):",
            "        return True",
            f"    known = {{float(n) for n in re.findall({NUMBER!r}, earlier)}}",
            "    for event in steps[current_step_index]['substeps']:",
            "        if str(event.get('role', '')).lower() in ('user', 'human', 'system', 'tool'):",
            "            continue",
            f"        for amount in re.findall({DOLLARS!r}, str(event.get('content', ''))):",
            "            if float(amount) not in known:",
            "                return False",
            "    return True"], content_regex=r'"context"\s*:|"handoff_to"|\$\s?\d')]
    raise ValueError(f'Unknown perspective {perspective_id}')


def _reject_reason(raw, memory, trajectory=None, timeout=2):
    """Layer 1: (constraint, None) if usable, else (None, compiler-style reason fed back on retry)."""
    if memory is not None and digest(raw) in memory.forbidden:
        return None, 'Previously rejected output (forbidden fingerprint); emit a different constraint'
    try:
        c = validate_constraint(raw)
        if c['check_type'] == 'python_check':
            check_source('\n'.join(c['python_check']['code_lines']))
    except (ValueError, SyntaxError) as exc:
        return None, f'{type(exc).__name__}: {exc}'
    if c['assertion_name'] == PLACEHOLDERS['assertion_name']:
        return None, 'assertion_name copies the schema placeholder "unique_snake_case"; name the actual obligation'
    if c['check_type'] == 'nl_check' and any(str(item).strip().lower() == PLACEHOLDERS['judge_rubric']
                                             for item in c['nl_check']['judge_rubric']):
        return None, 'judge_rubric copies the schema placeholder "objective criterion"; write concrete criteria'
    if c['check_type'] == 'python_check' and trajectory and trajectory.get('steps'):
        # Dry-run on the first step and the first tool-call step: catches IR key errors before 60 runtime ERRORs.
        offsets = [0] + [next((o for o, s in enumerate(trajectory['steps'])
                               if any(e.get('tool_calls') for e in s['substeps'])), 0)]
        for offset in dict.fromkeys(offsets):
            checked = _python(c, prefix(trajectory, offset), offset, timeout)
            if 'error' in checked:
                return None, f"Python check failed runtime dry-run on trajectory: {checked['error']}. {DRY_RUN_HINT}"
    return c, None


def synthesize_perspective_constraints(perspective_id, trajectory, policy, tools, provider, memory=None, max_retries=2):
    """[A1_Pi]+[A2_Pi]: perspective-scoped LLM constraints (Layer 1 validated, anti-repeat retried) plus seeds.

    Returns (constraints, synthesis_log). Provider failures leave the deterministic seeds in place.
    """
    spec = PERSPECTIVES[perspective_id]
    base = {'perspective': perspective_id, 'perspective_name': spec['name'], 'perspective_focus': spec['focus'],
            'taxonomy_subset': taxonomy_subset(perspective_id), 'domain': trajectory['domain'],
            'policy': policy, 'tools': tools, 'observed_agent_roles': sorted({
                raw for k, raw, _ in (step_actor(trajectory, o) for o in range(len(trajectory['steps'])))
                if k not in NON_AGENTS})}
    if memory is not None:
        base['memory_exemplars'] = memory.retrieve_exemplars(perspective_id)
        base['forbidden_patterns'] = memory.retrieve_forbidden(perspective_id)
    accepted, fingerprints, log = [], set(), []
    for stage in ('perspective_global', 'perspective_dynamic'):
        payload = dict(base)
        if stage == 'perspective_dynamic':
            # ponytail: one oneshot call per perspective, not 3*T stepwise calls; stepwise if recall demands it.
            payload.update(trajectory=trajectory, global_constraints=list(accepted), generation='oneshot')
        for attempt in range(max_retries + 1):
            entry = {'stage': stage, 'attempt': attempt, 'accepted': [], 'rejected': []}
            log.append(entry)
            try:
                answer = provider.complete(stage, payload)
            except Exception as exc:
                entry['error'] = f'{type(exc).__name__}: {exc}'
                break
            items = answer.get('constraints') if isinstance(answer, dict) else None
            if not isinstance(items, list):
                items, entry['rejected'] = [], [{'reason': 'Synthesis must return {"constraints": [...]}', 'output': answer}]
            for raw in items:
                c, reason = _reject_reason(raw, memory, trajectory)
                if reason:
                    entry['rejected'].append({'reason': reason, 'output': raw, 'fingerprint': digest(raw)})
                    if memory is not None:
                        memory.record_forbidden(perspective_id, raw, reason)
                    continue
                key = digest(c)
                if key in fingerprints:
                    continue
                fingerprints.add(key)
                c['taxonomy_targets'] = [t for t in c['taxonomy_targets'] if t in spec['categories']] or [spec['categories'][0]]
                if c['assertion_name'] in {a['assertion_name'] for a in accepted}:
                    c['assertion_name'] += f'_{len(accepted)}'
                c['origin'] = 'llm'
                accepted.append(c)
                entry['accepted'].append(c['assertion_name'])
            if not entry['rejected']:
                break
            # Changing the payload also changes the CachedProvider request hash, so the retry is a fresh call.
            payload = {**payload, 'retry_feedback': [
                {'reason': r['reason'], 'note': 'FORBIDDEN OUTPUT (DO NOT REPEAT): ' + json.dumps(r['output'], ensure_ascii=False)}
                for r in entry['rejected']]}
    return accepted + seed_constraints_for_perspective(perspective_id, tools), log


@lru_cache(maxsize=None)
def _seed_function(code, function_name):
    env = {'__builtins__': {**{n: getattr(builtins, n) for n in SAFE_BUILTINS}, '__import__': safe_import}}
    exec(compile(check_source(code), '<seed-check>', 'exec'), env)
    return env[function_name]


def _check_seed_inprocess(raw, trajectory, offset):
    """check_constraint() for built-in seeds without the per-call worker subprocess: same row schema and verdicts.

    ponytail: no timeout or rlimits, so only seeds authored in this file may run here; LLM code stays sandboxed.
    """
    step = trajectory['steps'][offset]
    c = validate_constraint(raw)
    result = {'step_index': step['index'], 'assertion_name': c['assertion_name'], 'verdict': 'ERROR',
              'rule_fingerprint': constraint_fingerprint(c), 'taxonomy_targets': c['taxonomy_targets'],
              'constraint_type': c['constraint_type'], 'check_type': c['check_type'], 'check_hint': c['check_hint']}
    if not guard_matches(c, step):
        result['verdict'] = 'SKIP'
        return result
    observed = prefix(trajectory, offset)
    result['evidence'] = observed['steps'][max(0, offset-3):offset+1]
    try:
        value = _seed_function('\n'.join(c['python_check']['code_lines']),
                               c['python_check']['function_name'])(observed, offset)
        if type(value) is not bool:
            raise ValueError('Python check must return a boolean')
        result['verdict'] = 'SAT' if value else 'VIOL'
    except Exception as exc:  # same text as check_constraint wrapping a worker error
        result['error'] = f'ValueError: {type(exc).__name__}: {str(exc)[:1000]}'
    return result


def _check_all(c, trajectory, provider, context):
    rows = []
    for offset in range(len(trajectory['steps'])):
        row = (_check_seed_inprocess(c, trajectory, offset) if c.get('origin') == 'seed' else
               check_constraint(c, trajectory, offset, provider, context))
        row['offset'] = offset
        rows.append(row)
    return rows


def run_perspective_checks(perspective_id, constraints, trajectory, provider, context):
    """[A3_Pi]: strict guarded checks; an LLM constraint that SKIPs every step has its whole guard relaxed once (Trap 1).

    Returns (checks, effective_constraints), the latter holding relaxed copies where relaxation happened.
    """
    checks, effective = [], []
    for c in constraints:
        rows = _check_all(c, trajectory, provider, context)
        if c.get('origin') != 'seed' and rows and all(r['verdict'] == 'SKIP' for r in rows):
            c = copy.deepcopy(c)
            c['event_trigger'] = dict(ANY_EVENT)  # e.g. role_name "retail_agent" when the trace says "assistant"
            c['relaxed'] = True
            rows = _check_all(c, trajectory, provider, context)
        for row in rows:
            row.update(perspective=perspective_id, origin=c.get('origin', 'llm'), relaxed=bool(c.get('relaxed')))
        effective.append(c)
        checks.extend(rows)
    return checks, effective


def step_actor(trajectory, offset):
    """(canonical agent, raw role, offset of that agent's action). Tool outputs are owned by their caller."""
    steps = trajectory['steps']
    events = steps[offset]['substeps']
    if not events:
        return 'unknown', 'unknown', offset
    for event in events:
        if agent(event['role']) not in PASSIVE:
            return agent(event['role']), event['role'], offset
    if any(agent(e['role']) == 'tool' for e in events):
        for back in range(offset - 1, -1, -1):
            for event in steps[back]['substeps']:
                if agent(event['role']) not in PASSIVE:
                    return agent(event['role']), event['role'], back
    return agent(events[0]['role']), events[0]['role'], offset


def compute_blame_matrix(trajectory, checks):
    """[B3] B[i][k] = V/(V+S+eps) * (1 - (t_min-1)/T) over autonomous agents k; t = 1-based position of k's action."""
    T = max(len(trajectory['steps']), 1)
    actors = [step_actor(trajectory, o) for o in range(len(trajectory['steps']))]
    names = {k for k, _, _ in actors}
    agents = sorted(names - set(NON_AGENTS)) or sorted(names)
    matrix = {pid: {k: 0.0 for k in agents} for pid in ORDER}
    for pid in ORDER:
        for k in matrix[pid]:
            mine = [r for r in checks if r.get('perspective') == pid and actors[r['offset']][0] == k]
            viol = [actors[r['offset']][2] + 1 for r in mine if r['verdict'] == 'VIOL']
            if viol:
                sat = sum(r['verdict'] == 'SAT' for r in mine)
                matrix[pid][k] = len(viol) / (len(viol) + sat + EPS) * (1.0 - (min(viol) - 1) / T)
    return matrix


def encode_candidates(trajectory, checks, blame_matrix):
    """[B4] one (agent, step) candidate per autonomous agent action with a violation (Trap 2), or per agent action
    when nothing is violated. Violations on a tool output are folded into the calling agent's step."""
    steps = trajectory['steps']
    T = max(len(steps), 1)
    actors = [step_actor(trajectory, o) for o in range(len(steps))]
    viol = [r for r in checks if r['verdict'] == 'VIOL' and actors[r['offset']][0] not in NON_AGENTS]
    pool = (sorted({actors[r['offset']][2] for r in viol})
            or [o for o, (k, _, a) in enumerate(actors) if k not in NON_AGENTS and a == o]
            or [o for o, s in enumerate(steps) if s['substeps']])
    candidates = []
    for offset in pool:
        k, raw, actor_offset = actors[offset]
        here = [r for r in viol if actors[r['offset']][2] == offset]
        local = [sum(r['perspective'] == pid for r in here) for pid in ORDER]
        density = [min(1.0, v / 2.0) for v in local]
        blame = [blame_matrix[pid].get(k, 0.0) for pid in ORDER]
        features = blame + density + [
            min(1.0, sum(r.get('check_type') == 'python_check' for r in here) / 2.0),
            1.0 - offset / T,
            sum(v > 0 for v in local) / 3.0,
            max(density)]
        scores = {pid: density[i] + blame[i] for i, pid in enumerate(ORDER)}
        dominant = max(ORDER, key=lambda pid: (scores[pid], -ORDER.index(pid)))
        targets = Counter(t for r in here if r['perspective'] == dominant for t in r.get('taxonomy_targets', [])
                          if t in PERSPECTIVES[dominant]['categories'])
        category = max(targets, key=lambda t: (targets[t], -CATEGORIES.index(t))) if targets else PERSPECTIVES[dominant]['categories'][0]
        names = sorted({r['assertion_name'] for r in here})
        rationale = (f"{dominant} flagged {len(here)} violation(s) at step {steps[offset]['index']} "
                     f"({', '.join(names)}); {k} blame under {dominant} = {blame[ORDER.index(dominant)]:.3f}."
                     if here else f"No perspective violations; step {steps[offset]['index']} ranked by temporal prior.")
        refs = sorted({offset} | {r['offset'] for r in here})  # the agent's step plus folded tool outputs
        candidates.append({
            'agent': k, 'raw_agent': raw, 'step_index': steps[offset]['index'], 'offset': offset,
            'actor_offset': actor_offset, 'features': features, 'perspective_scores': scores,
            'dominant_perspective': dominant, 'dominant_category': category, 'rationale': rationale,
            'evidence_refs': [{'step_index': steps[o]['index'], 'substep_index': steps[o]['substeps'][0]['sub_index']}
                              for o in refs if steps[o]['substeps']],
            'violated_checks': [{'assertion_name': r['assertion_name'], 'perspective': r['perspective'],
                                 'step_index': r['step_index'], 'taxonomy_targets': r.get('taxonomy_targets', []),
                                 'check_type': r.get('check_type'), 'origin': r.get('origin')} for r in here]})
    return candidates
