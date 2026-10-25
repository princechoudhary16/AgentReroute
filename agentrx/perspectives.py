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
# Domain-general. The LEADING segment of a tool's name says whether calling it can change state, so
# 'get_order_details' does not qualify on the strength of containing 'order'. This is vocabulary, not a
# tool list: a list of retail tool names belongs in the retail pack (see RETAIL_MARKERS).
MUTATING_VERBS = ('cancel', 'modify', 'change', 'update', 'set', 'create', 'add', 'remove', 'delete',
                  'exchange', 'return', 'refund', 'submit', 'place', 'book', 'send', 'transfer', 'apply',
                  'assign', 'schedule', 'upload', 'write', 'save', 'post', 'pay', 'purchase')
# An agent that says it could not act has not silently skipped the action.
REFUSAL = r'cannot|can not|unable|could not|couldn\'t|failed|error|\?'
ROUTING_DRIFT = r'cannot handle|cannot proceed|wrong specialist'

# ---- retail pack vocabulary, read only by clauses that domain_packs() activates.
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


# ---------------------------------------------------------------------- tool vocabulary
def _is_schema(value):
    """A tool schema declares its arguments, in one of the two shapes in circulation."""
    return isinstance(value, dict) and (isinstance(value.get('parameters'), dict)
                                        or isinstance(value.get('inputs'), dict))


def _schema_args(schema):
    """``(declared, required)`` argument names, or ``(None, ())`` when the schema declares none.

    ``None`` means the vocabulary is unknown and arguments must not be judged at all, which is a
    different thing from an empty tuple (declared to take no arguments). Two shapes exist:

    * JSON Schema under ``parameters`` (benchmark/env/retail.py TOOL_SCHEMAS), which states ``required``;
    * ``inputs`` as name -> type description, which is what the AgentRx release carries (all 377 schemas
      across data/benchmark/cases.json) and which has no optionality marker, so nothing is required
      there and only undeclared argument names are judged.
    """
    if not isinstance(schema, dict):
        return None, ()
    params = schema.get('parameters')
    if isinstance(params, dict):
        props, required = params.get('properties'), params.get('required')
        return (tuple(props) if isinstance(props, dict) else None,
                tuple(r for r in required if isinstance(r, str)) if isinstance(required, list) else ())
    inputs = schema.get('inputs')
    if isinstance(inputs, dict):
        return tuple(inputs), ()
    return None, ()


def tool_catalog(tools):
    """``{name: {'declared': tuple|None, 'required': tuple}}`` from any tools payload in circulation.

    Four shapes reach the seeds: ``{'tools': {name: schema}}`` (test fixtures and the AgentRx tau
    cases), ``{agent: [schema, ...]}`` (our own corpus export and the recovery harness task view), a
    flat ``{name: schema}`` (rl_cross_branch, which flattens that task view), and a bare list.

    An explicit ``tools`` key is authoritative, so its keys are tool names even when the values declare
    no arguments. Without that key only schema-shaped values count, which is what keeps the AgentRx
    Magentic payload honest: it is ``{'protocol': ..., 'agents': ...}`` and both values are dicts, so a
    bare key scan reports 'protocol' and 'agents' as the known tools and then flags every real tool call
    as unknown. Those 44 traces carry no tool calls, so that bug is latent rather than live today.
    """
    catalog = {}

    def add(name, schema):
        if isinstance(name, str) and name:
            declared, required = _schema_args(schema)
            catalog[name] = {'declared': declared, 'required': required}

    def from_list(items):
        for item in items:
            if isinstance(item, dict):
                spec = item.get('function') if isinstance(item.get('function'), dict) else item
                add(spec.get('name'), spec)

    spec, explicit = tools, False
    if isinstance(tools, dict) and 'tools' in tools:
        spec, explicit = tools['tools'], True
    if isinstance(spec, list):
        from_list(spec)
    elif isinstance(spec, dict):
        if spec and all(isinstance(v, list) for v in spec.values()):
            for items in spec.values():
                from_list(items)
        else:
            for name, value in spec.items():
                if explicit or _is_schema(value):
                    add(name, value)
    return catalog


def tool_names(tools):
    return sorted(tool_catalog(tools))


def mutating_tools(catalog):
    """Catalog names whose leading verb denotes a state change. ``get_order_details`` does not qualify,
    which is why the verb is read from the first name segment and not searched for anywhere in it."""
    return sorted(n for n in catalog if n.split('_')[0].lower() in MUTATING_VERBS)


# ---------------------------------------------------------------------- domain packs
RETAIL_MARKERS = ('cancel_pending_order', 'modify_pending_order_items', 'modify_pending_order_address',
                  'modify_pending_order_payment', 'return_delivered_order_items',
                  'exchange_delivered_order_items', 'modify_user_address')


def domain_packs(tools):
    """Which domain packs the observed tool vocabulary activates.

    Harness^2's split, made real: a check written for one domain must not sit in the core, where it
    inflates that domain's score and contributes nothing anywhere else. A pack adds clauses to a core
    seed rather than adding seeds of its own, so a trace outside the domain runs a strictly smaller
    check under the same assertion name, and no caller has to know which packs exist.
    """
    return ('retail',) if set(tool_catalog(tools)) & set(RETAIL_MARKERS) else ()


def repeats_earlier_tool_result(steps_by_index, step_index):
    """Whether the tool result at ``step_index`` duplicates an earlier identical ``(tool, arguments)`` call.

    Python twin of the stuck-loop clause inside the generated p2 seed, for callers that classify an
    already-ranked candidate (rl_cross_branch._fault_label) instead of checking a prefix. The seed's
    clause is source text executed in the worker sandbox and cannot import this module, so the two are
    held in step by a test rather than by sharing code. Only steps up to ``step_index`` are consulted,
    which is the same prefix the seed sees.
    """
    pending, before, now = {}, [], []
    for index in sorted(steps_by_index):
        for event in steps_by_index[index]['substeps']:
            for call in event.get('tool_calls') or []:
                if isinstance(call, dict):
                    pending[str(call.get('name', ''))] = json.dumps(call.get('arguments'), sort_keys=True)
            if str(event.get('role', '')).lower() != 'tool':
                continue
            name = str(event.get('name', ''))
            triple = (name, pending.get(name, ''), str(event.get('content', '')))
            (now if index == step_index else before).append(triple)
        if index == step_index:
            break
    return any(triple in before for triple in now)


def _seed(name, targets, kind, hint, body, content_regex='*'):
    c = validate_constraint({
        'assertion_name': name, 'taxonomy_targets': list(targets), 'constraint_type': kind,
        'event_trigger': {**ANY_EVENT, 'content_regex': content_regex}, 'check_hint': hint, 'check_type': 'python_check',
        'python_check': {'function_name': name, 'args': ['trajectory', 'current_step_index'],
                         'code_lines': [f'def {name}(trajectory, current_step_index):'] + body}})
    c['origin'] = 'seed'
    return c


def _p1_progress(catalog, packs):
    """Non-empty, non-repeating agent messages, and a state change that was asked for actually attempted.

    The write obligation is read from the REQUEST and the TOOL CALLS, never from the wording of the
    agent's claim. The previous version matched a regex copied from the fault injector's own completion
    strings, so it scored on our benchmark and stayed silent on every external trace.

    A seed check only ever sees ``steps[:current_step_index + 1]`` (constraints.prefix), so every rule
    here is answerable from the observed prefix alone.
    """
    return _seed('p1_seed_nonempty_progress', ['Instruction/PlanAdherenceFailure'], 'PROTOCOL',
                 "Non-tool agent messages must be non-empty, must not repeat the same role's previous "
                 'message verbatim, and must not conclude a request for a state change when no '
                 'state-changing tool call has been made since that request.', [
        "    import json",
        "    import re",
        "    passive = ('user', 'human', 'system', 'tool')",
        "    steps = trajectory['steps']",
        f"    mutating = {mutating_tools(catalog)!r}",
        f"    verbs = {MUTATING_VERBS!r}",
        "    here = steps[current_step_index]['substeps']",
        "    handoff_here = any(e.get('name') == 'handoff' for e in here)",
        "    for event in here:",
        "        role = str(event.get('role', ''))",
        "        if role.lower() in passive or event.get('tool_calls'):",
        "            continue",
        "        content = str(event.get('content', '')).strip()",
        "        if not content:",
        "            return False",
        "        for earlier in reversed(steps[:current_step_index]):",
        "            same = [e for e in earlier['substeps'] if e.get('role') == role]",
        "            if same:",
        "                if str(same[-1].get('content', '')).strip() == content:",
        "                    return False",
        "                break",
        "        # No mutating tool exists in this domain, so no reply can owe a state change.",
        "        if handoff_here or not mutating:",
        "            continue",
        "        request, since = None, 0",
        "        for offset in range(current_step_index - 1, -1, -1):",
        "            for e in steps[offset]['substeps']:",
        "                if e.get('name') != 'handoff':",
        "                    continue",
        "                try:",
        "                    data = json.loads(str(e.get('content', '')))",
        "                except ValueError:",
        "                    continue",
        "                if isinstance(data, dict) and str(data.get('handoff_to', '')).lower() == role.lower():",
        "                    request = str(data.get('message', '')) + ' ' + json.dumps(data.get('context', {}))",
        "                    since = offset",
        "            if request is not None:",
        "                break",
        "        if request is None:",
        "            for offset in range(current_step_index - 1, -1, -1):",
        "                for e in steps[offset]['substeps']:",
        "                    if str(e.get('role', '')).lower() in ('user', 'human'):",
        "                        request = str(e.get('content', ''))",
        "                        since = offset",
        "                if request is not None:",
        "                    break",
        "        if request is None:",
        "            continue",
        "        if not any(re.search(r'\\b' + v, request, re.I) for v in verbs):",
        "            continue",
        f"        if re.search({REFUSAL!r}, content, re.I):",
        "            continue",
        "        done = False",
        "        for s in steps[since:current_step_index]:",
        "            for e in s['substeps']:",
        "                for call in e.get('tool_calls') or []:",
        "                    if isinstance(call, dict) and str(call.get('name', '')) in mutating:",
        "                        done = True",
        "        if not done:",
        "            return False",
        "    return True"])


def _p1_routing(catalog, packs):
    """Refusal drift and repeated identical actions (core), plus retail handoff-target rules (pack)."""
    retail = [
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
    ] if 'retail' in packs else ["            pass"]
    return _seed('p1_seed_specialist_routing_alignment', ['IntentPlanMisalignment'], 'PROTOCOL',
                 'Agents must not drift into refusal or wrong-specialist replies, must not repeat an '
                 'identical tool call a 3rd time, and (retail) must hand off to a specialist that owns '
                 'the work.', [
        "    import json",
        "    import re",
        "    steps = trajectory['steps']",
        "    earlier = ' '.join(str(e.get('content', '')) for s in steps[:current_step_index] for e in s['substeps'])",
        "    seen = [(e.get('role'), call.get('name'), json.dumps(call.get('arguments'), sort_keys=True))",
        "            for s in steps[:current_step_index + 1] for e in s['substeps']",
        "            for call in e.get('tool_calls') or [] if isinstance(call, dict)]",
        "    issuer = ''",
        "    for s in steps[:current_step_index + 1]:",
        "        for e in s['substeps']:",
        "            if e.get('name') == 'handoff':",
        "                issuer = str(e.get('role', '')).lower()",
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
        *retail,
        "            continue",
        "        # The agent that received the work is the drift signal; whoever ISSUED the most recent",
        "        # handoff is only relaying the symptom back, so its own refusal is an echo. Reading the",
        "        # issuer off the trace keeps a role name like 'planner' out of a domain-general check.",
        f"        if str(event.get('role', '')).lower() != issuer and re.search({ROUTING_DRIFT!r},",
        "                                                                     str(event.get('content', '')), re.I):",
        "            return False",
        "        for call in event.get('tool_calls') or []:",
        "            if isinstance(call, dict) and seen.count((event.get('role'), call.get('name'),",
        "                                                      json.dumps(call.get('arguments'), sort_keys=True))) >= 3:",
        "                return False",
        "    return True"], content_regex=f'(?i){ROUTING_DRIFT}|"arguments"|"handoff_to"')


def _p2_invocation(catalog, packs):
    """Tool calls must match their declared schema; repeated identical results are a stuck loop.

    What changed and why. The schema clause is new: ``tools`` was already being passed in and only its
    names were used, so no rule compared a call's arguments with what the tool declares, which is where
    most tau failures start (trace 34 step 17 calls modify_pending_order_items with item_ids /
    new_item_ids / payment_method_id against declared item_ids_to_add / item_ids_to_remove / order_id).

    The ``"status": "processing"`` regex is gone: it was a copy of the injector's own payload. A stuck
    tool is now the structural fact that the identical ``(name, arguments)`` call returned an identical
    result earlier in the prefix.

    A bare tool error is no longer a violation. It fired on 44 tau steps in the baseline, including
    recovered lookups (trace 34 step 6 errors, step 7 succeeds by another route), and a seed cannot look
    ahead to see the recovery because it only receives the observed prefix. An error is evidence here
    only when the same call already failed, which is observable in the prefix; an agent that ignores a
    failure and claims success is P3's obligation, not this one's.
    """
    return _seed('p2_seed_valid_tool_invocation', ['InvalidInvocation', 'SystemFailure'], 'SCHEMA',
                 'Tool calls need object arguments, a known tool name, and argument names the tool '
                 'declares; a result identical to an earlier identical call is a stuck loop.', [
        "    import json",
        "    import re",
        f"    catalog = {dict(catalog)!r}",
        "    steps = trajectory['steps']",
        "    here = steps[current_step_index]['substeps']",
        "    for event in here:",
        "        for call in event.get('tool_calls') or []:",
        "            if not isinstance(call, dict) or not isinstance(call.get('arguments'), dict):",
        "                return False",
        "            name = str(call.get('name', ''))",
        "            if catalog and name not in catalog:",
        "                return False",
        "            spec = catalog.get(name) or {}",
        "            declared = spec.get('declared')",
        "            if declared is None:",
        "                continue",
        "            if [k for k in call['arguments'] if k not in declared]:",
        "                return False",
        "            if [k for k in spec.get('required') or () if k not in call['arguments']]:",
        "                return False",
        "    # (tool, arguments, result) for every tool result in the prefix, so a repeat is observable.",
        "    pending, before, now = {}, [], []",
        "    for offset, s in enumerate(steps[:current_step_index + 1]):",
        "        for e in s['substeps']:",
        "            for call in e.get('tool_calls') or []:",
        "                if isinstance(call, dict):",
        "                    pending[str(call.get('name', ''))] = json.dumps(call.get('arguments'), sort_keys=True)",
        "            if str(e.get('role', '')).lower() != 'tool':",
        "                continue",
        "            name = str(e.get('name', ''))",
        "            triple = (name, pending.get(name, ''), str(e.get('content', '')))",
        "            if offset == current_step_index:",
        "                now.append(triple)",
        "            else:",
        "                before.append(triple)",
        "    for triple in now:",
        "        if triple in before:",
        "            return False",
        f"        if re.search({ERROR_MARKERS!r}, triple[2], re.I) or triple[2].lstrip().startswith('Error'):",
        "            for other in before:",
        "                if other[0] == triple[0] and other[1] == triple[1] and (",
        f"                        re.search({ERROR_MARKERS!r}, other[2], re.I) or other[2].lstrip().startswith('Error')):",
        "                    return False",
        "    return True"])


def _p3_provenance(catalog, packs):
    return _seed('p3_seed_no_contradicted_tool_output',
                 ['MisinterpretationOfToolOutput', 'InventionOfNewInformation'], 'PROVENANCE',
                 'After an empty/error tool output, the next agent message must not claim success or cite unseen IDs.', [
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
        "    return True"])


def _p3_handoff(catalog, packs):
    """Handoff completeness and money grounding (core), plus retail identifier carry-over (pack)."""
    retail = [
        f"        if isinstance(context, dict) and re.search({HANDOFF_ID!r}, earlier) and not any(",
        "                context.get(k) for k in ('order_id', 'user_id', 'item_ids')):",
        "            return False",
    ] if 'retail' in packs else []
    return _seed('p3_seed_handoff_and_fact_grounding', ['MisinterpretationOfToolOutput'], 'RELATIONAL_POST',
                 'Handoffs must carry context and an unredacted message, (retail) must carry the '
                 'order/user ids seen so far, and dollar amounts after a priced tool output must appear '
                 'in the observed prefix.', [
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
        *retail,
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
        "    return True"], content_regex=r'"context"\s*:|"handoff_to"|\$\s?\d')


_BUILDERS = {'P1_INTENT_PLAN': (_p1_progress, _p1_routing),
             'P2_TOOL_EXECUTION': (_p2_invocation,),
             'P3_HANDOFF_GROUNDING': (_p3_provenance, _p3_handoff)}


def seed_constraints_for_perspective(perspective_id, tools):
    """D-10: one wildcard-guarded floor seed per perspective, so it never collapses to all-SKIP, plus
    guarded specialist seeds that judge only the steps their rule applies to (vacuous SATs would dilute
    V/(V+S)). Domain packs activated by the observed tool vocabulary add clauses to these same seeds."""
    if perspective_id not in _BUILDERS:
        raise ValueError(f'Unknown perspective {perspective_id}')
    catalog = tool_catalog(tools)
    return [build(catalog, domain_packs(tools)) for build in _BUILDERS[perspective_id]]



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
    """[B3] B[i][k] = V/(V+S+eps) over autonomous agents k: how often perspective i finds k in violation.

    Earliness used to multiply this by ``(1 - (t_min-1)/T)``, which counted temporal position three
    times over: here, again as feature 7 in encode_candidates, and a third time as select_action's
    tie-break on offset. An early recovered hiccup then outranked a fatal late failure. Blame is now a
    pure violation rate per agent, and position is expressed exactly once, in feature 7.
    """
    actors = [step_actor(trajectory, o) for o in range(len(trajectory['steps']))]
    names = {k for k, _, _ in actors}
    agents = sorted(names - set(NON_AGENTS)) or sorted(names)
    matrix = {pid: {k: 0.0 for k in agents} for pid in ORDER}
    for pid in ORDER:
        for k in matrix[pid]:
            mine = [r for r in checks if r.get('perspective') == pid and actors[r['offset']][0] == k]
            viol = sum(r['verdict'] == 'VIOL' for r in mine)
            if viol:
                sat = sum(r['verdict'] == 'SAT' for r in mine)
                matrix[pid][k] = viol / (viol + sat + EPS)
    return matrix


def _handoff_folds(trajectory, viol, actors):
    """Violating offsets to fold into an upstream violated handoff, so a cause outranks its consequence.

    A specialist starts a fresh conversation on every handoff, so everything it does in that
    sub-conversation is downstream of the handoff it received. When the handoff step is ITSELF in
    violation (ids dropped, wrong specialist), the specialist's confusion is evidence FOR that handoff
    being the fault, not a competing hypothesis. This is the same fold the tool-output case already
    uses, one level up: a violation on a tool result belongs to the agent that made the call.

    Only a violated handoff folds. A specialist that errs under a clean handoff keeps its own candidacy,
    which is what keeps an injected skip-write attributable to the specialist rather than the planner.

    Without this, removing the earliness factor from blame let a downstream symptom outrank its cause:
    on the demo's dropped-context scenario the planner's handoff at step 6 and the specialist's drift at
    step 7 scored 0.175 against 0.183, and the specialist won. Restoring a position prior would have
    papered over that, at the cost of the external traces where position points the wrong way.
    """
    steps = trajectory['steps']
    handoffs = []
    for offset, step in enumerate(steps):
        for event in step['substeps']:
            if event.get('name') != 'handoff':
                continue
            try:
                data = json.loads(str(event.get('content', '')))
            except ValueError:
                continue
            if isinstance(data, dict) and data.get('handoff_to'):
                handoffs.append((offset, agent(str(data['handoff_to']))))
    owned = {actors[r['offset']][2] for r in viol}
    folds = {}
    for offset, target in handoffs:
        if offset not in owned:
            continue
        for other in sorted(owned):
            if other <= offset or actors[other][0] != target:
                continue
            if not [h for h, t in handoffs if offset < h <= other and t == target]:
                folds[other] = offset        # no newer handoff to this agent in between
    return folds


def encode_candidates(trajectory, checks, blame_matrix):
    """[B4] one (agent, step) candidate per autonomous agent action with a violation (Trap 2), or per agent action
    when nothing is violated. Violations on a tool output are folded into the calling agent's step."""
    steps = trajectory['steps']
    T = max(len(steps), 1)
    actors = [step_actor(trajectory, o) for o in range(len(steps))]
    viol = [r for r in checks if r['verdict'] == 'VIOL' and actors[r['offset']][0] not in NON_AGENTS]
    folds = _handoff_folds(trajectory, viol, actors)

    def owner(row):
        base = actors[row['offset']][2]
        return folds.get(base, base)

    acting = [o for o, (k, _, a) in enumerate(actors) if k not in NON_AGENTS and a == o]
    # With violations, every violating agent action is a suspect and theta ranks them. With none, there
    # is nothing to rank: a field of candidates whose order theta decides produced step 2 or 3 on 30 of
    # 44 Magentic traces. The last agent action is the only defensible guess without evidence, and
    # emitting it alone makes that answer independent of the weights rather than an artefact of them.
    pool = (sorted({owner(r) for r in viol})
            or acting[-1:]
            or [o for o, s in enumerate(steps) if s['substeps']][-1:])
    candidates = []
    for offset in pool:
        k, raw, actor_offset = actors[offset]
        here = [r for r in viol if owner(r) == offset]
        local = [sum(r['perspective'] == pid for r in here) for pid in ORDER]
        density = [min(1.0, v / 2.0) for v in local]
        blame = [blame_matrix[pid].get(k, 0.0) for pid in ORDER]
        if viol:
            # Earliness within the trace. The handoff plan proposed measuring this within the violating
            # pool instead, on the premise that the earliest unrecovered violation is the likeliest
            # cause. That premise does not survive the 29 external tau traces: the annotated step is the
            # earliest violating candidate 3 times and the latest 4 times, and the pool-relative version
            # scored 10.3% exact / 34.5% acc@3 against 13.8% / 41.4% for this one. Kept as measured.
            temporal = 1.0 - offset / T
        else:
            # Nothing fired anywhere in the trajectory, so there is no suspect to be early among. Ranking
            # by earliness here answered step 2 or 3 on 30 of 44 Magentic traces, where the annotated step
            # is the last one 30% of the time and step 2 or 3 essentially never. With no evidence the
            # honest prior is the end of the run, and the ranking must say so rather than invert it.
            temporal = offset / T
        features = blame + density + [
            min(1.0, sum(r.get('check_type') == 'python_check' for r in here) / 2.0),
            temporal,
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
