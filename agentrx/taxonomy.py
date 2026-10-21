"""Paper's nine categories, with an explicit abstention outcome."""
import re
from importlib.resources import files
import json

CATEGORIES = (
    'Instruction/PlanAdherenceFailure', 'InventionOfNewInformation',
    'InvalidInvocation', 'MisinterpretationOfToolOutput', 'IntentPlanMisalignment',
    'UnderspecifiedUserIntent', 'IntentNotSupported', 'GuardrailsTriggered',
    'SystemFailure', 'Inconclusive',
)


def token(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


ALIASES = {token(c): c for c in CATEGORIES}
for alias, target in {
    'Instruction Adherence Failure': CATEGORIES[0],
    'Plan Adherence Failure': CATEGORIES[0],
    'Misinterpretation of Tool Output / Handoff Failure': CATEGORIES[3],
    'Under-specified User Intent': CATEGORIES[5],
    'Inconclusive (USE SPARINGLY)': CATEGORIES[9],
}.items():
    ALIASES[token(alias)] = target


def category(value):
    if type(value) is int or (isinstance(value, str) and value.isdigit()):
        number = int(value)
        if 1 <= number <= 10:
            return CATEGORIES[number - 1]
    try:
        return ALIASES[token(value)]
    except KeyError:
        raise ValueError(f'Unknown failure category: {value!r}') from None


def agent(value):
    value = re.sub(r'^Orchestrator \((?:thought|termination condition|-> [^)]+)\)$', 'Orchestrator', str(value), flags=re.I)
    result = token(value)
    aliases = {'assistantagent': 'assistant', 'websurferagent': 'websurfer',
               'filesurferagent': 'filesurfer', 'orchestratoragent': 'orchestrator', 'human': 'user'}
    return aliases.get(result, result)


def taxonomy(checklist=False):
    records = json.loads(files('agentrx').joinpath('resources/taxonomy.json').read_text(encoding='utf-8'))
    return [{'category': CATEGORIES[int(k) - 1], 'name': v['name'],
             'description': v['desc_checklist' if checklist else 'desc_standard'],
             **({'checklist': v['checklist']} if checklist else {})}
            for k, v in records.items()]
