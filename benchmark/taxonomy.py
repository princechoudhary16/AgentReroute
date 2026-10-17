"""The project's seven failure modes and mappings from AgentRx and MAST labels."""
import json
import re
from functools import lru_cache
from importlib.resources import files

OUT_OF_SCOPE, INCONCLUSIVE = 'out_of_scope', 'inconclusive'


@lru_cache(maxsize=1)
def spec():
    return json.loads(files('benchmark').joinpath('resources/seven_modes.json').read_text(encoding='utf-8'))


MODES = tuple(m['id'] for m in spec()['modes'])
LABELS = MODES + (OUT_OF_SCOPE,)


def _token(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def _aliases():
    table = {}
    for m in spec()['modes']:
        for alias in (m['id'], m['name'], str(m['number'])):
            table[_token(alias)] = m['id']
    for alias in (OUT_OF_SCOPE, 'other', 'outofscope'):
        table[_token(alias)] = OUT_OF_SCOPE
    for alias in (INCONCLUSIVE, 'abstain', 'none', 'unknown'):
        table[_token(alias)] = INCONCLUSIVE
    extra = {'Instruction adherence': 'instruction_adherence', 'Plan adherence': 'instruction_adherence',
             'Cycle': 'cycles', 'Loop': 'cycles', 'Tool failures': 'tool_failure',
             'Misinterpretation': 'misinterpretation_of_tool_output',
             'Context propagation failures': 'context_propagation', 'Missing details': 'missing_details'}
    table.update({_token(k): v for k, v in extra.items()})
    return table


def mode(value):
    """Normalise a mode name, number or alias to its id. Raises ValueError if unknown."""
    try:
        return _aliases()[_token(value)]
    except KeyError:
        raise ValueError(f'Unknown failure mode: {value!r}') from None


def from_agentrx(category):
    """(mode, fit) for an AgentRx category name."""
    table = spec()['mappings']['agentrx']
    key = {_token(k): k for k in table}.get(_token(category))
    if key is None:
        raise ValueError(f'Unknown AgentRx category: {category!r}')
    return table[key]['mode'], table[key]['fit']


def to_agentrx(mode_id):
    return spec()['mappings']['to_agentrx'][mode(mode_id)]


def from_mast(label):
    table = spec()['mappings']['mast']
    code = re.match(r'\s*(FM-\d\.\d+)', str(label))
    for key, value in table.items():
        if (code and key.startswith(code.group(1) + ' ')) or _token(key) == _token(label):
            return value['mode'], value['fit']
    raise ValueError(f'Unknown MAST label: {label!r}')


def describe(checklist=True):
    """Mode definitions for prompts (baseline judges) and documentation."""
    return [{'mode': m['id'], 'name': m['name'], 'definition': m['definition'],
             **({'decision_rules': m['decision_rules']} if checklist else {})} for m in spec()['modes']]


def agent_name(value):
    return _token(value)
