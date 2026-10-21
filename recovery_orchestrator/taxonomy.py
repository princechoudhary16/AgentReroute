"""AgentRx failure categories, label normalization, and default handling.

Mirrors the normalization in the fault-attribution branch (``agentrx/taxonomy.py``)
without importing it, so the two branches stay independent.
"""
from __future__ import annotations

import re

CATEGORIES = (
    'Instruction/PlanAdherenceFailure', 'InventionOfNewInformation',
    'InvalidInvocation', 'MisinterpretationOfToolOutput', 'IntentPlanMisalignment',
    'UnderspecifiedUserIntent', 'IntentNotSupported', 'GuardrailsTriggered',
    'SystemFailure', 'Inconclusive',
)
INCONCLUSIVE = 'Inconclusive'

# Orchestrator-local label for loops found by the runtime detector. AgentRx has
# no "Cycles" category, so it never appears in FA output.
CYCLE = 'Cycle'


def _token(value) -> str:
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


_ALIASES = {_token(c): c for c in CATEGORIES}
_ALIASES.update({_token(alias): target for alias, target in {
    'Instruction Adherence Failure': CATEGORIES[0],
    'Plan Adherence Failure': CATEGORIES[0],
    'Misinterpretation of Tool Output / Handoff Failure': CATEGORIES[3],
    'Under-specified User Intent': CATEGORIES[5],
    'Inconclusive (USE SPARINGLY)': CATEGORIES[9],
}.items()})
_ALIASES[_token(CYCLE)] = CYCLE


def normalize_category(value) -> str:
    """Map any FA spelling (or 1-based number) to a canonical category name."""
    if type(value) is int or (isinstance(value, str) and value.isdigit()):
        number = int(value)
        if 1 <= number <= len(CATEGORIES):
            return CATEGORIES[number - 1]
    try:
        return _ALIASES[_token(value)]
    except KeyError:
        raise ValueError(f'Unknown failure category: {value!r}') from None


def normalize_agent(value) -> str | None:
    """Canonical agent token, e.g. 'Orchestrator (-> WebSurfer)' -> 'orchestrator'."""
    if value is None:
        return None
    value = re.sub(r'^Orchestrator \((?:thought|termination condition|-> [^)]+)\)$',
                   'Orchestrator', str(value), flags=re.I)
    result = _token(value)
    aliases = {'assistantagent': 'assistant', 'websurferagent': 'websurfer',
               'filesurferagent': 'filesurfer', 'orchestratoragent': 'orchestrator',
               'human': 'user'}
    return aliases.get(result, result)


# Handling modes. 'repair' means: roll back and pick a strategy from the registry.
ABSTAIN, RETRY, CLARIFY, ESCALATE, REPAIR = 'abstain', 'retry', 'clarify', 'escalate', 'repair'

HANDLING = {
    INCONCLUSIVE: ABSTAIN,
    'SystemFailure': RETRY,
    'UnderspecifiedUserIntent': CLARIFY,
    'IntentNotSupported': ESCALATE,
    'GuardrailsTriggered': ESCALATE,
}


def handling(category: str) -> str:
    return HANDLING.get(category, REPAIR)
