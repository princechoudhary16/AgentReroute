"""Fault categories understood by the repair strategies.

The canonical names are the AgentRx categories consumed by the orchestrator,
plus ``Cycle`` for loops found by runtime detectors. The benchmark's seven
failure modes are accepted as input and kept as a *subtype*, because some
strategies specialize on it (a handoff failure is repaired differently from a
misread tool output, although AgentRx files both under one category).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

INSTRUCTION_ADHERENCE = 'Instruction/PlanAdherenceFailure'
INVENTION = 'InventionOfNewInformation'
INVALID_INVOCATION = 'InvalidInvocation'
MISINTERPRETATION = 'MisinterpretationOfToolOutput'
INTENT_MISALIGNMENT = 'IntentPlanMisalignment'
UNDERSPECIFIED_INTENT = 'UnderspecifiedUserIntent'
INTENT_NOT_SUPPORTED = 'IntentNotSupported'
GUARDRAILS = 'GuardrailsTriggered'
SYSTEM_FAILURE = 'SystemFailure'
INCONCLUSIVE = 'Inconclusive'
CYCLE = 'Cycle'

# Order matches the 1-based numbering used by AgentRx prompts.
AGENTRX_CATEGORIES = (
    INSTRUCTION_ADHERENCE, INVENTION, INVALID_INVOCATION, MISINTERPRETATION,
    INTENT_MISALIGNMENT, UNDERSPECIFIED_INTENT, INTENT_NOT_SUPPORTED, GUARDRAILS,
    SYSTEM_FAILURE, INCONCLUSIVE,
)
CATEGORIES = AGENTRX_CATEGORIES + (CYCLE,)

# Subtypes refine a category; they come from the benchmark's mode ids.
HANDOFF = 'context_propagation'
OMISSION = 'missing_details'

_MODES = {
    'instruction_adherence': (INSTRUCTION_ADHERENCE, None),
    'drift': (INTENT_MISALIGNMENT, 'drift'),
    'cycles': (CYCLE, None),
    'tool_failure': (SYSTEM_FAILURE, 'tool_failure'),
    'misinterpretation_of_tool_output': (MISINTERPRETATION, None),
    HANDOFF: (MISINTERPRETATION, HANDOFF),
    OMISSION: (INVENTION, OMISSION),
    'inconclusive': (INCONCLUSIVE, None),
}

_EXTRA_ALIASES = {
    'Instruction Adherence Failure': INSTRUCTION_ADHERENCE,
    'Plan Adherence Failure': INSTRUCTION_ADHERENCE,
    'Misinterpretation of Tool Output / Handoff Failure': MISINTERPRETATION,
    'Under-specified User Intent': UNDERSPECIFIED_INTENT,
    'Inconclusive (USE SPARINGLY)': INCONCLUSIVE,
    'Cycles': CYCLE,
    'Loop': CYCLE,
}


def _token(value) -> str:
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


_CATEGORY_ALIASES = {_token(c): c for c in CATEGORIES}
_CATEGORY_ALIASES.update({_token(alias): target for alias, target in _EXTRA_ALIASES.items()})
_MODE_ALIASES = {_token(mode): mode for mode in _MODES}


class UnknownCategory(ValueError):
    pass


@dataclass(frozen=True)
class FaultType:
    category: str
    subtype: str | None = None


def normalize(value, subtype: str | None = None) -> FaultType:
    """Map an AgentRx label, a 1-based AgentRx number, or a benchmark mode id to a ``FaultType``.

    An explicit ``subtype`` (a benchmark mode id) overrides the one implied by ``value``.
    """
    if value is None:
        raise UnknownCategory('Missing fault category')
    if type(value) is int or (isinstance(value, str) and value.strip().isdigit()):
        number = int(value)
        if not 1 <= number <= len(AGENTRX_CATEGORIES):
            raise UnknownCategory(f'Fault category number out of range: {value!r}')
        fault = FaultType(AGENTRX_CATEGORIES[number - 1])
    else:
        key = _token(value)
        if key in _MODE_ALIASES:
            category, implied = _MODES[_MODE_ALIASES[key]]
            fault = FaultType(category, implied)
        elif key in _CATEGORY_ALIASES:
            fault = FaultType(_CATEGORY_ALIASES[key])
        else:
            raise UnknownCategory(f'Unknown fault category: {value!r}')
    if subtype is not None:
        key = _token(subtype)
        if key not in _MODE_ALIASES:
            raise UnknownCategory(f'Unknown fault subtype: {subtype!r}')
        mode_category, implied = _MODES[_MODE_ALIASES[key]]
        if mode_category != fault.category:
            raise UnknownCategory(f'Subtype {subtype!r} does not refine {fault.category}')
        fault = FaultType(fault.category, implied)
    return fault


def normalize_agent(value) -> str | None:
    """Canonical agent token: 'Orchestrator (-> WebSurfer)' -> 'orchestrator', 'Human' -> 'user'."""
    if value is None:
        return None
    text = re.sub(r'^Orchestrator \((?:thought|termination condition|-> [^)]+)\)$', 'Orchestrator',
                  str(value), flags=re.I)
    token = re.sub(r'[^a-z0-9_]', '', text.lower())
    aliases = {'assistantagent': 'assistant', 'websurferagent': 'websurfer', 'filesurferagent': 'filesurfer',
               'orchestratoragent': 'orchestrator', 'human': 'user'}
    return aliases.get(token, token)
