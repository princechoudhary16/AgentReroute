"""The strategy catalogue: one ladder of strategies per fault category."""
from __future__ import annotations

from .base import REGISTRY_KINDS, RepairContext, Skip, Strategy, StrategySpec
from .escalation import AbstainUndiagnosed, CompensateAndEscalate, TargetedClarification, VerifiedUnmodifiedReplay
from .execution import (HardCallRepair, LoopForceChange, LoopNudge, LoopReplan, LoopRollback, SwitchOnSchemaDrift,
                        ToolOutcomeRouting)
from .grounding import AbstainOrAsk, CompleteOmittedDetails, GroundedReplacement
from .interpretation import CrossSourceCheck, IndependentReading, RepairHandoff, StructuredReading
from .planning import ConfirmAmbiguousWrites, IntentContractReplan, PolicyGate, PolicyGateFresh

CATALOGUE = (
    GroundedReplacement, CompleteOmittedDetails, AbstainOrAsk,
    RepairHandoff, StructuredReading, IndependentReading, CrossSourceCheck,
    IntentContractReplan, ConfirmAmbiguousWrites,
    PolicyGate, PolicyGateFresh,
    HardCallRepair, SwitchOnSchemaDrift,
    LoopNudge, LoopForceChange, LoopReplan, LoopRollback,
    ToolOutcomeRouting,
    TargetedClarification,
    CompensateAndEscalate,
    VerifiedUnmodifiedReplay, AbstainUndiagnosed,
)


def default_strategies() -> tuple[Strategy, ...]:
    strategies = tuple(cls() for cls in CATALOGUE)
    ids = [s.spec.id for s in strategies]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise RuntimeError(f'duplicate strategy ids: {sorted(duplicates)}')
    return strategies


__all__ = ['CATALOGUE', 'REGISTRY_KINDS', 'RepairContext', 'Skip', 'Strategy', 'StrategySpec', 'default_strategies']
