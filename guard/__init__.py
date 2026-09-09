"""llm-data-leak-guard — guard package.

Public contracts live in :mod:`guard.schema`. Detectors, the redactor and the
combined pipeline are added on top of these models in later milestones.
"""

from guard.schema import (
    Finding,
    FindingKind,
    GuardResult,
    MASK_TEMPLATE,
    mask_for,
)

__all__ = [
    "Finding",
    "FindingKind",
    "GuardResult",
    "MASK_TEMPLATE",
    "mask_for",
]

__version__ = "0.1.0"
