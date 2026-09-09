"""Core data contracts for llm-data-leak-guard.

These Pydantic v2 models are the stable interface shared by every detector,
the redactor, the FastAPI proxy and the test suite. A detector produces
``Finding`` objects; the redactor turns them into a redacted string and the
whole inspection is reported as a ``GuardResult``.

Masking uses a single, uniform format: a finding is replaced in the outgoing
text by ``[LABEL]`` (for example ``[PHONE]`` or ``[API_KEY]``). Keeping one
mask shape makes redacted prompts predictable for the downstream model and
easy to assert on in tests.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# A leak always belongs to exactly one of these families.
#   pii       - personal data (names, phones, emails, passports, cards, ...)
#   secret    - credentials/keys (API keys, private keys, bearer tokens, ...)
#   requisite - business identifiers (contract/account numbers, ...)
FindingKind = Literal["pii", "secret", "requisite"]

# Single mask format used everywhere. A label ``PHONE`` becomes ``[PHONE]``.
MASK_TEMPLATE = "[{label}]"


def mask_for(label: str) -> str:
    """Return the uniform mask string for a given label, e.g. ``[PHONE]``."""
    return MASK_TEMPLATE.format(label=label)


class Finding(BaseModel):
    """A single detected leak inside an inspected text.

    ``start`` / ``end`` are Python string offsets (half-open, ``text[start:end]``)
    into the *original* text, so the redactor can splice masks in deterministically.
    """

    model_config = ConfigDict(extra="forbid")

    kind: FindingKind = Field(..., description="Leak family: pii, secret or requisite.")
    label: str = Field(..., min_length=1, description="Human-readable class, e.g. PHONE, API_KEY.")
    start: int = Field(..., ge=0, description="Start offset (inclusive) in the original text.")
    end: int = Field(..., ge=0, description="End offset (exclusive) in the original text.")
    detector: str = Field(..., min_length=1, description="Detector that produced this finding.")

    @model_validator(mode="after")
    def _validate_span(self) -> "Finding":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        return self

    @property
    def length(self) -> int:
        """Number of characters this finding spans in the original text."""
        return self.end - self.start

    @property
    def mask(self) -> str:
        """The replacement string written into the redacted text, e.g. ``[PHONE]``."""
        return mask_for(self.label)

    def overlaps(self, other: "Finding") -> bool:
        """True if this finding's span intersects another's (used to de-duplicate)."""
        return self.start < other.end and other.start < self.end


class GuardResult(BaseModel):
    """Outcome of inspecting one outgoing prompt before it reaches the model."""

    model_config = ConfigDict(extra="forbid")

    original_len: int = Field(..., ge=0, description="Character length of the original input.")
    redacted: str = Field(..., description="Input with every finding replaced by its mask.")
    findings: List[Finding] = Field(default_factory=list, description="Everything the guard caught.")
    latency_ms: float = Field(..., ge=0, description="Wall-clock inspection time in milliseconds.")

    @property
    def is_clean(self) -> bool:
        """True when nothing sensitive was detected."""
        return len(self.findings) == 0

    @property
    def count(self) -> int:
        """How many leaks were caught."""
        return len(self.findings)
