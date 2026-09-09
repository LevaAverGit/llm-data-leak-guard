"""Requisite detector: business reference numbers (contracts, invoices, accounts).

Unlike personal data (Presidio) or credentials (entropy/regex), a contract or
invoice number has no universal shape — it is whatever an organisation decides to
print next to the word *contract*. So this detector is deliberately *keyword
anchored*: it looks for a business noun (``contract``, ``invoice``, ``agreement``
...), optionally a number marker (``No.``, ``#``, ``№``), and then captures the
reference token that follows. Anchoring on the noun keeps precision high — a bare
``44-AB/2024`` in free text is ambiguous, but ``contract 44-AB/2024`` is not.

Only the reference token itself is masked (``[CONTRACT]``); the anchoring noun is
left in place so the redacted prompt still reads naturally for the downstream
model.

Entry point: :func:`detect` — ``detect(text) -> list[Finding]`` — the uniform
detector interface consumed by :mod:`guard.redactor`.
"""

from __future__ import annotations

import re
from typing import List

from guard.schema import Finding

# Detector id recorded on every Finding; matches ``detector_hint`` in the corpus.
DETECTOR_NAME = "requisites_regex"

# Every requisite this detector emits carries the same label.
LABEL = "CONTRACT"

# Business nouns that legitimise a following reference number. Kept specific on
# purpose: generic words like "order" or "number" alone are too noisy, so they
# are only honoured together with a stronger anchor or a number marker.
_ANCHOR = (
    r"(?:contract|agreement|invoice|account|purchase\s+order"
    r"|reference|ref|договор|сч[её]т)"
)

# Optional filler between the anchor noun and the reference: whitespace,
# punctuation and a number marker such as "No.", "number", "№" or "#".
_LEAD = r"[\s:.#№-]*(?:no\.?|number|nr\.?|№|#)?[\s:.#№-]*"

# A reference token: alphanumeric blocks joined by '-' or '/', e.g. "44-AB/2024"
# or "2024/07-1567". The token is captured (group 1) so we can mask exactly it.
_REF = r"([A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+)"

_PATTERN = re.compile(rf"\b{_ANCHOR}\b{_LEAD}{_REF}", re.IGNORECASE)


def _looks_like_reference(token: str) -> bool:
    """Reject shapes that carry a separator but are not real reference numbers.

    Requires a digit and enough substance that a trivial ``2-3`` style fragment
    does not get masked: at least four characters, and either a letter present
    or four or more digits overall.
    """
    if len(token) < 4:
        return False
    digits = sum(c.isdigit() for c in token)
    if digits == 0:
        return False
    has_alpha = any(c.isalpha() for c in token)
    return has_alpha or digits >= 4


def detect(text: str) -> List[Finding]:
    """Return every requisite (contract/invoice/account reference) in ``text``."""
    findings: List[Finding] = []
    for match in _PATTERN.finditer(text):
        ref = match.group(1)
        if not _looks_like_reference(ref):
            continue
        findings.append(
            Finding(
                kind="requisite",
                label=LABEL,
                start=match.start(1),
                end=match.end(1),
                detector=DETECTOR_NAME,
            )
        )
    return findings


# Alias under the class-specific name, for callers that prefer an explicit import.
detect_requisites = detect


class RequisitesDetector:
    """Object wrapper around :func:`detect` for registry-style consumers."""

    name = DETECTOR_NAME

    def detect(self, text: str) -> List[Finding]:
        return detect(text)


__all__ = [
    "DETECTOR_NAME",
    "LABEL",
    "detect",
    "detect_requisites",
    "RequisitesDetector",
]
