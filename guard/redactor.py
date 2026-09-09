"""Redactor + combined guard pipeline.

This is the join point of the guard: it takes the ``Finding`` objects produced
by every detector and turns them, plus the original text, into a single
``GuardResult``. Three responsibilities:

1. **Merge** — findings from different detectors can overlap (a phone number
   caught by two recognisers, a secret spanning a substring another rule also
   matched). Overlaps are de-duplicated with a *longest-span-wins* rule so each
   region of the text is masked exactly once.
2. **Redact** — surviving spans are rewritten *right-to-left* (highest offset
   first) so that splicing a ``[LABEL]`` mask never invalidates the offsets of
   spans still to be applied.
3. **Report** — everything is packaged into a ``GuardResult`` whose
   ``latency_ms`` measures the whole guard pass (detection + merge + redaction).

The combined pipeline, :func:`guard`, discovers the available detector modules
(``guard.pii``, ``guard.secrets``, ``guard.requisites``), runs each over the
input and reduces the union of findings through the steps above. Detectors are
discovered defensively: a module that is not importable in the current
environment (for example when a heavy ML dependency is absent) is simply skipped,
so the guard still runs with whatever detectors are present. Robustness extends
to call time too — if a resolved detector raises while inspecting text (a lazily
loaded dependency turning out to be missing, or a transient error), that single
detector is skipped and the remaining detectors still redact.

No raw input is logged here; only the redacted text ever leaves this module.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import Callable, Iterable, List, Optional, Sequence

from guard.schema import Finding, GuardResult

logger = logging.getLogger(__name__)

# A detector is any callable mapping text to a list of findings.
Detector = Callable[[str], List[Finding]]

# Detector modules that make up the combined pipeline. Order is presentation
# order only; merge de-duplication does not depend on it.
_DETECTOR_MODULES = ("guard.pii", "guard.secrets", "guard.requisites")

# Function names accepted as a module's detector entry point, most-preferred
# first. Every detector module in this project exposes ``detect``; the extra
# aliases keep discovery robust to naming differences.
_ENTRY_POINTS = (
    "detect",
    "detect_pii",
    "detect_secrets",
    "detect_requisites",
    "run",
    "find",
)


def _find_entry_point(module) -> Optional[Detector]:
    """Return the first callable on ``module`` matching a known entry-point name."""
    for name in _ENTRY_POINTS:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None


def load_detectors() -> List[Detector]:
    """Import the detector modules and collect their entry points.

    Modules that cannot be imported in the current environment are skipped, so
    the pipeline degrades gracefully to the detectors that are available.
    """
    detectors: List[Detector] = []
    for module_name in _DETECTOR_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            # Missing/incomplete detector module — skip it rather than fail.
            continue
        entry = _find_entry_point(module)
        if entry is not None:
            detectors.append(entry)
    return detectors


def merge_findings(findings: Iterable[Finding]) -> List[Finding]:
    """De-duplicate overlapping findings, keeping the longest span.

    Findings are considered greedily from longest to shortest; a finding is kept
    only if it does not overlap one already kept. Ties (equal length) are broken
    deterministically by start offset, then label, then detector, so the result
    is stable regardless of detector ordering. The returned list is sorted by
    ``start`` and is guaranteed to contain no overlapping spans.
    """
    ordered = sorted(
        findings,
        key=lambda f: (-f.length, f.start, f.label, f.detector),
    )
    kept: List[Finding] = []
    for finding in ordered:
        if any(finding.overlaps(other) for other in kept):
            continue
        kept.append(finding)
    kept.sort(key=lambda f: f.start)
    return kept


def _apply_masks(text: str, merged: Sequence[Finding]) -> str:
    """Splice each finding's mask into ``text``, right-to-left.

    Assumes ``merged`` contains no overlapping spans (see :func:`merge_findings`).
    Working from the highest start offset downwards keeps every not-yet-applied
    span's offsets valid as earlier text is left untouched until its turn.
    """
    redacted = text
    for finding in sorted(merged, key=lambda f: f.start, reverse=True):
        redacted = redacted[: finding.start] + finding.mask + redacted[finding.end :]
    return redacted


def redact(text: str, findings: Iterable[Finding]) -> str:
    """Return ``text`` with every finding replaced by its ``[LABEL]`` mask.

    Safe to call with raw, possibly-overlapping findings: they are merged first.
    """
    return _apply_masks(text, merge_findings(findings))


def build_result(
    text: str,
    findings: Iterable[Finding],
    latency_ms: float,
) -> GuardResult:
    """Assemble a ``GuardResult`` from raw findings over ``text``.

    Findings are merged (longest-span-wins) before masking and reporting, so the
    ``findings`` stored on the result are exactly the spans that were redacted.
    """
    merged = merge_findings(findings)
    return GuardResult(
        original_len=len(text),
        redacted=_apply_masks(text, merged),
        findings=merged,
        latency_ms=latency_ms,
    )


def guard(text: str, detectors: Optional[Sequence[Detector]] = None) -> GuardResult:
    """Run the full guard pass over ``text`` and return a ``GuardResult``.

    Times the whole pass — detection, merge and redaction — and reports it as
    ``latency_ms``. When ``detectors`` is omitted, the available detector modules
    are discovered automatically (:func:`load_detectors`).
    """
    start = time.perf_counter()

    active = list(detectors) if detectors is not None else load_detectors()

    raw: List[Finding] = []
    for detector in active:
        try:
            raw.extend(detector(text))
        except Exception:
            # One detector failing (e.g. a heavy ML dependency missing, or a
            # transient runtime error) must never take down redaction for the
            # other classes. Skip it and keep going. The raw text is never
            # logged — only the detector's module/identity.
            logger.warning(
                "detector %r failed; skipping",
                getattr(detector, "__module__", detector),
            )
            continue

    merged = merge_findings(raw)
    redacted = _apply_masks(text, merged)

    latency_ms = (time.perf_counter() - start) * 1000.0

    return GuardResult(
        original_len=len(text),
        redacted=redacted,
        findings=merged,
        latency_ms=latency_ms,
    )


__all__ = [
    "Detector",
    "load_detectors",
    "merge_findings",
    "redact",
    "build_result",
    "guard",
]
