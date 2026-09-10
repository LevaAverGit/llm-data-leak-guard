"""PII detection built on Microsoft Presidio plus custom Russian recognizers.

This module turns Presidio ``RecognizerResult`` objects into the project's
:class:`~guard.schema.Finding` contract. It covers:

* names, emails and payment cards via Presidio's predefined recognizers
  (spaCy ``en_core_web_sm`` for names, regex + Luhn for cards);
* Russian phone numbers, internal passports, INN (taxpayer id) and SNILS
  (insurance id) via custom recognizers registered into the Presidio engine.

Design note — lazy initialisation
---------------------------------
Presidio and its spaCy model are a heavy optional dependency. They are imported
*inside* the builder functions, never at module import time, so that
``import guard.pii`` always succeeds. The NLP stack is loaded on the first
``detect`` call; if either the ``presidio-analyzer`` package or the spaCy model
is missing, a :class:`PiiUnavailableError` is raised with an actionable message
instead of an opaque ``ImportError``/``OSError`` from deep inside the stack.

The pure-Python helpers below (regexes and checksum validators) carry no heavy
imports, so the detection-critical logic can be unit-tested on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from guard.schema import Finding

__all__ = [
    "PiiDetector",
    "PiiUnavailableError",
    "detect",
    "luhn_ok",
    "inn_ok",
    "snils_ok",
    "SPACY_MODEL",
]

# spaCy model the proxy/bench/tests expect (installed via `make install`).
SPACY_MODEL = "en_core_web_sm"

# ``detector`` string written onto every Finding, mirroring the corpus hints.
_DETECTOR = "presidio"
_DETECTOR_RU = "presidio_ru"

# Presidio entity type -> Finding label. Everything here is ``kind="pii"``.
_ENTITY_TO_LABEL = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "CREDIT_CARD": "CARD",
    "RU_PHONE": "PHONE",
    "RU_PASSPORT": "PASSPORT",
    "RU_INN": "INN",
    "RU_SNILS": "SNILS",
}

# Only these entities are requested from the analyzer; other predefined
# recognizers (US SSN, IBAN, dates, ...) are intentionally not solicited.
_REQUESTED_ENTITIES = list(_ENTITY_TO_LABEL)

# Entities produced by our custom RU recognizers (labelled ``presidio_ru``).
_CUSTOM_ENTITIES = frozenset({"RU_PHONE", "RU_PASSPORT", "RU_INN", "RU_SNILS"})

# Below this analyzer score a result is discarded. Kept low so pattern-only
# recognizers (phone/passport) survive, but non-zero so results a recognizer
# explicitly invalidated (checksum failure -> score 0) are dropped.
_SCORE_THRESHOLD = 0.1


# --------------------------------------------------------------------------- #
# Custom RU recognizer patterns (half-open matches, boundary-guarded)
# --------------------------------------------------------------------------- #
# +7 / 8 prefix, then 10 digits grouped 3-3-2-2 with optional spaces, dashes
# and parentheses around the operator code. Guards stop mid-number matches.
_RU_PHONE_REGEX = (
    r"(?<!\d)(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
)
# RU internal passport: 4-digit series, space, 6-digit number.
_RU_PASSPORT_REGEX = r"(?<!\d)\d{4}\s\d{6}(?!\d)"
# INN: 10 digits (legal entity) or 12 digits (individual), checksum re-checked.
_RU_INN_REGEX = r"(?<!\d)\d{10}(?:\d{2})?(?!\d)"
# SNILS: XXX-XXX-XXX YY (space or dash before the 2-digit control block).
_RU_SNILS_REGEX = r"(?<!\d)\d{3}-\d{3}-\d{3}[\s\-]\d{2}(?!\d)"


# --------------------------------------------------------------------------- #
# Pure-Python checksum validators (no heavy imports; unit-testable directly)
# --------------------------------------------------------------------------- #
def _digits(value: str) -> str:
    """Strip every non-digit character, leaving only the numeric payload."""
    return re.sub(r"\D", "", value)


def luhn_ok(value: str) -> bool:
    """Return True if the digits of ``value`` satisfy the Luhn checksum."""
    ds = [int(c) for c in _digits(value)]
    if len(ds) < 12:
        return False
    total = 0
    parity = len(ds) % 2
    for i, d in enumerate(ds):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def inn_ok(value: str) -> bool:
    """Validate a Russian INN (10-digit legal entity or 12-digit individual)."""
    ds = _digits(value)

    def _check(weights: List[int], digits: str) -> int:
        return (sum(w * int(c) for w, c in zip(weights, digits)) % 11) % 10

    if len(ds) == 10:
        weights = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        return _check(weights, ds[:9]) == int(ds[9])
    if len(ds) == 12:
        w11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        w12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        return _check(w11, ds[:10]) == int(ds[10]) and _check(w12, ds[:11]) == int(ds[11])
    return False


def snils_ok(value: str) -> bool:
    """Validate a Russian SNILS control number (XXX-XXX-XXX YY)."""
    ds = _digits(value)
    if len(ds) != 11:
        return False
    body, control = ds[:9], int(ds[9:])
    total = sum((9 - i) * int(c) for i, c in enumerate(body))
    if total < 100:
        expected = total
    elif total in (100, 101):
        expected = 0
    else:
        expected = total % 101
        if expected == 100:
            expected = 0
    return expected == control


# Post-detection checksum gate keyed by Finding label. Labels absent here
# (PHONE, PASSPORT, EMAIL, PERSON) carry no checksum and are always accepted.
_CHECKSUM_GATE = {
    "INN": inn_ok,
    "SNILS": snils_ok,
    "CARD": luhn_ok,
}


class PiiUnavailableError(RuntimeError):
    """Raised when Presidio or its spaCy model cannot be imported/loaded.

    The message tells the caller exactly how to install the missing piece,
    so a fresh checkout without the NLP stack fails loudly but clearly rather
    than with an opaque traceback from deep inside Presidio.
    """


@dataclass
class _Candidate:
    """Internal, pre-dedup detection carrying the analyzer score."""

    start: int
    end: int
    label: str
    detector: str
    score: float


# --------------------------------------------------------------------------- #
# Lazy Presidio wiring
# --------------------------------------------------------------------------- #
def _build_ru_recognizers() -> List[Any]:
    """Construct the custom Russian ``PatternRecognizer`` instances.

    Defined here (not at module scope) so the ``presidio_analyzer`` import stays
    lazy. Checksum-bearing recognizers get a ``validate_result`` that shapes the
    score; the authoritative checksum gate still runs again in :meth:`detect`.
    """
    from presidio_analyzer import Pattern, PatternRecognizer

    class _RuRecognizer(PatternRecognizer):
        def __init__(
            self,
            entity: str,
            name: str,
            regex: str,
            score: float,
            context: Optional[List[str]] = None,
            validator: Optional[Callable[[str], bool]] = None,
        ) -> None:
            super().__init__(
                supported_entity=entity,
                name=name,
                patterns=[Pattern(name=name, regex=regex, score=score)],
                context=context,
                supported_language="en",
            )
            self._validator = validator

        def validate_result(self, pattern_text: str) -> Optional[bool]:
            if self._validator is None:
                return None  # no checksum -> leave the pattern score as-is
            return bool(self._validator(pattern_text))

    return [
        _RuRecognizer(
            "RU_PHONE",
            "ru_phone_recognizer",
            _RU_PHONE_REGEX,
            0.75,
            context=["phone", "call", "tel", "mobile", "contact", "number"],
        ),
        _RuRecognizer(
            "RU_PASSPORT",
            "ru_passport_recognizer",
            _RU_PASSPORT_REGEX,
            0.6,
            context=["passport", "series", "document", "id"],
        ),
        _RuRecognizer(
            "RU_INN",
            "ru_inn_recognizer",
            _RU_INN_REGEX,
            0.7,
            context=["inn", "taxpayer", "counterparty", "tax"],
            validator=inn_ok,
        ),
        _RuRecognizer(
            "RU_SNILS",
            "ru_snils_recognizer",
            _RU_SNILS_REGEX,
            0.7,
            context=["snils", "insurance", "pension"],
            validator=snils_ok,
        ),
    ]


def _build_analyzer(model_name: str) -> Any:
    """Build a Presidio ``AnalyzerEngine`` using the small spaCy model.

    Raises :class:`PiiUnavailableError` with a fix-it message if the Presidio
    package or the spaCy model is missing.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as exc:  # package not installed
        raise PiiUnavailableError(
            "presidio-analyzer is not installed. Set up the project with "
            "`make install` (or `pip install -r requirements.txt`)."
        ) from exc

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": model_name}],
    }
    try:
        nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
    except (OSError, ValueError) as exc:  # spaCy model not downloaded
        raise PiiUnavailableError(
            f"spaCy model '{model_name}' is not available. Install it with "
            f"`python -m spacy download {model_name}` (run automatically by "
            "`make install`)."
        ) from exc

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    for recognizer in _build_ru_recognizers():
        analyzer.registry.add_recognizer(recognizer)
    return analyzer


# --------------------------------------------------------------------------- #
# Public detector
# --------------------------------------------------------------------------- #
class PiiDetector:
    """Detects personal data in a text and maps hits to ``Finding`` objects.

    The heavy Presidio engine is created lazily on first use (or eagerly when
    ``eager=True``), so constructing a ``PiiDetector`` is cheap and side-effect
    free. Detection is deterministic: overlapping candidates are resolved by
    preferring the longer span, then the higher analyzer score.
    """

    def __init__(self, model: str = SPACY_MODEL, eager: bool = False) -> None:
        self._model = model
        self._analyzer: Any = None
        if eager:
            self._ensure_ready()

    @property
    def is_ready(self) -> bool:
        """True once the Presidio engine has been built."""
        return self._analyzer is not None

    def _ensure_ready(self) -> None:
        if self._analyzer is None:
            self._analyzer = _build_analyzer(self._model)

    def detect(self, text: str) -> List[Finding]:
        """Return every PII ``Finding`` in ``text`` (sorted by start offset)."""
        if not text:
            return []
        self._ensure_ready()

        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=_REQUESTED_ENTITIES,
            score_threshold=_SCORE_THRESHOLD,
        )

        candidates: List[_Candidate] = []
        for result in results:
            label = _ENTITY_TO_LABEL.get(result.entity_type)
            if label is None:
                continue
            span_text = text[result.start : result.end]
            # spaCy's statistical NER sometimes swallows a numeric identifier
            # (INN, SNILS, passport, ...) into a PERSON span, e.g. it tags
            # "INN 770123456703" as a person. A real personal name never
            # contains a digit, so a digit-bearing PERSON is a false positive:
            # drop it so it neither masks context as [PERSON] nor shadows the
            # precise structured finding produced by the RU recognizers.
            if label == "PERSON" and any(ch.isdigit() for ch in span_text):
                continue
            gate = _CHECKSUM_GATE.get(label)
            if gate is not None and not gate(span_text):
                continue  # checksum failed -> not a real identifier
            detector = _DETECTOR_RU if result.entity_type in _CUSTOM_ENTITIES else _DETECTOR
            candidates.append(
                _Candidate(
                    start=result.start,
                    end=result.end,
                    label=label,
                    detector=detector,
                    score=float(result.score),
                )
            )

        return [
            Finding(kind="pii", label=c.label, start=c.start, end=c.end, detector=c.detector)
            for c in _resolve_overlaps(candidates)
        ]


def _resolve_overlaps(candidates: List[_Candidate]) -> List[_Candidate]:
    """Greedily keep non-overlapping candidates, preferring longer, stronger spans."""
    chosen: List[_Candidate] = []
    for cand in sorted(
        candidates, key=lambda c: (-(c.end - c.start), -c.score, c.start, c.label)
    ):
        if any(cand.start < k.end and k.start < cand.end for k in chosen):
            continue
        chosen.append(cand)
    chosen.sort(key=lambda c: c.start)
    return chosen


# Process-wide singleton so the spaCy model is loaded at most once.
_default_detector: Optional[PiiDetector] = None


def detect(text: str) -> List[Finding]:
    """Convenience wrapper detecting PII with a shared, lazily-built detector."""
    global _default_detector
    if _default_detector is None:
        _default_detector = PiiDetector()
    return _default_detector.detect(text)
