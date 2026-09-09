"""Secret / credential detection for llm-data-leak-guard.

Regex signatures for well-known credential formats (provider API keys, PEM
private-key blocks, bearer / JWT tokens and credentials embedded in URLs) plus a
Shannon-entropy gate on the "loose" rules, mirroring the gitleaks approach:
fixed-prefix provider tokens are trusted on their signature alone, while generic
``keyword = value`` assignments and opaque bearer tokens must *also* look random
enough (high entropy) before they are reported. That split keeps recall high on
real credentials while cutting false positives on low-entropy filler such as
``password = changeme``.

Each match is returned as a :class:`guard.schema.Finding` with half-open offsets
into the original text (``text[start:end]`` is the raw secret), so the redactor
can splice ``[LABEL]`` masks in deterministically. This module never logs a raw
secret value -- only the offsets and label leave it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from math import log2
from typing import List, Optional

from guard.schema import Finding

__all__ = ["detect_secrets", "detect", "shannon_entropy", "ENTROPY_MIN"]

# Minimum Shannon entropy (bits per character) for a candidate produced by a
# "loose" rule to be reported. Fixed-prefix provider rules bypass this gate
# because their signature is already a strong enough signal on its own.
#
# 3.0 bits/char sits below the ~4.5-5.5 of base64/hex secrets but above common
# low-entropy filler: "changeme" ~= 2.75, so it is dropped, while a real random
# value clears the bar. This is the false-positive control the project headlines.
ENTROPY_MIN = 3.0


def shannon_entropy(value: str) -> float:
    """Shannon entropy of ``value`` in bits per character (``0.0`` for empty)."""
    if not value:
        return 0.0
    n = len(value)
    return -sum((count / n) * log2(count / n) for count in Counter(value).values())


@dataclass(frozen=True)
class _SecretRule:
    """One credential signature.

    ``group`` selects which regex group's span is masked (``0`` = whole match).
    ``entropy_min``, when set, drops a candidate whose captured value falls below
    that Shannon entropy -- the gitleaks-style gate on the loose rules.
    """

    name: str
    label: str
    pattern: "re.Pattern[str]"
    group: int = 0
    entropy_min: Optional[float] = None


# Order matters: more specific / fixed-prefix rules come first so that, on an
# exact-span tie, de-duplication keeps the more precise detector's finding.
_RULES: tuple[_SecretRule, ...] = (
    # PEM-armored private-key block (RSA/EC/OPENSSH/PGP/generic). Matched whole,
    # from the BEGIN line through the END line.
    _SecretRule(
        name="pem_private_key",
        label="PRIVATE_KEY",
        pattern=re.compile(
            r"-----BEGIN[A-Z0-9 ]*?PRIVATE KEY-----"
            r"[\s\S]*?"
            r"-----END[A-Z0-9 ]*?PRIVATE KEY-----"
        ),
    ),
    # OpenAI-style secret key: sk-... including project keys sk-proj-...
    _SecretRule(
        name="openai_api_key",
        label="API_KEY",
        pattern=re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"),
    ),
    # AWS access key id: AKIA + 16 upper-case letters / digits.
    _SecretRule(
        name="aws_access_key_id",
        label="API_KEY",
        pattern=re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![0-9A-Z])"),
    ),
    # GitHub tokens: classic ghp_/gho_/ghu_/ghs_/ghr_ and fine-grained github_pat_.
    _SecretRule(
        name="github_token",
        label="API_KEY",
        pattern=re.compile(
            r"(?<![A-Za-z0-9])(?:gh[opusr]_[A-Za-z0-9]{20,}"
            r"|github_pat_[A-Za-z0-9_]{20,})"
        ),
    ),
    # JWT / structured bearer token: eyJ<header>.<payload>.<signature>.
    _SecretRule(
        name="jwt_token",
        label="BEARER_TOKEN",
        pattern=re.compile(
            r"eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}"
        ),
    ),
    # Opaque bearer token after an "Authorization: Bearer" / "Bearer " prefix.
    # Entropy-gated: the token itself, not the "Bearer" prefix, is masked.
    _SecretRule(
        name="bearer_token",
        label="BEARER_TOKEN",
        pattern=re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/=-]{20,})"),
        group=1,
        entropy_min=ENTROPY_MIN,
    ),
    # Credentials embedded in a URL: scheme://user:PASSWORD@host -- only the
    # password portion is masked.
    _SecretRule(
        name="url_password",
        label="PASSWORD",
        pattern=re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s:/@]+:([^\s:/@]+)@"),
        group=1,
    ),
    # Generic assignment: secret/token/password/api_key = <high-entropy value>.
    # Entropy-gated to avoid flagging low-entropy filler like "changeme".
    _SecretRule(
        name="entropy_assignment",
        label="SECRET",
        pattern=re.compile(
            r"(?i)\b(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|"
            r"client[_-]?secret|auth[_-]?token|password|passwd|pwd|token)\b"
            r"\s*[:=]\s*['\"`]?([^\s'\"`]{12,})['\"`]?"
        ),
        group=1,
        entropy_min=ENTROPY_MIN,
    ),
)


def _dedupe(findings: List[Finding]) -> List[Finding]:
    """Drop overlapping findings, preferring the longest span.

    On an exact-span tie the earlier (more specific) rule wins, because a stable
    sort preserves the rule iteration order for equal keys. The result is sorted
    by start offset, ready for the redactor.
    """
    ordered = sorted(findings, key=lambda f: (f.start, -f.length))
    kept: List[Finding] = []
    for finding in ordered:
        if any(finding.overlaps(other) for other in kept):
            continue
        kept.append(finding)
    kept.sort(key=lambda f: f.start)
    return kept


def detect_secrets(text: str) -> List[Finding]:
    """Scan ``text`` for credentials and return de-duplicated ``Finding`` spans.

    Runs every signature, applies the Shannon-entropy gate on the loose rules,
    then removes overlaps so the redactor never masks the same characters twice.
    """
    if not text:
        return []

    findings: List[Finding] = []
    for rule in _RULES:
        for match in rule.pattern.finditer(text):
            start, end = match.span(rule.group)
            if start < 0 or end <= start:
                continue  # group did not participate in this match
            value = text[start:end]
            if rule.entropy_min is not None and shannon_entropy(value) < rule.entropy_min:
                continue
            findings.append(
                Finding(
                    kind="secret",
                    label=rule.label,
                    start=start,
                    end=end,
                    detector=rule.name,
                )
            )
    return _dedupe(findings)


# Convenience alias so the combined pipeline can import a uniform ``detect``.
detect = detect_secrets
