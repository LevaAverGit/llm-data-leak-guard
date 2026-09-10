"""FastAPI guard proxy — ``proxy.app:app``.

Run locally with::

    python -m uvicorn proxy.app:app --host 127.0.0.1 --port 8000

Endpoint of record: ``POST /guard/complete``. It

1. receives a prompt,
2. runs the guard over it,
3. forwards **only the redacted prompt** to a mock model, and
4. returns the ``GuardResult`` plus the mock completion.

Safety invariants enforced here:

* The raw prompt is used only inside the guard call. It is **never logged** and
  **never forwarded** — the mock model only ever sees ``result.redacted``.
* The response carries the ``GuardResult`` (redacted text, findings, latency).
  ``Finding`` objects hold only span offsets and class labels (``PHONE``,
  ``API_KEY``, ...), never the raw sensitive substring, so the response is safe
  to return and to log.
"""

from __future__ import annotations

import importlib
import inspect as _inspect
import logging
import os
from typing import Callable, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from guard import GuardResult, __version__ as guard_version

logger = logging.getLogger("guard.proxy")

# Name of the mock model reported in responses. There is no real LLM behind the
# proxy on purpose (see PLAN.md): the guard is deterministic and the demo stays
# self-contained and testable.
MOCK_MODEL_NAME = "mock-echo-001"


# --------------------------------------------------------------------------- #
# Guard pipeline resolution
# --------------------------------------------------------------------------- #
# The combined guard pipeline (detectors + redactor) lives in the ``guard``
# package (currently ``guard.redactor.guard``). To stay decoupled from its exact
# public name we resolve it dynamically the first time it is needed, trying the
# idiomatic entrypoints in priority order and validating that a candidate really
# takes a single text argument (so a two-argument helper such as
# ``redactor.redact(text, findings)`` is not mistaken for the pipeline). An
# explicit override is honoured via ``GUARD_ENTRYPOINT="module:attr"``.
_GUARD_MODULES = ("guard", "guard.pipeline", "guard.redactor")
_GUARD_NAMES = (
    "inspect",
    "inspect_prompt",
    "run_guard",
    "guard",
    "guard_prompt",
    "run",
    "protect",
    "scan",
    "redact",
)
# Method names tried when a resolved entrypoint turns out to be a class.
_METHOD_CANDIDATES = ("inspect", "run", "run_guard", "guard", "scan", "redact", "__call__")

_guard_callable: Optional[Callable[[str], GuardResult]] = None


def _accepts_single_text_arg(fn: Callable) -> bool:
    """True if ``fn`` can be called as ``fn(text)`` (<=1 required positional)."""
    try:
        sig = _inspect.signature(fn)
    except (TypeError, ValueError):
        return True  # builtins / C callables: assume compatible
    required_positional = 0
    accepts_positional = False
    for param in sig.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            accepts_positional = True
            if param.default is param.empty:
                required_positional += 1
        elif param.kind is param.VAR_POSITIONAL:
            return True
    return accepts_positional and required_positional <= 1


def _as_guard_callable(obj: object) -> Optional[Callable[[str], GuardResult]]:
    """Turn a resolved object into a ``(text) -> GuardResult`` callable.

    Accepts a plain function, a bound method, or a class (instantiated with no
    arguments and then probed for a suitable inspection method). The chosen
    callable must accept a single text argument.
    """
    if obj is None:
        return None
    if isinstance(obj, type):
        try:
            instance = obj()
        except Exception:
            return None
        for method in _METHOD_CANDIDATES:
            candidate = getattr(instance, method, None)
            if callable(candidate) and _accepts_single_text_arg(candidate):
                return candidate  # type: ignore[return-value]
        return None
    if callable(obj) and _accepts_single_text_arg(obj):
        return obj  # type: ignore[return-value]
    return None


def _candidate_pairs() -> List[tuple]:
    pairs: List[tuple] = []
    override = os.environ.get("GUARD_ENTRYPOINT", "").strip()
    if override and ":" in override:
        mod_name, attr_name = override.split(":", 1)
        pairs.append((mod_name.strip(), attr_name.strip()))
    for mod_name in _GUARD_MODULES:
        for attr_name in _GUARD_NAMES:
            pairs.append((mod_name, attr_name))
    return pairs


def _resolve_guard() -> Optional[Callable[[str], GuardResult]]:
    """Locate and cache the guard pipeline callable, or return ``None``."""
    global _guard_callable
    if _guard_callable is not None:
        return _guard_callable

    for mod_name, attr_name in _candidate_pairs():
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            continue
        fn = _as_guard_callable(getattr(module, attr_name, None))
        if fn is not None:
            _guard_callable = fn
            logger.info("guard pipeline resolved: %s:%s", mod_name, attr_name)
            return _guard_callable
    return None


def _guard_ready() -> bool:
    """Honest readiness signal: True only if the pipeline actually runs.

    Resolving the callable is not enough — a resolved guard can still raise at
    call time (e.g. a lazily loaded detector's dependency is missing). This runs
    the resolved pipeline on a constant, non-sensitive probe string and reports
    ``False`` if resolution or that run fails, so a health/probe endpoint does
    not advertise readiness for a pipeline that would 500. The probe text is a
    fixed literal, never user input, so nothing sensitive is involved.
    """
    guard_fn = _resolve_guard()
    if guard_fn is None:
        return False
    try:
        guard_fn("ping")
        return True
    except Exception:
        logger.warning("guard readiness probe failed")
        return False


def run_guard(prompt: str) -> GuardResult:
    """Run the guard over ``prompt`` and return a validated ``GuardResult``.

    Raises ``HTTPException(503)`` if the guard pipeline is not available yet
    (e.g. detectors not installed), and ``HTTPException(500)`` if the pipeline
    raises. Neither error path echoes the raw prompt.
    """
    guard_fn = _resolve_guard()
    if guard_fn is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Guard pipeline is not available. Install dependencies "
                "(`make install`) so the detectors and redactor can load."
            ),
        )
    try:
        result = guard_fn(prompt)
    except Exception:
        # Log the failure without any prompt content.
        logger.exception("guard pipeline raised while inspecting a prompt")
        raise HTTPException(status_code=500, detail="Guard inspection failed.")

    if not isinstance(result, GuardResult):
        # Be strict: the pipeline must honour the shared contract.
        try:
            result = GuardResult.model_validate(result)
        except Exception:
            logger.error("guard pipeline returned a non-GuardResult value")
            raise HTTPException(status_code=500, detail="Guard returned an invalid result.")
    return result


# --------------------------------------------------------------------------- #
# Optional Prometheus metrics
# --------------------------------------------------------------------------- #
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )

    _METRICS_ENABLED = True
    _REQUESTS = Counter(
        "guard_requests_total", "Total /guard/complete requests handled."
    )
    _FINDINGS = Counter(
        "guard_findings_total", "Total leaks caught, by label.", ["label"]
    )
    _LATENCY = Histogram(
        "guard_latency_ms",
        "Guard inspection latency in milliseconds.",
        buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
    )
except Exception:  # pragma: no cover - prometheus_client is optional at runtime
    _METRICS_ENABLED = False


# --------------------------------------------------------------------------- #
# Mock model
# --------------------------------------------------------------------------- #
def mock_complete(redacted_prompt: str, model: str) -> str:
    """Deterministic stand-in for a real model call.

    It receives only the already-redacted prompt, demonstrating that no
    sensitive content is forwarded downstream. No network call is made.
    """
    return (
        f"[{model}] mock completion. The guard forwarded a redacted prompt of "
        f"{len(redacted_prompt)} characters; every sensitive span was replaced "
        f"with a [LABEL] mask before this model was called. "
        f"Redacted prompt seen by the model: {redacted_prompt!r}"
    )


# --------------------------------------------------------------------------- #
# Request / response contracts
# --------------------------------------------------------------------------- #
class CompleteRequest(BaseModel):
    """Body for ``POST /guard/complete``."""

    model_config = ConfigDict(extra="forbid")

    # Upper bound caps detector/NER work per request and bounds the blast radius
    # of any pathological input; 20k chars comfortably fits real prompts.
    prompt: str = Field(..., min_length=1, max_length=20_000, description="Outgoing prompt to inspect.")
    model: Optional[str] = Field(
        default=None, description="Optional mock model name to attribute the completion to."
    )


class CompleteResponse(BaseModel):
    """Response for ``POST /guard/complete``.

    Carries the guard result (redacted text + findings + latency) and the mock
    completion. Nothing in this payload contains raw sensitive input.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Mock model the redacted prompt was sent to.")
    completion: str = Field(..., description="Mock model output (generated from redacted text only).")
    caught: int = Field(..., ge=0, description="Number of leaks the guard caught.")
    guard: GuardResult = Field(..., description="Full guard inspection result.")


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="llm-data-leak-guard proxy",
    version=guard_version,
    summary="Inspect → redact → forward: strips PII/secrets/requisites before a prompt reaches the model.",
)


@app.get("/")
def root() -> dict:
    """Service metadata and available endpoints."""
    return {
        "service": "llm-data-leak-guard",
        "version": guard_version,
        "guard_ready": _guard_ready(),
        "mock_model": MOCK_MODEL_NAME,
        "metrics_enabled": _METRICS_ENABLED,
        "endpoints": {
            "complete": "POST /guard/complete",
            "health": "GET /health",
            "metrics": "GET /metrics" if _METRICS_ENABLED else None,
        },
    }


@app.get("/health")
def health() -> dict:
    """Liveness / readiness probe.

    ``guard_ready`` is true only when the guard pipeline actually runs on a
    constant probe string — not merely resolves — so it never advertises a
    pipeline that would 500 at call time.
    """
    return {"status": "ok", "guard_ready": _guard_ready()}


@app.post("/guard/complete", response_model=CompleteResponse)
def guard_complete(req: CompleteRequest) -> CompleteResponse:
    """Inspect a prompt, redact it, and return the guard result + mock completion.

    The raw ``req.prompt`` is passed only to the guard. It is never logged and
    never handed to the model — only ``result.redacted`` crosses that boundary.
    """
    result = run_guard(req.prompt)

    model = req.model or MOCK_MODEL_NAME
    # Only the redacted text is forwarded to the model.
    completion = mock_complete(result.redacted, model)

    # Structured, non-sensitive telemetry: counts, class labels and latency
    # only. Labels (PHONE, API_KEY, ...) are class names, not sensitive values.
    labels = [f.label for f in result.findings]
    logger.info(
        "guard.complete original_len=%d caught=%d labels=%s latency_ms=%.2f",
        result.original_len,
        result.count,
        labels,
        result.latency_ms,
    )

    if _METRICS_ENABLED:
        _REQUESTS.inc()
        _LATENCY.observe(result.latency_ms)
        for label in labels:
            _FINDINGS.labels(label=label).inc()

    return CompleteResponse(
        model=model,
        completion=completion,
        caught=result.count,
        guard=result,
    )


if _METRICS_ENABLED:

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        """Prometheus exposition of guard counters and latency histogram."""
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
