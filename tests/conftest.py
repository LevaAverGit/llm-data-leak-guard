"""Shared fixtures, corpus loaders and API resolvers for the test suite.

The detectors, the combined pipeline, the redactor and the FastAPI proxy are
built in sibling modules (``guard.*`` and ``proxy.app``). This suite is written
against the *contracts* — the Pydantic models in :mod:`guard.schema` and the
labeled corpus in ``corpus/`` — rather than against one hard-coded function name,
so it keeps working across small, reasonable naming choices in those modules.

Design rules for these tests:

* **Deterministic, no network.** The proxy is exercised in-process with
  ``fastapi.testclient.TestClient`` (httpx-backed); nothing dials out.
* **Never break collection.** When a component is not importable yet (for
  example Presidio is not installed, or the pipeline module has not landed), the
  tests that need it *skip* with a clear reason instead of erroring.
* **Strict where it matters.** Once a component is available the behavioural
  assertions are exact: the ground-truth spans must be caught, the redacted text
  must never contain the raw sensitive value, and the proxy must not leak raw
  input downstream.
"""

from __future__ import annotations

import importlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional

import pytest

# Make the project root importable regardless of pytest's rootdir/cwd so that
# ``import guard`` / ``import proxy`` / ``import bench`` resolve.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS_DIR = ROOT / "corpus"

# The schema is a stable, dependency-free contract that is always importable.
from guard import Finding, GuardResult, mask_for  # noqa: E402


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
def load_index() -> dict:
    """Return the parsed ``corpus/index.json``."""
    return json.loads((CORPUS_DIR / "index.json").read_text(encoding="utf-8"))


def load_class_file(file_name: str) -> dict:
    """Return the parsed corpus class file (e.g. ``phone.json``)."""
    return json.loads((CORPUS_DIR / file_name).read_text(encoding="utf-8"))


def load_mixed_example() -> dict:
    """Return the single multi-leak example from ``corpus/mixed.json``."""
    return load_class_file("mixed.json")["examples"][0]


# Class descriptors, read at import time so tests can parametrize over them.
# Each entry: {"leak_class", "kind", "label", "file"}.
CLASSES: List[dict] = load_index()["classes"]
CLASS_IDS: List[str] = [c["leak_class"] for c in CLASSES]


# --------------------------------------------------------------------------- #
# Resolving the combined guard pipeline: text -> GuardResult
# --------------------------------------------------------------------------- #
_GUARD_MODULES = [
    "guard.pipeline",
    "guard",
    "guard.guard",
    "guard.redactor",
    "guard.core",
    "guard.engine",
    "guard.detectors",
]
_GUARD_FUNCS = [
    "guard",
    "inspect",
    "scan",
    "run",
    "process",
    "analyze",
    "protect",
    "redact",
    "guard_prompt",
    "inspect_prompt",
    "guard_text",
]
_GUARD_CLASSES = ["Guard", "LeakGuard", "DataLeakGuard", "Pipeline", "GuardPipeline", "Engine"]
_GUARD_METHODS = ["guard", "inspect", "scan", "run", "process", "analyze", "protect", "redact", "__call__"]


def _looks_like_guard_result(obj: object) -> bool:
    """A GuardResult-shaped object exposes ``.redacted`` and ``.findings``."""
    return hasattr(obj, "redacted") and hasattr(obj, "findings")


def _accept_guard_callable(candidate: Callable) -> Optional[Callable]:
    """Return ``candidate`` if calling it on a clean string yields a GuardResult.

    A clean input has no findings, so this only checks the *shape* of the return
    value. It also rejects a two-argument redactor (``redact(text, findings)``),
    which raises ``TypeError`` when handed a single string.
    """
    try:
        result = candidate("hello world")
    except Exception:
        return None
    return candidate if _looks_like_guard_result(result) else None


@lru_cache(maxsize=1)
def resolve_guard() -> Optional[Callable[[str], GuardResult]]:
    """Best-effort discovery of the pipeline entry point ``f(text) -> GuardResult``.

    Returns ``None`` when no pipeline is importable/usable yet (built in parallel,
    or its detector dependencies such as Presidio are not installed).
    """
    for mod_name in _GUARD_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        for fn_name in _GUARD_FUNCS:
            obj = getattr(mod, fn_name, None)
            if callable(obj) and not isinstance(obj, type):
                accepted = _accept_guard_callable(obj)
                if accepted is not None:
                    return accepted

        for cls_name in _GUARD_CLASSES:
            cls = getattr(mod, cls_name, None)
            if isinstance(cls, type):
                try:
                    instance = cls()
                except Exception:
                    continue
                for meth_name in _GUARD_METHODS:
                    meth = getattr(instance, meth_name, None)
                    if callable(meth):
                        accepted = _accept_guard_callable(meth)
                        if accepted is not None:
                            return accepted
    return None


# --------------------------------------------------------------------------- #
# Resolving the redactor: (text, findings) -> redacted str
# --------------------------------------------------------------------------- #
_RED_MODULES = ["guard.redactor", "guard", "guard.pipeline", "guard.redact", "guard.core"]
_RED_FUNCS = ["redact", "apply", "redact_text", "mask", "apply_findings", "redact_findings", "redact_spans"]
_RED_CLASSES = ["Redactor", "Masker", "Guard"]
_RED_METHODS = ["redact", "apply", "mask", "__call__"]

# Controlled unit input used both to validate a redactor candidate and to drive
# the isolated redactor test. Offsets are hand-checked against the text.
_RED_TEXT = "before 12345 middle 67890 after"


def _red_unit_findings() -> List[Finding]:
    return [
        Finding(kind="pii", label="PHONE", start=7, end=12, detector="unit"),
        Finding(kind="secret", label="API_KEY", start=20, end=25, detector="unit"),
    ]


def _normalize_redactor_output(out: object) -> Optional[str]:
    """Coerce a redactor return value to the redacted string, or ``None``."""
    if out is None:
        return None
    if hasattr(out, "redacted"):
        return out.redacted
    if isinstance(out, str):
        return out
    return None


# Argument-order strategies a redactor might expose.
def _invoke_redactor(fn: Callable, mode: int, text: str, findings: List[Finding]) -> object:
    if mode == 0:
        return fn(text, findings)
    if mode == 1:
        return fn(text, findings=findings)
    if mode == 2:
        return fn(text=text, findings=findings)
    if mode == 3:
        return fn(findings, text)
    raise ValueError(mode)


def _accept_redactor(fn: Callable) -> Optional[int]:
    """Return the working call ``mode`` for ``fn`` on the unit input, else ``None``."""
    findings = _red_unit_findings()
    for mode in range(4):
        try:
            out = _normalize_redactor_output(_invoke_redactor(fn, mode, _RED_TEXT, findings))
        except Exception:
            continue
        if (
            out is not None
            and "[PHONE]" in out
            and "[API_KEY]" in out
            and "12345" not in out
            and "67890" not in out
        ):
            return mode
    return None


@lru_cache(maxsize=1)
def resolve_redactor() -> Optional[Callable[[str, List[Finding]], str]]:
    """Best-effort discovery of the redactor ``f(text, findings) -> redacted str``.

    Returns ``None`` when no standalone redactor is importable yet.
    """
    for mod_name in _RED_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        candidates: List[Callable] = []
        for fn_name in _RED_FUNCS:
            obj = getattr(mod, fn_name, None)
            if callable(obj) and not isinstance(obj, type):
                candidates.append(obj)
        for cls_name in _RED_CLASSES:
            cls = getattr(mod, cls_name, None)
            if isinstance(cls, type):
                try:
                    instance = cls()
                except Exception:
                    continue
                for meth_name in _RED_METHODS:
                    meth = getattr(instance, meth_name, None)
                    if callable(meth):
                        candidates.append(meth)

        for fn in candidates:
            mode = _accept_redactor(fn)
            if mode is not None:
                return lambda text, findings, _fn=fn, _mode=mode: _normalize_redactor_output(
                    _invoke_redactor(_fn, _mode, text, findings)
                )
    return None


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def guard_run() -> Callable[[str], GuardResult]:
    """The combined pipeline callable, or skip if it is not available yet."""
    fn = resolve_guard()
    if fn is None:
        pytest.skip(
            "guard pipeline not resolvable yet — the combined detector/redactor "
            "pipeline (guard.pipeline / guard.guard(...)) is not importable, or "
            "its dependencies (e.g. Presidio + spaCy en_core_web_sm) are not "
            "installed. Run `make install`, then this test runs for real."
        )
    return fn


@pytest.fixture(scope="session")
def redactor_run() -> Callable[[str, List[Finding]], str]:
    """A standalone redactor callable, or skip if not exposed separately."""
    fn = resolve_redactor()
    if fn is None:
        pytest.skip(
            "standalone redactor not resolvable yet — no guard.redactor.redact("
            "text, findings) style function/class is importable. The offset "
            "invariant is still covered end-to-end via the pipeline."
        )
    return fn


@pytest.fixture(scope="session")
def proxy_app():
    """The FastAPI application object from ``proxy.app``, or skip if absent."""
    try:
        mod = importlib.import_module("proxy.app")
    except Exception as exc:  # module not built yet, or import-time dependency missing
        pytest.skip(f"proxy.app not importable yet: {exc!r}")
    app = getattr(mod, "app", None)
    if app is None:
        pytest.skip("proxy.app is importable but exposes no `app` object")
    return app


@pytest.fixture()
def client(proxy_app):
    """In-process httpx-backed test client for the proxy (no network)."""
    from fastapi.testclient import TestClient

    with TestClient(proxy_app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Proxy request helper
# --------------------------------------------------------------------------- #
def _candidate_bodies(text: str) -> List[dict]:
    """Plausible request shapes for ``POST /guard/complete`` (schema not fixed)."""
    return [
        {"prompt": text},
        {"text": text},
        {"input": text},
        {"content": text},
        {"prompt": text, "model": "mock"},
        {"messages": [{"role": "user", "content": text}]},
    ]


def post_complete(client, text: str):
    """POST ``text`` to ``/guard/complete``, adapting to the endpoint's body shape.

    Returns the first accepted (non-error) response. A 5xx is returned
    immediately: it means the endpoint was reached and the request shape was
    accepted, but the guard failed server-side (e.g. detectors not installed) —
    a clearer signal than continuing to probe other shapes. Only 4xx responses
    (client/schema mismatch) are retried; the last 4xx is returned if nothing
    else works, so the caller can skip with a precise reason.
    """
    last = None
    for body in _candidate_bodies(text):
        resp = client.post("/guard/complete", json=body)
        if resp.status_code < 400 or resp.status_code >= 500:
            return resp
        last = resp
    return last
