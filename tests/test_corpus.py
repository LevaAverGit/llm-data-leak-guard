"""Corpus and schema contract integrity.

These tests need no detectors, no network and no external models — they validate
the on-disk contract itself: that every labeled span in the synthetic corpus is
self-consistent, lines up with the :class:`guard.schema.Finding` model, and that
the data-safety invariants (everything synthetic; secret placeholders are
obviously invalid) actually hold. They run in every environment.
"""

from __future__ import annotations

import pytest

from guard import Finding, GuardResult, mask_for
from guard.schema import FindingKind

from conftest import (
    CLASS_IDS,
    CLASSES,
    load_class_file,
    load_index,
    load_mixed_example,
)

_VALID_KINDS = set(getattr(FindingKind, "__args__", ("pii", "secret", "requisite")))
_SECRET_LABELS = {"API_KEY", "PRIVATE_KEY", "BEARER_TOKEN"}


def _all_class_files():
    """Yield (descriptor, parsed-file) for every class listed in index.json."""
    for descriptor in CLASSES:
        yield descriptor, load_class_file(descriptor["file"])


# --------------------------------------------------------------------------- #
# Index coverage
# --------------------------------------------------------------------------- #
def test_index_marks_corpus_synthetic():
    assert load_index()["synthetic"] is True


def test_index_covers_ten_leak_classes():
    assert len(CLASSES) == 10
    # Every class advertises the four fields the tests rely on.
    for descriptor in CLASSES:
        assert set(descriptor) >= {"leak_class", "kind", "label", "file"}


@pytest.mark.parametrize("descriptor", CLASSES, ids=CLASS_IDS)
def test_class_file_header_matches_index(descriptor):
    data = load_class_file(descriptor["file"])
    assert data["synthetic"] is True
    assert data["leak_class"] == descriptor["leak_class"]
    assert data["kind"] == descriptor["kind"]
    assert data["label"] == descriptor["label"]
    assert data["kind"] in _VALID_KINDS
    assert data["examples"], "each class file must carry at least one example"


# --------------------------------------------------------------------------- #
# Span invariants — the core contract text[start:end] == span.text
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("descriptor", CLASSES, ids=CLASS_IDS)
def test_spans_are_self_consistent(descriptor):
    data = load_class_file(descriptor["file"])
    for example in data["examples"]:
        text = example["text"]
        for span in example["spans"]:
            start, end = span["start"], span["end"]
            # Half-open Python offsets that actually slice out the labeled value.
            assert 0 <= start < end <= len(text)
            assert text[start:end] == span["text"], (
                f"{example['id']}: text[{start}:{end}] != span text"
            )
            # The span's own label/kind match the file-level class contract.
            assert span["label"] == data["label"]
            assert span["kind"] == data["kind"]


def test_mixed_example_spans_are_self_consistent():
    example = load_mixed_example()
    text = example["text"]
    labels = {span["label"] for span in example["spans"]}
    # The README before/after demo relies on this exact multi-leak mix.
    assert labels == {"PERSON", "PHONE", "CONTRACT", "API_KEY"}
    for span in example["spans"]:
        assert text[span["start"] : span["end"]] == span["text"]
        assert span["kind"] in _VALID_KINDS


# --------------------------------------------------------------------------- #
# Every labeled span is a constructible Finding (schema alignment)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("descriptor", CLASSES, ids=CLASS_IDS)
def test_every_span_builds_a_valid_finding(descriptor):
    data = load_class_file(descriptor["file"])
    for example in data["examples"]:
        for span in example["spans"]:
            finding = Finding(
                kind=span["kind"],
                label=span["label"],
                start=span["start"],
                end=span["end"],
                detector="corpus",
            )
            assert finding.length == span["end"] - span["start"]
            assert finding.mask == mask_for(span["label"]) == f"[{span['label']}]"
            assert example["text"][finding.start : finding.end] == span["text"]


# --------------------------------------------------------------------------- #
# Data-safety invariants (mirrors the DATA-SAFETY GATE)
# --------------------------------------------------------------------------- #
def test_secret_placeholders_are_obviously_invalid():
    """Every secret-class value must be a synthetic EXAMPLE placeholder."""
    for descriptor, data in _all_class_files():
        if data["kind"] != "secret":
            continue
        for example in data["examples"]:
            assert "EXAMPLE" in example["text"], (
                f"{example['id']}: secret example must contain the EXAMPLE marker"
            )
            for span in example["spans"]:
                assert span["label"] in _SECRET_LABELS
                assert "EXAMPLE" in span["text"], (
                    f"{example['id']}: secret span must be an EXAMPLE placeholder"
                )


def test_emails_use_reserved_example_domains():
    data = load_class_file("email.json")
    for example in data["examples"]:
        for span in example["spans"]:
            assert "@example." in span["text"], "emails must stay on reserved example.* domains"


def test_all_corpus_files_declare_synthetic():
    for _descriptor, data in _all_class_files():
        assert data.get("synthetic") is True
    assert load_mixed_example()  # mixed file present and non-empty


# --------------------------------------------------------------------------- #
# Schema micro-contracts used by every downstream component
# --------------------------------------------------------------------------- #
def test_finding_rejects_non_positive_span():
    with pytest.raises(Exception):
        Finding(kind="pii", label="PHONE", start=10, end=10, detector="x")
    with pytest.raises(Exception):
        Finding(kind="pii", label="PHONE", start=10, end=5, detector="x")


def test_finding_forbids_extra_fields():
    with pytest.raises(Exception):
        Finding(kind="pii", label="PHONE", start=0, end=3, detector="x", note="nope")


def test_finding_overlap_detection():
    a = Finding(kind="pii", label="PHONE", start=0, end=5, detector="x")
    b = Finding(kind="pii", label="PHONE", start=4, end=9, detector="x")
    c = Finding(kind="pii", label="PHONE", start=5, end=9, detector="x")
    assert a.overlaps(b) and b.overlaps(a)
    assert not a.overlaps(c)  # half-open: [0,5) and [5,9) touch but do not overlap


def test_guardresult_clean_and_count():
    clean = GuardResult(original_len=11, redacted="hello world", findings=[], latency_ms=0.0)
    assert clean.is_clean and clean.count == 0

    finding = Finding(kind="secret", label="API_KEY", start=0, end=3, detector="x")
    dirty = GuardResult(original_len=3, redacted="[API_KEY]", findings=[finding], latency_ms=1.0)
    assert not dirty.is_clean and dirty.count == 1
