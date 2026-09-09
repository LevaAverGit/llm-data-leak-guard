"""Redaction correctness: offsets are preserved and the raw value never survives.

Two complementary checks:

* An *isolated* redactor test (when a standalone redactor is exposed) drives it
  with hand-placed findings and asserts the surrounding text is preserved exactly
  and each sensitive slice becomes its ``[LABEL]`` mask.
* An end-to-end *splice invariant* over the corpus: whatever the pipeline finds,
  the redacted string must equal the original with each (merged, non-overlapping)
  finding replaced by its mask — nothing else moved. This proves the redactor
  splices at the right offsets, and, together with the corpus, that no raw
  sensitive value is ever emitted.
"""

from __future__ import annotations

import pytest

from guard import Finding

from conftest import CLASS_IDS, CLASSES, load_class_file, load_mixed_example


def _splice(text: str, findings) -> str:
    """Rebuild the expected redacted string from non-overlapping findings."""
    parts = []
    cursor = 0
    for finding in sorted(findings, key=lambda f: (f.start, f.end)):
        parts.append(text[cursor : finding.start])
        parts.append(finding.mask)
        cursor = finding.end
    parts.append(text[cursor:])
    return "".join(parts)


def _assert_non_overlapping(findings, context: str) -> None:
    ordered = sorted(findings, key=lambda f: (f.start, f.end))
    for prev, nxt in zip(ordered, ordered[1:]):
        assert prev.end <= nxt.start, (
            f"{context}: findings overlap after merge — {prev.label}[{prev.start}:{prev.end}] "
            f"and {nxt.label}[{nxt.start}:{nxt.end}]"
        )


# --------------------------------------------------------------------------- #
# Isolated redactor (skips if the redactor is not exposed on its own)
# --------------------------------------------------------------------------- #
def test_redactor_preserves_surrounding_text(redactor_run):
    text = "before 12345 middle 67890 after"
    findings = [
        Finding(kind="pii", label="PHONE", start=7, end=12, detector="unit"),
        Finding(kind="secret", label="API_KEY", start=20, end=25, detector="unit"),
    ]
    redacted = redactor_run(text, findings)
    # Exact splice: non-sensitive spans are byte-for-byte preserved and in order.
    assert redacted == "before [PHONE] middle [API_KEY] after"
    # And the raw sensitive slices are gone.
    assert "12345" not in redacted
    assert "67890" not in redacted


def test_redactor_masks_a_single_span_at_its_offsets(redactor_run):
    text = "key sk-proj-EXAMPLE-not-real-000 rotate now"
    findings = [Finding(kind="secret", label="API_KEY", start=4, end=32, detector="unit")]
    assert text[4:32] == "sk-proj-EXAMPLE-not-real-000"
    redacted = redactor_run(text, findings)
    assert redacted == "key [API_KEY] rotate now"
    assert "sk-proj-EXAMPLE-not-real-000" not in redacted


# --------------------------------------------------------------------------- #
# End-to-end splice invariant over the corpus (needs the pipeline)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("descriptor", CLASSES, ids=CLASS_IDS)
def test_pipeline_redaction_is_an_exact_offset_splice(guard_run, descriptor):
    data = load_class_file(descriptor["file"])
    for example in data["examples"]:
        text = example["text"]
        result = guard_run(text)
        _assert_non_overlapping(result.findings, example["id"])
        # The redactor must reproduce exactly: original text with masks spliced in.
        assert result.redacted == _splice(text, result.findings), (
            f"{example['id']}: redacted text is not an exact offset splice of the findings"
        )


@pytest.mark.parametrize("descriptor", CLASSES, ids=CLASS_IDS)
def test_pipeline_never_emits_the_raw_sensitive_value(guard_run, descriptor):
    data = load_class_file(descriptor["file"])
    for example in data["examples"]:
        result = guard_run(example["text"])
        for span in example["spans"]:
            assert span["text"] not in result.redacted, (
                f"{example['id']}: raw {span['label']} value survived redaction"
            )


def test_mixed_prompt_redaction_leaks_nothing(guard_run):
    example = load_mixed_example()
    text = example["text"]
    result = guard_run(text)
    _assert_non_overlapping(result.findings, "mixed")
    assert result.redacted == _splice(text, result.findings)
    for span in example["spans"]:
        assert span["text"] not in result.redacted
