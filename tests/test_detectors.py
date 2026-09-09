"""One test per leak class: the ground-truth spans are caught and redacted.

Each corpus class file lists the spans a detector *must* find. For every example
we run the combined pipeline and assert that:

* every ground-truth span is covered by a finding carrying the expected label,
* the raw sensitive value no longer appears in the redacted text, and
* the uniform ``[LABEL]`` mask does appear in its place.

These tests skip cleanly until the pipeline (and its Presidio/spaCy backend) is
installed; once it is, they run for real and deterministically.
"""

from __future__ import annotations

import pytest

from guard import Finding, GuardResult

from conftest import CLASS_IDS, CLASSES, load_class_file, load_mixed_example


def _covering_finding(findings, start, end, label):
    """Return a finding with ``label`` whose span overlaps ``[start, end)``, else None."""
    for finding in findings:
        if finding.label == label and finding.start < end and start < finding.end:
            return finding
    return None


@pytest.mark.parametrize("descriptor", CLASSES, ids=CLASS_IDS)
def test_leak_class_is_caught_and_redacted(guard_run, descriptor):
    """For each leak class, every labeled span is detected and masked."""
    data = load_class_file(descriptor["file"])
    label = descriptor["label"]

    for example in data["examples"]:
        text = example["text"]
        result = guard_run(text)

        assert isinstance(result, GuardResult)
        assert result.original_len == len(text)

        for span in example["spans"]:
            match = _covering_finding(result.findings, span["start"], span["end"], label)
            assert match is not None, (
                f"{example['id']}: expected a {label} finding covering "
                f"[{span['start']}:{span['end']}] ({span['text']!r}), "
                f"got {[(f.label, f.start, f.end) for f in result.findings]}"
            )
            assert match.kind == descriptor["kind"]

            # The raw value must be gone and the mask must be present.
            assert span["text"] not in result.redacted, (
                f"{example['id']}: raw {label} value leaked into redacted text"
            )
            assert f"[{label}]" in result.redacted


def test_mixed_prompt_catches_every_class(guard_run):
    """The realistic multi-leak prompt loses all four sensitive spans."""
    example = load_mixed_example()
    text = example["text"]
    result = guard_run(text)

    assert isinstance(result, GuardResult)
    for span in example["spans"]:
        match = _covering_finding(result.findings, span["start"], span["end"], span["label"])
        assert match is not None, f"mixed: {span['label']} span not caught ({span['text']!r})"
        assert span["text"] not in result.redacted
        assert f"[{span['label']}]" in result.redacted


def test_pipeline_returns_contract_types(guard_run):
    """The pipeline honours the GuardResult / Finding contract exactly."""
    result = guard_run("Please call +7 900 123-45-67 about the invoice.")
    assert isinstance(result, GuardResult)
    assert isinstance(result.latency_ms, float) and result.latency_ms >= 0
    assert isinstance(result.findings, list)
    for finding in result.findings:
        assert isinstance(finding, Finding)
    # count / is_clean stay consistent with the findings list.
    assert result.count == len(result.findings)
    assert result.is_clean == (len(result.findings) == 0)


def test_clean_prompt_is_untouched(guard_run):
    """A prompt with nothing sensitive passes through verbatim."""
    text = "Please summarise the quarterly roadmap in three bullet points."
    result = guard_run(text)
    assert result.redacted == text
    assert result.is_clean
    assert result.count == 0
