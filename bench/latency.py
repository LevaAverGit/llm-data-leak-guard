"""Measure guard redaction latency over the synthetic corpus.

Run with::

    python -m bench.run        # used by `make bench`
    python -m bench.latency    # equivalent
    python -m bench.latency --repeats 50 --json

What it reports:

* Overall latency across every corpus example: mean, p50, p95, p99, min, max
  (milliseconds), timed with a wall-clock ``perf_counter`` around each guard
  call after a warm-up pass (the first Presidio/spaCy call is much slower and is
  excluded).
* Per-class latency and detection coverage — how many ground-truth spans the
  guard actually caught, i.e. "caught N of N".
* A README-ready headline line for the mixed before/after prompt.

Only redacted text is printed. The corpus is fully synthetic (see
``corpus/README.md``), so no real sensitive data is involved at any point.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from guard import GuardResult
from guard.redactor import guard  # the combined detector + redactor pipeline

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
DEFAULT_REPEATS = 25


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
class Example:
    """One corpus example: the text and its ground-truth spans."""

    __slots__ = ("leak_class", "example_id", "text", "spans")

    def __init__(self, leak_class: str, example_id: str, text: str, spans: List[dict]):
        self.leak_class = leak_class
        self.example_id = example_id
        self.text = text
        self.spans = spans


def _load_class_file(path: Path) -> List[Example]:
    data = json.loads(path.read_text(encoding="utf-8"))
    leak_class = data.get("leak_class", path.stem)
    out: List[Example] = []
    for ex in data.get("examples", []):
        out.append(Example(leak_class, ex["id"], ex["text"], ex.get("spans", [])))
    return out


def load_examples() -> List[Example]:
    """Load every class listed in ``index.json`` (the mixed prompt is separate)."""
    index = json.loads((CORPUS_DIR / "index.json").read_text(encoding="utf-8"))
    examples: List[Example] = []
    for entry in index["classes"]:
        examples.extend(_load_class_file(CORPUS_DIR / entry["file"]))
    return examples


def load_mixed() -> Optional[Example]:
    """Load the single mixed before/after prompt, if present."""
    path = CORPUS_DIR / "mixed.json"
    if not path.exists():
        return None
    ex = _load_class_file(path)
    return ex[0] if ex else None


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile (``pct`` in [0, 100]); 0.0 if empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _span_caught(span: dict, result: GuardResult) -> bool:
    """A ground-truth span is caught if any finding's span overlaps it."""
    s, e = span["start"], span["end"]
    for f in result.findings:
        if f.start < e and s < f.end:
            return True
    return False


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def time_call(guard_fn: Callable[[str], GuardResult], text: str) -> Tuple[float, GuardResult]:
    """Return (wall-clock latency in ms, result) for one guard call."""
    start = time.perf_counter()
    result = guard_fn(text)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, result


def run_benchmark(repeats: int = DEFAULT_REPEATS) -> dict:
    """Run the full latency + coverage benchmark and return a summary dict."""
    guard_fn = guard
    examples = load_examples()
    if not examples:
        raise RuntimeError(f"No corpus examples found under {CORPUS_DIR}.")

    # Warm-up: the first guard call loads spaCy/Presidio and is not representative.
    guard_fn(examples[0].text)

    # Coverage pass: one call per example, record findings vs ground truth.
    per_class: Dict[str, dict] = {}
    for ex in examples:
        _, result = time_call(guard_fn, ex.text)
        bucket = per_class.setdefault(
            ex.leak_class,
            {"examples": 0, "spans": 0, "caught": 0, "latencies": []},
        )
        bucket["examples"] += 1
        bucket["spans"] += len(ex.spans)
        bucket["caught"] += sum(1 for span in ex.spans if _span_caught(span, result))

    # Timing pass: repeat every example, collect per-call latencies.
    all_latencies: List[float] = []
    for _ in range(max(1, repeats)):
        for ex in examples:
            elapsed_ms, _ = time_call(guard_fn, ex.text)
            all_latencies.append(elapsed_ms)
            per_class[ex.leak_class]["latencies"].append(elapsed_ms)

    total_spans = sum(b["spans"] for b in per_class.values())
    total_caught = sum(b["caught"] for b in per_class.values())

    summary = {
        "repeats": max(1, repeats),
        "examples": len(examples),
        "calls_timed": len(all_latencies),
        "latency_ms": {
            "mean": _mean(all_latencies),
            "p50": percentile(all_latencies, 50),
            "p95": percentile(all_latencies, 95),
            "p99": percentile(all_latencies, 99),
            "min": min(all_latencies) if all_latencies else 0.0,
            "max": max(all_latencies) if all_latencies else 0.0,
        },
        "coverage": {
            "spans": total_spans,
            "caught": total_caught,
            "ratio": (total_caught / total_spans) if total_spans else 0.0,
        },
        "per_class": {
            name: {
                "examples": b["examples"],
                "spans": b["spans"],
                "caught": b["caught"],
                "coverage": (b["caught"] / b["spans"]) if b["spans"] else 0.0,
                "mean_ms": _mean(b["latencies"]),
                "p95_ms": percentile(b["latencies"], 95),
            }
            for name, b in per_class.items()
        },
    }

    # Mixed before/after headline (README metric).
    mixed = load_mixed()
    if mixed is not None:
        mixed_latencies: List[float] = []
        mixed_result: Optional[GuardResult] = None
        for _ in range(max(1, repeats)):
            elapsed_ms, mixed_result = time_call(guard_fn, mixed.text)
            mixed_latencies.append(elapsed_ms)
        assert mixed_result is not None
        caught = sum(1 for span in mixed.spans if _span_caught(span, mixed_result))
        summary["mixed"] = {
            "spans": len(mixed.spans),
            "caught": caught,
            "median_ms": percentile(mixed_latencies, 50),
            "redacted": mixed_result.redacted,
        }

    return summary


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_report(summary: dict) -> None:
    lat = summary["latency_ms"]
    cov = summary["coverage"]

    print("llm-data-leak-guard — redaction benchmark")
    print("=" * 56)
    print(
        f"corpus examples: {summary['examples']}   "
        f"repeats: {summary['repeats']}   "
        f"timed calls: {summary['calls_timed']}"
    )
    print()
    print("latency per guard call (ms):")
    print(
        f"  mean {lat['mean']:7.2f}   p50 {lat['p50']:7.2f}   "
        f"p95 {lat['p95']:7.2f}   p99 {lat['p99']:7.2f}"
    )
    print(f"  min  {lat['min']:7.2f}   max {lat['max']:7.2f}")
    print()
    print(
        f"detection coverage: caught {cov['caught']} of {cov['spans']} "
        f"ground-truth spans ({cov['ratio'] * 100:.1f}%)"
    )
    print()

    print("per class:")
    header = f"  {'class':<16} {'ex':>3} {'caught/spans':>13} {'cov%':>6} {'mean ms':>9} {'p95 ms':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in sorted(summary["per_class"]):
        c = summary["per_class"][name]
        print(
            f"  {name:<16} {c['examples']:>3} "
            f"{str(c['caught']) + '/' + str(c['spans']):>13} "
            f"{c['coverage'] * 100:>5.0f}% {c['mean_ms']:>9.2f} {c['p95_ms']:>9.2f}"
        )

    mixed = summary.get("mixed")
    if mixed is not None:
        print()
        print("mixed before/after prompt (README headline):")
        print(
            f"  caught: {mixed['caught']} of {mixed['spans']}  ·  "
            f"latency: {mixed['median_ms']:.0f} ms (median)"
        )
        print(f"  redacted: {mixed['redacted']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.latency",
        description="Measure guard redaction latency over the synthetic corpus.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Timing repetitions per example (default: {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the summary as JSON instead of a formatted report.",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_benchmark(repeats=args.repeats)
    except Exception as exc:  # noqa: BLE001 - report any setup/detector failure cleanly
        print(f"benchmark error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Hint: run `make install` so the detectors "
            "(Presidio/spaCy) and the redactor are available.",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        _print_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
