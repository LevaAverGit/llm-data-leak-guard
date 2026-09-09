"""Benchmark entrypoint used by ``make bench`` (``python -m bench.run``).

Thin wrapper so the Makefile target and the module both resolve to the same
implementation in :mod:`bench.latency`.
"""

from __future__ import annotations

from bench.latency import main

if __name__ == "__main__":
    raise SystemExit(main())
