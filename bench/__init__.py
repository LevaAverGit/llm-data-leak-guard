"""llm-data-leak-guard — benchmark layer.

Measures real guard redaction latency (mean / p95 / ...) over the synthetic
corpus and reports detection coverage ("caught N of N"). The numbers produced
here fill the headline metric in the project README.

Entrypoints: ``python -m bench.run`` (used by ``make bench``) and
``python -m bench.latency``; both call :func:`bench.latency.main`.
"""
