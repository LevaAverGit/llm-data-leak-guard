"""llm-data-leak-guard — FastAPI proxy layer.

The proxy sits in front of a model. It runs the guard over an incoming prompt,
forwards **only** the redacted prompt to a (mock) model, and returns the
``GuardResult`` together with the mock completion. The raw prompt is never
logged and never forwarded — only masked text leaves this process.

The ASGI app lives in :mod:`proxy.app` (``proxy.app:app``). It is imported
lazily there rather than here so importing this package does not pull in
FastAPI or build the guard pipeline as a side effect.
"""
