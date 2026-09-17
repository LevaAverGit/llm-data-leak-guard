# llm-data-leak-guard

[![CI](https://github.com/LevaAverGit/llm-data-leak-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/LevaAverGit/llm-data-leak-guard/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)

A guard layer that **inspects an outgoing prompt and strips sensitive content
— personal data, secrets, business identifiers — before it ever reaches the
LLM.** It sits in front of the model as a FastAPI proxy: inspect → redact →
forward. The model (and the model provider's logs) only ever see masked text.

*Why it matters: an employee pasting client data into an LLM is personal-data processing outside your perimeter — a 152-FZ / GDPR breach in the making. This proxy removes it before the model, or its provider's logs, ever see it.*

```
before:  Client Ivan Petrov, +7 900 123-45-67, contract 44-AB/2024,
         key sk-proj-EXAMPLE-not-real-000000000000000000000000,
         please help draft a reply.

after:   [PERSON], [PHONE], contract [CONTRACT], key [API_KEY],
         please help draft a reply.

caught:  4 of 4  ·  latency: tens of ms (run make bench)
```

> From `make bench` over the synthetic corpus: **20 of 20 ground-truth spans
> caught (100%)** across all ten leak classes. A guard call is dominated by the
> spaCy NLP pass and lands in the tens of milliseconds on a commodity laptop
> (the pure-regex secret/requisite detectors are sub-millisecond); it moves with
> hardware and CPU load, so run `make bench` for the numbers on your machine.
> Either way it is small next to the LLM round-trip it sits in front of.
> All example values are synthetic and the API key is an obviously-invalid
> `EXAMPLE` placeholder. Note the guard masks *`Client Ivan Petrov`* as a single
> `[PERSON]` — spaCy's name entity greedily absorbs the leading word `Client`
> (over-redaction of a harmless context word, never under-redaction of the
> name); see [What I learned](#what-i-learned).

## How it works

```
prompt ─▶ [ guard ] ─▶ redacted prompt ─▶ LLM
             │
             ├─ pii detector        (Presidio + spaCy en_core_web_sm; RU recognizers)
             ├─ secret detector     (regex + entropy, gitleaks-style)
             ├─ requisite detector  (custom patterns: contracts / accounts)
             └─ redactor            (replace every finding with a single [LABEL] mask)
```

1. **Detect.** Each detector returns `Finding` objects (`kind`, `label`,
   `start`, `end`, `detector`) — see the contracts in
   [`guard/schema.py`](guard/schema.py).
2. **Merge & redact.** Overlapping findings are de-duplicated, then each span is
   replaced by a uniform `[LABEL]` mask.
3. **Report.** The whole inspection is returned as a `GuardResult`
   (`original_len`, `redacted`, `findings`, `latency_ms`).
4. **Forward.** The FastAPI endpoint `/guard/complete` sends only the redacted
   text on to a (mock) model call.

Detection quality is measured against a fully synthetic, span-labeled corpus in
[`corpus/`](corpus/) — one class per leak type (phone, email, passport, INN,
SNILS, card, API key, private key, bearer token, contract number).

### Quick start

```bash
make install     # venv + deps + spaCy en_core_web_sm
make test        # one test per leak class
make bench       # real redaction latency over the corpus
make run         # start the guard proxy on 127.0.0.1:8000
```

## Why this is commercially valuable

DLP-for-LLM ("data loss prevention" at the prompt boundary) is a concrete,
budgeted compliance need — not a nice-to-have. The moment employees paste real
work into ChatGPT, Copilot or an internal assistant, regulated data crosses an
uncontrolled boundary:

- **Regulatory exposure with real fines.** Leaking personal data through a
  prompt is still processing personal data. Under **152-FZ** (Russia) and
  **GDPR** (EU) that means fines, mandatory breach notification, and — for
  152-FZ — data-localization obligations that a third-party US model endpoint
  violates outright. A guard that provably removes phones, passports, INN/SNILS
  and card numbers before egress is a direct control against those penalties.
- **Trade-secret and credential leakage.** Pasted API keys, private keys and
  contract terms don't just risk a fine — they hand attackers live credentials
  and hand competitors your commercial terms. This is exactly the class of leak
  that shows up in provider logs and prompt-history features.
- **It maps to money a business already spends.** Enterprises pay for DLP,
  CASB and secret-scanning today. This is the same control moved to the LLM
  boundary, and its value is *measurable*: "caught N of N leak classes at X ms
  of added latency" is a risk-reduction number a security team can put in a
  vendor review or an audit.
- **Deploys where the risk is.** As a lightweight proxy in front of any model,
  it needs no change to the model or the vendor — the cheapest possible place
  to insert a compliance control.

In short: it converts an unmanaged, per-employee leakage path into a single
measurable, auditable chokepoint — which is precisely what makes it something a
business will pay for.

## What I learned

Honest findings from building and benchmarking the guard against the synthetic
corpus (`make bench`, 25 repeats per example):

- **Presidio ships no Russian recognizers — you have to add them.** Out of the
  box Presidio is US-centric and misses RU phone, passport, INN and SNILS
  entirely. The project adds custom `PatternRecognizer`s with checksum
  validation (INN/SNILS mod-11, card Luhn). The checksum gate is what turns a
  permissive "10 or 12 digits" pattern into a *precise* detector: a random
  12-digit number is not masked, a valid INN is. Without it, recall stays high
  but precision collapses on any long number.

- **spaCy's `PERSON` entity over-fires on identifiers and greedily eats context
  words.** `en_core_web_sm` tagged `INN 770123456703` as a `PERSON` with score
  0.85 — and because the merge rule is *longest-span-wins*, that false PERSON
  shadowed the exact `RU_INN` finding and broke redaction. The fix that made the
  suite green: **drop any `PERSON` whose span contains a digit** (a real name
  never does). Separately, in the mixed prompt spaCy folds the leading word
  `Client` into `Client Ivan Petrov`, so the mask covers `Client` too. That is
  the core trade-off of ML-NER masking: it is *safe by construction* (it prefers
  over- to under-redaction) but not surgical on name boundaries.

- **The entropy threshold only works with a prefix-vs-loose split.**
  `ENTROPY_MIN = 3.0` bits/char sits below real base64/hex secrets (~4.5–5.5)
  but above low-entropy filler (`changeme` ≈ 2.75). Fixed-prefix provider keys
  (`sk-`, `AKIA…`, `ghp_…`) bypass the gate on their signature alone; only the
  *loose* rules — opaque bearer tokens and generic `key = value` assignments —
  are entropy-gated. Gate everything and you drop structured keys that happen to
  be low-entropy; gate nothing and `password = changeme` gets masked as a
  secret. The split is the whole reason false positives stay low.

- **Latency is dominated by the spaCy NLP pass, and the first call must be
  warmed up.** Every guard call runs the text through Presidio/spaCy, which puts
  steady-state cost in the tens of milliseconds per call (hardware- and
  load-dependent); the very first call is far slower (model load) and is
  excluded from the benchmark. The pure-regex detectors (secrets, requisites)
  are sub-millisecond — the slowest class is the multi-line `private_key` block,
  confirming cost scales with text length through the NLP engine, not with the
  number of secret rules.

- **The safety win is architectural, not statistical.** Even where a detector
  could miss, the design guarantees the raw value never leaves: the proxy passes
  only `result.redacted` to the model and logs only class labels and counts, so
  a benchmark or a production log can never contain a raw phone or key. Detection
  quality reduces *what slips through*; it can never cause a *leak of what was
  caught*.

## Limitations

- Detection is heuristic/ML — both **false positives and misses are possible**.
  The guard reduces risk; it is not a guarantee.
- The Russian recognizers cover **common** formats for phone / passport / INN /
  SNILS, not every possible variant.
- The demo proxy forwards to a **mock** model. Wiring a real provider is a
  deliberate extension point, kept out of scope so the guard stays deterministic
  and testable.
- The guard **never logs raw sensitive input** — only redacted text reaches
  logs and reports. Anything upstream of the guard is out of its control.

## License

MIT — see [`LICENSE`](LICENSE). Author: Lev Averyanov.
