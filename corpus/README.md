# Synthetic evaluation corpus

This folder holds the labeled examples used to measure detection quality
("caught N of N") and to drive the before/after demo. Each file is one leak
class with ground-truth spans marking exactly what a detector **must** find.

> [!IMPORTANT]
> **Every value in this corpus is synthetic and fabricated for testing.**
> There is no real personal data here, and none of the keys, tokens or private
> keys are valid credentials. Placeholder secrets deliberately contain the word
> `EXAMPLE` (for example `sk-proj-EXAMPLE-not-real-000000000000000000000000`).
> Numbers that carry a checksum (INN, SNILS, bank card) are generated with a
> *valid checksum* — so checksum-validating detectors are genuinely exercised —
> but from arbitrary synthetic digit bases that identify no real person or
> entity.

## File format

Each `*.json` file:

```json
{
  "leak_class": "phone",
  "kind": "pii",
  "label": "PHONE",
  "detector_hint": "presidio",
  "synthetic": true,
  "description": "...",
  "examples": [
    {
      "id": "phone-1",
      "text": "Please call the client back at +7 900 123-45-67 before noon.",
      "spans": [
        { "start": 31, "end": 47, "label": "PHONE", "kind": "pii",
          "text": "+7 900 123-45-67" }
      ]
    }
  ]
}
```

- `spans[*].start` / `end` are half-open Python string offsets, so
  `text[start:end] == text` field of the span (the test suite asserts this).
- `kind` and `label` line up with the `Finding` contract in `guard/schema.py`.
- `index.json` lists every class file for iteration in tests.

## Leak classes

| File                  | class            | kind      | label         | what it covers |
|-----------------------|------------------|-----------|---------------|----------------|
| `phone.json`          | phone            | pii       | `PHONE`       | RU / international phone numbers, varied formatting |
| `email.json`          | email            | pii       | `EMAIL`       | email addresses on reserved `example.*` domains |
| `passport.json`       | passport         | pii       | `PASSPORT`    | RU internal passport (4-digit series + 6-digit number) |
| `inn.json`            | inn              | pii       | `INN`         | RU taxpayer number, 10-digit (legal) and 12-digit (individual), valid checksum |
| `snils.json`          | snils            | pii       | `SNILS`       | RU insurance account number `XXX-XXX-XXX YY`, valid control digit |
| `card.json`           | card             | pii       | `CARD`        | 16-digit payment card PAN, Luhn-valid (reserved `4111…` Visa test number) |
| `api_key.json`        | api_key          | secret    | `API_KEY`     | OpenAI / AWS / GitHub style keys (EXAMPLE placeholders) |
| `private_key.json`    | private_key      | secret    | `PRIVATE_KEY` | PEM private-key block (synthetic FAKE body) |
| `bearer_token.json`   | bearer_token     | secret    | `BEARER_TOKEN`| JWT / bearer authorization token (synthetic payload) |
| `contract_number.json`| contract_number  | requisite | `CONTRACT`    | contract / agreement reference numbers |

`mixed.json` is not a class of its own: it is one realistic prompt that combines
a name, a phone, a contract number and an API key, used for the README
before/after demo and the "caught N of N" headline metric.

## Adding examples

Keep every new value synthetic. If you need a checksum-valid RU identifier or a
Luhn-valid card, generate it from an arbitrary base rather than copying a real
one, and keep secret placeholders obviously invalid (`EXAMPLE`, trailing zeros).
