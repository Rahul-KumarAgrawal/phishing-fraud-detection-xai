# P6.2 CompPhish frozen external-evaluation candidate

## Development boundary

The prohibited development corpus is PHIUSIL, MURL, LegitPhish, and URL-Phish.
All four are included because they have contributed to URL fitting, augmentation,
diagnosis, feature work, threshold work, error analysis, or development-facing
evaluation. URL-Phish is explicitly not treated as an independent final test.

## Frozen file

`data/processed/compphish_external_frozen.csv` is evaluation-only. It must not
be used for training, validation, threshold selection, feature selection, or
iterative error analysis. P6.1 structural inspection is recorded as prior
exposure; no model prediction has been run against it.

| Measure | Value |
|---|---:|
| Raw mapping rows | 15,358 |
| Exact duplicate rows removed | 7 |
| Unique URLs before filtering | 15,351 |
| Retained rows | 6,443 |
| Retained legitimate / phishing | 4,490 / 1,953 |
| Retained hostnames / domains / templates | 6,012 / 5,720 / 6,258 |

The retained CSV SHA-256 is
`56f6e8545a7df5f675b740d0858ee0fb4b2fa920a085d4aceae2838bfbc0e377`.

## Exclusions

Rows are exact-deduplicated by retaining the lowest `serial_number`. For
multi-reason rows, the primary-disposition order is malformed, exact URL,
normalized URL, hostname, registered domain, then template. Nonexclusive flags
show the full impact of every rule.

| Primary disposition | Rows |
|---|---:|
| Exact URL | 293 |
| Normalized URL | 7 |
| Hostname | 3,125 |
| Registered domain | 5,275 |
| Template | 208 |
| Malformed | 0 |
| Retained | 6,443 |

Nonexclusive overlap flags are 293 exact URLs, 300 normalized URLs, 3,425
hostnames, 8,700 registered domains, and 2,477 templates. The strict domain
rule is responsible for most exclusions because it protects against related
URLs hosted under a development-seen registrable domain. It reduces class
balance from 53.1% legitimate to 69.7% legitimate and is therefore a material
representativeness limitation.

## Verification

Post-filter overlap is zero for exact URL, normalized URL, hostname, registered
domain, and template. The deterministic command is:

```text
python scripts/freeze_compphish_external.py
```

Two consecutive runs produced the same retained-file SHA-256. The manifest
records source-file hashes, definitions, dependency versions, counts, the
creation timestamp, and all zero-overlap assertions.

## Recommendation

Treat the frozen CSV as a leakage-controlled **candidate** external evaluation
set. It is not fully blind because its structure was inspected in P6.1, and it
is structurally shifted after strict group exclusions. Do not use it for any
model decision. It can support one final evaluation only after the training and
development protocol is approved and frozen.
