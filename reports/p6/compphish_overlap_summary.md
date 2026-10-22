# P6.1 CompPhish v4 cross-source overlap audit

**Status:** complete. `scripts/audit_compphish_overlap.py` writes each source
comparison atomically to `compphish_overlap.json`; an interrupted run remains
marked `in_progress` and cannot be mistaken for a complete result. It uses raw
URL strings only and does not visit destinations or consume HTML features.

## Population and keys

The mapping workbook has 15,358 rows and 15,351 exact unique URLs after seven
duplicates. It contains 8,154 legitimate (`0`) and 7,197 phishing (`1`) unique
URLs. Five keys were compared against every deduplicated local source: exact
trimmed raw URL; conservative normalization (lowercase scheme/host, no fragment,
default ports removed); parsed hostname; offline-PSL registrable domain; and the
project template rule (scheme/query/fragment removed, trailing slash removed,
digit runs masked).

## Per-source intersections

| Source | Exact URLs | Normalized URLs | Hostnames | Registered domains | Templates |
|---|---:|---:|---:|---:|---:|
| PHIUSIL | 42 (0 conflicts) | 42 (0) | 1,264 (22) | 1,676 (90) | 1,161 (18) |
| LegitPhish | 31 (0) | 31 (0) | 406 (11) | 596 (22) | 342 (0) |
| MURL | 0 (0) | 0 (0) | 725 (101) | 2,462 (633) | 464 (9) |
| URL-Phish | 233 (1) | 238 (1) | 690 (12) | 941 (36) | 401 (3) |

Parentheses are keys with conflicting labels. Counts use each level's own
unique-key denominator; directional percentages are in the machine-readable
matrix. The exact URL-Phish conflict is `https://kakaku.com/`.

## Aggregate exclusions

| Exclusion level | CompPhish overlapping keys | Retained keys | Any label conflicts |
|---|---:|---:|---:|
| Exact URL | 293 / 15,351 | 15,058 | 1 |
| Normalized URL | 298 / 15,311 | 15,013 | 1 |
| Hostname | 2,441 / 13,510 | 11,069 | 121 |
| Registered domain | 3,563 / 9,436 | 5,873 | 685 |
| Template | 1,868 / 14,268 | 12,400 | 27 |

Excluding exact duplicates alone does not establish independence. A strict
registered-domain exclusion removes 37.8% of CompPhish domain groups, and a
template exclusion removes 13.1% of templates. Label conflicts on shared
hosting/platform domains require group-level exclusion rather than relabeling.

## Recommendation

CompPhish is suitable as a **candidate external-evaluation source only after a
frozen group-excluded subset is constructed**. Exclude every CompPhish record
sharing an exact/normalized URL, registered domain, or template with the chosen
development corpus, retain a manifest of reasons, and do not use the remaining
set to select features, thresholds, or models. It is not presently an untouched
external test set.
