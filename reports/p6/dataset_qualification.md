# P6 — URL dataset qualification and evaluation design

**Run:** 2026-10-19. **Scope:** data audit only. No URL model, feature schema,
threshold, artifact, or SMS code was changed; no training was run.

## Decision

The four local sources are useful for development diagnostics but cannot supply
an independent final claim. `URL-Phish` has already informed development and
has 438 exact rows in the current augmented fit set. `LegitPhish` shares 36,935
exact URLs with PHIUSIL. MURL has 402 exact URL conflicts with LegitPhish.

Use a newly qualified external corpus, held completely outside source selection,
split design, tuning, and feature selection, for the final evaluation. Do not
retrain until that corpus has been obtained and this plan is approved.

## A. Inventory, labels, and provenance

| Source | Raw rows | Deduplicated rows (legit / phish) | Project label mapping | Intended status |
|---|---:|---:|---|---|
| PHIUSIL (UCI, 2023, CC BY 4.0) | 235,795 | 235,370 (134,850 / 100,520) | raw `1 -> 0`, `0 -> 1` | development source |
| LegitPhish (Mendeley v2, 2025, CC BY 4.0) | 101,219 | 100,872 (37,380 / 63,492) | raw `1 -> 0`, `0 -> 1` | development diagnostic only |
| MURL (Mendeley v2, 2023, CC BY 4.0) | 661,854 | 661,851 (432,241 / 229,610) | `0 -> 0`, `1 -> 1` | augmentation candidate only |
| URL-Phish (Mendeley v2, 2024, CC BY 4.0) | 116,600 | 115,231 (98,641 / 16,590) | `0 -> 0`, `1 -> 1` | development-informed; never final |

The documented source-card totals are raw totals. The lower post-dedup totals
are expected and recorded in `dataset_summary.json`; they are not a label-map
failure. Empty and unmapped rows are excluded before deduplication.

## B. Structural distributions

`structural_by_source_class.csv` is a fixed-seed 10,000-row per-class sample
when a class exceeds that size. It measures raw URL strings, so missing scheme
is distinct from `http`; it does not visit destinations. Key percentages:

| Source / class | HTTPS | no scheme | path | query | median length | mean digits | mean hyphens |
|---|---:|---:|---:|---:|---:|---:|---:|
| PHIUSIL legitimate | 100.0 | 0.0 | 0.0 | 0.0 | 27 | 0.05 | 0.08 |
| PHIUSIL phishing | 48.7 | 0.0 | 26.9 | 6.2 | 34 | 4.13 | 0.69 |
| LegitPhish legitimate | 100.0 | 0.0 | 1.1 | 0.1 | 27 | 0.07 | 0.10 |
| LegitPhish phishing | 5.1 | 2.3 | 99.4 | 1.2 | 34 | 13.50 | 0.10 |
| MURL legitimate | 0.0 | 100.0 | 86.7 | 16.4 | 45 | 5.40 | 1.87 |
| MURL phishing | 0.0 | 100.0 | 76.9 | 13.7 | 35 | 5.93 | 0.61 |
| URL-Phish legitimate | 34.0 | 0.0 | 23.6 | 2.1 | 26 | 0.15 | 0.24 |
| URL-Phish phishing | 91.5 | 0.0 | 56.0 | 14.5 | 38 | 5.88 | 0.94 |

The prior `17.9%`/`86.2%` URL-Phish HTTPS statement is not reproducible. Exact
deduplicated counts are **34.94%** HTTPS for legitimate (34,467 / 98,641) and
**91.43%** for phishing (15,168 / 16,590); the sampled values
above corroborate it. The old numbers must not be used.

## C. Leakage and conflicts

Full-file overlap results are in `overlaps.json`, using exact raw URL,
offline-PSL registered domain, and scheme/query/fragment-stripped template.

| Pair | exact URLs | exact label conflicts | shared domains | domain conflicts | shared templates |
|---|---:|---:|---:|---:|---:|
| PHIUSIL / LegitPhish | 36,935 | 0 | 36,530 | 47 | 36,947 |
| PHIUSIL / MURL | 0 | 0 | 10,218 | 2,395 | 4,159 |
| PHIUSIL / URL-Phish | 595 | 0 | 7,200 | 112 | 5,060 |
| LegitPhish / MURL | 402 | 402 | 3,275 | 1,506 | 910 |
| LegitPhish / URL-Phish | 176 | 0 | 2,130 | 24 | 1,379 |
| MURL / URL-Phish | 0 | 0 | 3,976 | 1,464 | 171 |

Do not merge rows where any source disagrees on the exact label. Domain conflict
does not by itself prove a bad label because hosted domains can contain both
benign and phishing pages, but it rules out a domain as an isolation group.

Current split audit: random train/test has 0 exact URLs but 2,234 shared
registered domains and 1,360 shared templates. The domain split has 0 shared
domains but 143 templates. The template-clean split has 0 exact URLs, domains,
and templates against domain training. The currently augmented fit set overlaps
URL-Phish on 438 exact URLs and LegitPhish on 26,545; neither can be final test.

## D. Candidate qualification

| Candidate | Official version and license | Reported classes / provenance | Qualification result |
|---|---|---|---|
| PhishVN | Mendeley `b97hxbxtpd`, v4, DOI `10.17632/b97hxbxtpd.4`, CC BY 4.0, published 2026-08-18 | 53,116: verified core 2,587 phish and 16,410 legitimate; 34,119 bronze phishing from ChongLuaDao/OpenPhish | promising for an external challenge set, but retain verified core separately from bronze labels |
| CompPhish | Mendeley `fmbs4kp9wz`, v4, DOI `10.17632/fmbs4kp9wz.4`, CC BY 4.0, published 2026-08-24 | 15,358: 7,204 phishing from PhishTank/OpenPhish; 8,154 legitimate from DataForSEO Top-1000; Sep 2024–Aug 2025 | strongest candidate for independent final evaluation if its URL mapping passes overlap audit |

Automated download was attempted against Mendeley’s official public API on this
run. It returned HTTP 403 with a Cloudflare challenge, so no candidate archive,
checksum, manifest, or extracted candidate row count can honestly be claimed as
verified. Manual acquisition must preserve the exact downloaded filename,
SHA-256, publication version, license text, and file manifest before the
candidate may enter this repository. Do not use a browser copy of rows.

## E. Proposed protocol

1. Preserve the original archive and a SHA-256 manifest under `data/raw/`.
2. Normalize only for comparison: trim, lowercase host, canonicalize default
ports, remove fragment, and retain original string and label.
3. Exclude exact URL duplicates, cross-source exact label conflicts, and any
candidate URL/domain/template that overlaps the selected development pool.
Record every exclusion with reason and source IDs.
4. Build development splits with groups formed by the connected components of
registered domain **and** normalized template. This prevents a shared template
from crossing splits even on different domains. Use temporal ordering only for
records with documented collection timestamps.
5. Freeze the candidate external set before any retraining. It is used once,
after model and threshold selection, with no feedback into development.

## F. Recommendation and stop point

Use PHIUSIL plus only conflict-screened MURL rows for development; retain
LegitPhish for diagnostics. Retire URL-Phish from final-test status and mark it
development-informed. Prefer CompPhish v4 as the final external evaluation and
PhishVN v4 verified core as a secondary external set, after archive verification
and the exclusion audit above. No model retraining has begun.

## Reproducibility outputs

- `scripts/audit_url_datasets.py`
- `reports/p6/structural_by_source_class.csv`
- `reports/p6/scheme_path_crosstab.csv`
- `reports/p6/overlaps.json`
- `reports/p6/split_integrity.json`
- `reports/p6/audit_log.txt`
