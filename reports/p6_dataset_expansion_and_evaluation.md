# P6 — URL Dataset Expansion and Evaluation Design

**Status:** data audit and candidate review complete; no model, feature, threshold,
or artifact changes made. No retraining performed.
**Audit date:** 2026-10-19.

## Executive summary

The current URL results are dominated by collection and formatting differences.
The four raw URL corpora already in the workspace have sharply different scheme,
path, query, length, digit, and token distributions. The project has also used
both external URL corpora to choose changes: **LegitPhish is a development set**
and **URL-Phish is development-informed**, so neither can support a fresh final
generalization claim.

The most promising new public resources found are **PhishVN v4** (versioned,
documented confidence tiers and group-aware temporal split) and **CompPhish v4**
(recent raw URLs plus HTML, documented sources and CC BY 4.0). Neither archive
was downloaded in this phase. Their public metadata is verified below; row-level
structure and overlap with the project's corpora remain to be measured before
assigning either a training or final-test role. In particular, PhishVN says its
scheme signal is a collection artifact, so it cannot by itself resolve the
project's HTTP/HTTPS gap.

**Recommendation:** keep PHIUSIL and MURL as the existing training pool for now;
use a verified, row-level-audited release of CompPhish and/or PhishVN only for
development after overlap analysis; reserve a newly collected, source-documented
and timestamped corpus for the final test. If PhishVN's dated gold/silver test
slice passes overlap checks, report it as a separate Vietnamese-targeted
secondary evaluation, not as broad global evidence. Do not train or tune until
this plan is reviewed.

## 1. Audit method and artifacts

`scripts/p6_existing_data_audit.py` reads each raw URL/label pair, removes blank
URLs and exact duplicates, maps the documented labels, and measures scheme,
path, query, fragment, host-label count proxy, URL length, digit count, hyphen
count, and project suspicious-token count. It performs no URL requests and does
not load, score, or write model artifacts. Results are in
[`structural_distributions.csv`](p6/structural_distributions.csv).

The subdomain column is deliberately labelled a **proxy**: it counts hosts with
more than two dot-separated labels and does not use the Public Suffix List or
special-case `www`. It is useful for source comparison but is not the canonical
`tldextract` subdomain feature. Ratios below use the distinct, non-empty rows in
each source after exact-URL deduplication. These are full-corpus descriptive
statistics from raw URL strings, not model features recomputed through the
production extractor.

## 2. Existing data: composition and structure

Project convention: `0 = legitimate/benign`, `1 = phishing`. Scheme and path
shares below are from the deduplicated raw URLs. Counts differ from source-card
raw totals by the exact duplicates removed.

| Source / role | Legitimate | Phishing | HTTPS: legit / phish | HTTP: legit / phish | Path: legit / phish | Query: legit / phish | Fragment: legit / phish |
|---|---:|---:|---:|---:|---:|---:|---:|
| PHIUSIL — training/evaluation | 134,850 | 100,520 | 100% / 48.63% | 0% / 51.37% | 0% / 27.14% | 0% / 6.04% | 0% / 0.83% |
| MURL — augmentation only | 432,241 | 229,610 | 0% / 0% | 0% / 0.04% | 86.77% / 76.00% | 16.74% / 13.82% | 0.02% / 0.55% |
| LegitPhish — development | 37,380 | 63,492 | 99.99% / 5.53% | 0.01% / 92.42% | 1.05% / 99.37% | 0.10% / 1.18% | 0% / 1.34% |
| URL-Phish — development-informed | 98,641 | 16,590 | **34.94% / 91.43%** | 65.06% / 8.57% | 23.39% / 55.86% | 2.01% / 14.56% | 0.19% / 1.08% |

Additional per-class measurements:

| Source / class | Median URL length | Mean URL length | Mean digits | Mean hyphens | Mean suspicious-token matches | Host labels >2 proxy |
|---|---:|---:|---:|---:|---:|---:|
| PHIUSIL legitimate | 27 | 27.23 | 0.051 | 0.083 | 0.007 | 100% |
| PHIUSIL phishing | 34 | 46.22 | 4.336 | 0.706 | 0.129 | 86.16% |
| MURL legitimate | 45 | 56.04 | 5.489 | 1.817 | 0.134 | 27.14% |
| MURL phishing | 35 | 52.02 | 5.949 | 0.585 | 0.237 | 46.15% |
| LegitPhish legitimate | 27 | 27.55 | 0.072 | 0.105 | 0.008 | 99.59% |
| LegitPhish phishing | 34 | 39.43 | 13.476 | 0.098 | 0.042 | 95.65% |
| URL-Phish legitimate | 26 | 29.10 | 0.147 | 0.244 | 0.003 | 85.82% |
| URL-Phish phishing | 38 | 56.31 | 5.756 | 0.937 | 0.216 | 69.06% |

`Suspicious-token matches` counts literal occurrences from the project's fixed
token list; it is not a claim that each match is malicious. A fragment is the
literal `#` component in the raw URL. MURL stores nearly all URLs without a
scheme, so its zero HTTPS share is a file-format fact, not evidence that its
sites used HTTP. MURL does add path-bearing benign examples, but it cannot teach
the HTTPS-plus-path region.

### Interpretation

- PHIUSIL has a near-perfect shape shortcut: every legitimate URL is HTTPS and
  bare-host, while only 27.14% of phishing URLs have a parsed path. Its
  in-distribution scores are not evidence of realistic transport/path
  generalization.
- LegitPhish repeats the same broad shortcut: almost all legitimate examples
  are HTTPS homepages, while almost all phishing URLs have paths and use HTTP.
- URL-Phish reverses the class direction on scheme (phishing is more often
  HTTPS) and contains substantially more path/query-bearing legitimate URLs.
  This source is more structurally useful, but it has already informed model
  development.
- MURL has path/query-bearing data in both classes but no scheme information.
  Its use as augmentation alone cannot correct transport-related dataset shift.
- Legitimate and phishing URL length, digit, hyphen, and suspicious-token rates
  vary by source. These measurements argue for source-stratified evaluation;
  pooling rows without provenance would hide the shift.

### Existing source provenance and dates

| Source | Documented provenance / time | Licensing and caveats |
|---|---|---|
| PHIUSIL | UCI record donated 2024-03-03; authors describe URLs as mostly latest at collection. Raw label `1=legitimate`, `0=phishing`. | CC BY 4.0. Every legitimate sample is a bare HTTPS host in the actual file. |
| MURL | Mendeley dataset; role in this project is augmentation. Its URL strings are scheme-less. | Project inventory records CC BY 4.0. Broad extra source classes are excluded by preparation. It shares registered domains with PHIUSIL. |
| LegitPhish v2 | Mendeley published 2025-05-22. Phishing from URLHaus/other repositories; legitimate examples from Wikipedia, Stack Overflow and other high-authority sources. Raw `0=phishing`, `1=legitimate`. | CC BY 4.0. 98.81% of deduplicated legitimate URLs are byte-identical to PHIUSIL; it has already informed development. |
| URL-Phish v2 | Mendeley published 2026-03-09; phishing source records were collected 2024-11 through 2025-09 from PhishTank; benign samples from ROR and trusted/popular domains. Labels `0=benign`, `1=phishing`. | CC BY 4.0. It has already informed model development and cannot be a final untouched set. |

## 3. Duplicates, overlap, and current splits

### Source-level duplicates and known overlap

- PHIUSIL: 235,795 raw records; 425 exact duplicate URLs removed. All 425
  duplicate rows explain the difference between 100,945 source-card phishing
  records and 100,520 unique phishing URLs. Legitimate unique count remains
  134,850.
- LegitPhish: 101,219 source-card rows; 346 duplicate URL rows removed (160
  legitimate, 186 phishing); one unlabelled record is excluded from the prepared
  set. Among the remaining 100,872, 36,935 legitimate URLs are byte-identical
  to PHIUSIL and no phishing URL is byte-identical.
- URL-Phish: 116,600 source-card rows; 1,369 duplicate URL rows removed (1,359
  benign, 10 phishing). Of the prepared set, 595 benign URLs are byte-identical
  to PHIUSIL; 22,625 rows have a registered domain appearing in PHIUSIL training;
  176 rows are byte-identical to LegitPhish.
- MURL: the prepared source has 661,854 rows before three exact duplicate rows
  are removed. Project inventory reports zero exact URL overlap with PHIUSIL
  but 10,219 shared registered domains. It is augmentation, not independent
  evaluation evidence.
- Cross-source normalized-template and label-conflict counts for all four
  complete raw corpora have **not** been established. The existing P6 audit
  script's public-suffix lookup did not complete in this environment. These
  counts remain a required gate before any mixing or external-test designation.

The old P6 log also emitted composition warnings because it compared
**post-dedup** class counts against **pre-dedup** source-card totals. Those
warnings are accounted for by the duplicate counts above; they are not evidence
of label corruption. New class counts should always be reported both before and
after deduplication.

### Split integrity

| Existing split | Counts / integrity | Appropriate interpretation |
|---|---|---|
| PHIUSIL random train / validation / test | 164,758 / 35,306 / 35,306; exact URLs are disjoint. Domains are shared by design. | In-distribution benchmark only. |
| PHIUSIL domain train / test | 196,779 / 38,591; zero shared exact URLs and zero shared registered domains. | Unseen-domain evaluation within PHIUSIL's collection process. |
| PHIUSIL template-clean test | 38,342; domain train has zero shared exact URLs, domains, or the project's current template keys. 249 rows were removed from domain test to make this view. | Better within-source novelty test, still not external generalization. |
| Historical contaminated evaluation | Random train+validation shared 32,826 exact URLs, 22,513 domains, and 86.19% of test templates with the purported domain test. | Superseded; those old figures were contaminated. |
| MURL-augmented fit / validation | 268,577 / 48,202; GroupShuffleSplit by registered domain; source report says fit and validation domains are disjoint. | Threshold/development split for the augmented pool, not external test. |
| LegitPhish / URL-Phish | Each is one prepared external file, no train/dev/test split. Both have been used to make development decisions. | Treat both as development evidence only from this point forward. |

The project's current `url_template()` drops scheme, query and fragment and masks
digit runs, but retains the hostname. It therefore finds same-host templates,
not cross-domain campaign templates. P6's split plan below defines an additional
host-independent campaign-template key and records both kinds.

## 4. Additional dataset candidates

Metadata and licensing below were checked against the dataset hosts or
publishers on 2026-10-19. **No candidate archive was downloaded in this phase**;
therefore archive hashes, row-level feature distributions, and overlaps with the
existing four sources are explicitly unverified.

| Candidate | Verified release facts | Strengths for this problem | Limitations / P6 disposition |
|---|---|---|---|
| **PhishVN v4 / v3.1.0 open** — [Mendeley DOI 10.17632/b97hxbxtpd.4](https://data.mendeley.com/datasets/b97hxbxtpd/4) | 53,116 URLs: 36,706 phishing (2,587 gold/silver core + 34,119 bronze) and 16,410 legitimate. Labels are named `phishing`/`benign`. Core dates 2020–2025; bronze dates mostly 2021–2024. CC BY 4.0. Published 2026-08-18. Mendeley lists a 5.54 MB open ZIP and per-file SHA-256 `MANIFEST.txt`; v4 data bytes equal v3. | Provenance, tiered confidence, scenario labels, more than one kind of benign source, fixed domain-grouped time-aware train/val/test split, and a checkable manifest. The official split sizes are 36,909 / 7,266 / 8,941. | Vietnamese-targeted only. The author explicitly says `is_https` is a collection artifact because schemes were stripped. Bronze phishing has reported 12.1% label noise (95% CI 7.1–20.0%); benign label audit reports 4.0% noise. Do not pool bronze positives as unquestioned truth. Row-level paths, queries, subdomains, overlaps, and manifest verification must be audited after download. Best as a locale-specific secondary development/evaluation corpus. |
| **CompPhish v4** — [Mendeley DOI 10.17632/fmbs4kp9wz.4](https://data.mendeley.com/datasets/fmbs4kp9wz/4) | 15,358 URLs: 7,204 phishing, 8,154 legitimate; labels `0=legitimate`, `1=phishing`. Phishing from PhishTank/OpenPhish; benign from DataForSEO Top-1000. Collection period 2024-09 through 2025-08. CC BY 4.0. Version 4 published 2026-08-24. Release includes raw URL-to-HTML mapping, feature dictionary, auxiliary data, and HTML. | Recent, raw URLs retained, source provenance and dates stated, both classes, HTML allows later audit of destination state. It is the closest candidate to a current mixed URL dataset among the public releases found. | The benign source is a ranked top-site list (possible easy-negative/source shortcut); phishing feeds overlap conceptually with existing development corpora; active-site HTML capture introduces survival bias. No published per-class HTTP/path/query distribution or split protocol was confirmed. Audit overlaps and structures first; use only as development/training after row-level checks, not the sole final test. v3's listed HTML archive was 692 MB; verify v4 file inventory before retrieval. |
| **PhishStorm v1** — [Aalto research record, DOI 10.24342/f49465b2-c68a-4182-9171-075f0ed797d5](https://research.aalto.fi/en/datasets/phishstorm-phishing-legitimate-url-dataset) | 96,018 URLs, balanced 48,009 legitimate / 48,009 phishing; labels `0=legitimate`, `1=phishing`; produced/published 2014. A 3.24 MB zip is listed. | Raw URL field is available; useful as a historical stress set if its data terms permit. | License is explicitly unspecified; very old collection. Do not download/reuse until rights are clarified; unsuitable as a contemporary training source. |
| **Phishing and Legitimate Websites Dataset v1.0** — [Zenodo DOI 10.5281/zenodo.21379702](https://zenodo.org/records/21379702) | Record says 60,000 URLs: 31,641 phishing from PhishTank and 28,359 legitimate from Tranco; published 2026-07-15. One 1.8 GB ZIP; record shows an MD5 for the archive. | Current public mixed corpus and URL-level features; potential structural comparison after acquisition. | The license was not confirmed from accessible record metadata; most payload is screenshots, and per-class URL shape/time and a leakage-resistant split are not documented on the record. **Do not use until the license, version contents, and overlap are verified.** |

PhishTank and OpenPhish are useful provenance feeds but phishing-only. Tranco/ROR
are useful benign sampling frames but not labeled URL corpora on their own. Any
fresh paired dataset built from feeds and allowlists must preserve source and
timestamp for every row and must audit class-specific sampling shortcuts.

### Download and provenance checks before use

1. Pin the exact dataset DOI/version and save its landing-page metadata, license,
   retrieval date, file list, and source record in the project.
2. For PhishVN, verify `MANIFEST.txt` SHA-256 for every extracted data file and
   record the archive hash generated locally. Preserve source, tier, original
   label, label timestamp, and supplied split.
3. For CompPhish, verify the v4 file mapping and document any hashes present in
   its manifest/readme. Generate local SHA-256 hashes for each downloaded file;
   do not download or open the large HTML bundle in the normal project runtime.
4. Recount labels before and after blank/duplicate filtering; assert the raw
   convention from source metadata and validate a sample of URLs against
   provenance/status fields. Never infer a label map only from class balance.
5. Keep raw archives read-only; write normalized URL tables and audit products
   under separate versioned paths. Record every duplicate, invalid row,
   conflict, and exclusion with a reason and source row ID.

## 5. Leakage-resistant evaluation plan

1. **Freeze inputs and lineage.** Preserve raw snapshots, version DOI, file
   hashes, labels as published, source/source-version, observed/first-seen date,
   and retrieval date. Use one canonical normalizer for matching but retain the
   original URL bytes.
2. **Deduplicate before splitting.** Remove exact duplicates across all sources
   using a normalized exact key (case-fold scheme/host, remove default ports,
   normalize safe percent encodings; preserve path/query semantics). Keep one
   row only when labels agree; quarantine conflicting URL labels for manual
   provenance review rather than majority-voting them. Emit an exclusion ledger.
3. **Create relatedness groups.** Compute registrable domains with a pinned
   Public Suffix List snapshot. Separately compute campaign-template keys from
   normalized host-independent path structure (masking ID-like numbers and
   high-entropy tokens while retaining meaningful path segments). Connect rows
   sharing a registrable domain **or** a specific normalized template; keep each
   connected group wholly in one split. Report common generic templates and
   define a pre-registered rule to quarantine templates so broad `/login`-type
   paths do not merge unrelated records into one giant group.
4. **Use grouped train/development splits.** Within the admitted training
   corpora, split by connected group; no exact URL, registered domain, or
   retained template may cross train/validation/development. Preserve source and
   class strata as far as group sizes allow. Report dropped/oversized groups.
5. **Keep final evaluation source- and time-separated.** Select the final corpus
   only after source-overlap analysis and before model fitting. The preferred
   final test is a new frozen prospective snapshot with documented phishing
   feed timestamps and a contemporaneous benign collection that has realistic
   paths, queries, subdomains, and HTTP/HTTPS diversity. Exclude all matching
   domains/templates from training/development; never tune features or threshold
   against final-test results.
6. **Use time only when supported by source timestamps.** For time-stamped
   phishing, use the source-attested first-seen/verification time, grouped by
   registrable domain, with an earlier training window and later final window.
   `scraped_at`/download time is not a substitute. Undated benign URLs should be
   reported separately, not assigned invented dates. The existing PHIUSIL,
   MURL, and URL-Phish files do not support a trustworthy temporal split of
   every row.
7. **Report strata and uncertainty.** Report metrics separately by source,
   confidence tier, HTTP/HTTPS, path/query presence, seen/unseen domain, and
   label conflicts. Include counts, confidence intervals, calibration, and
   threshold-free ranking metrics; do not average source-specific scores into
   one headline.

## 6. Dataset-role recommendation

| Role | Recommendation | Gate |
|---|---|---|
| **Training** | Keep PHIUSIL + MURL as the current baseline pool. Consider adding only audited PhishVN gold/silver rows or CompPhish v4 rows in a later approved experiment. Keep PhishVN bronze separate as a label-noise sensitivity stratum. | Verify source/label conflicts, exact URLs, registered domains, template overlap, class-conditional structure, and any MURL class filtering before use. |
| **Development / validation** | Use grouped folds within each admitted training source. CompPhish v4 is a strong candidate for a separate source-specific development evaluation. PhishVN's official grouped temporal split can provide a Vietnamese-targeted secondary check, after overlap audit. | Do not use any split or source used for final testing to select features, thresholds, or hyperparameters. Preserve PhishVN's supplied split for reproducibility. |
| **Final test** | Build a new frozen prospective external snapshot with a different collection window and a documented, structurally representative benign source. Reserve PhishVN's dated gold/silver test records as a secondary locale-specific test only if row-level overlap is zero. | Decide and hash the final snapshot before training/tuning. URL-Phish and LegitPhish are development-informed and are permanently excluded as untouched final tests. |

The public candidates do not yet establish a clean global final test: PhishVN is
locale-specific and strips schemes; CompPhish's source shape and cross-corpus
overlap need inspection; the Zenodo candidate's license was not confirmed; and
PhishStorm's license is unspecified. A prospective collection is therefore the
best route to the final claim.

## 7. Known documentation discrepancy

The current raw URL-Phish release in `data/raw/URLPhish_urls.csv` was recounted
directly. Before deduplication, benign rows have 35,093/100,000 HTTPS URLs
(35.09%); after deduplication, the measured rate is 34.94%. Phishing rows have
15,178/16,600 HTTPS URLs (91.43%). Several current documents state **17.9%**
benign HTTPS. That number does not match the raw URL column in the checked local
release. P6 records the direct measurement here and in
`structural_distributions.csv`; the older result/report files have not been
rewritten. Resolve the discrepancy against the original source archive before
using the scheme-rate explanation in future presentations.

## 8. Actions completed and not performed

- Added a read-only structural audit script and per-source/per-class CSV output.
- Reviewed the current inventory, dataset download/preparation code, split
  metadata, overlap reports, model comparison, and remediation reports.
- Verified candidate public metadata, label definitions, licenses where
  explicitly available, release sizes, and available checksum mechanisms.
- Did not download new datasets, change any model or feature, alter thresholds,
  change the SMS branch, promote the revised URL artifact, run retraining, or
  commit changes.
- No test suite was run as part of this data-only phase.

## Sources checked

- [UCI PHIUSIIL dataset record](https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset)
- [LegitPhish v2 Mendeley record](https://data.mendeley.com/datasets/hx4m73v2sf/2)
- [URL-Phish v2 Mendeley record](https://data.mendeley.com/datasets/65z9twcx3r/2)
- [PhishVN v4 Mendeley record](https://data.mendeley.com/datasets/b97hxbxtpd/4)
- [PhishVN Data in Brief article](https://doi.org/10.1016/j.dib.2026.113195)
- [CompPhish v4 Mendeley record](https://data.mendeley.com/datasets/fmbs4kp9wz/4)
- [PhishStorm Aalto research record](https://research.aalto.fi/en/datasets/phishstorm-phishing-legitimate-url-dataset)
- [Zenodo mixed phishing/legitimate website dataset](https://zenodo.org/records/21379702)
