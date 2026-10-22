# P10 — Dataset Remediation and Independent Benchmark Qualification

## Decision

P7 Random Forest is **not promotable**. P9 is its final recorded CompPhish
evaluation, and CompPhish must not become a source of subsequent tuning,
candidate selection, or P8 evaluation. P10 did not open the frozen CSV, rerun
inference, inspect individual CompPhish errors, or alter P9 artifacts.

The next model-development phase requires a new, source-separated data plan. A
new prospective benchmark is the only recommended broad final evaluation route;
PhishVN v4 verified core is a promising, separately reported secondary
benchmark once acquired and audited.

## P9 read-only integrity review

`scripts/audit_p10_p9_integrity.py` read only P9's saved JSON and the evaluator
source. It confirms that the saved `TN=69, FP=4,421, FN=6, TP=1,947` matrix
arithmetically yields accuracy 0.3128977, balanced accuracy 0.5061476,
precision 0.3057475, recall 0.9969278, F1 0.4679726, FPR 0.9846325, and FNR
0.0030722. The evaluator asserts class order `[0, 1]`, selects
`predict_proba(X)[:, 1]` as the phishing score, and passes that score to
`roc_auc_score(y, scores)` and `average_precision_score(y, scores)`.

ROC-AUC cannot be reconstructed from a confusion matrix, so P10 verifies the
recorded computation path rather than falsely claiming a second calculation.
The audit did not run that evaluator, load a model, or read frozen rows.
Structured evidence: `reports/p10/p10_p9_integrity_review.json`.

## Development-data remediation findings

The following are measured facts from P6/P8 development reports:

| Source and class | HTTPS | Non-empty path | Query | Consequence |
|---|---:|---:|---:|---|
| PHIUSIIL legitimate | 100.0% | 0.0% | 0.0% | No legitimate HTTPS-with-path/query coverage. |
| PHIUSIIL phishing | 48.7% | 26.9% | 6.2% | Scheme and URL shape differ sharply by class. |
| MURL legitimate | 0.0% stored with scheme | 86.7% | 16.4% | Adds paths/queries but every stored URL is scheme-less. |
| MURL phishing | 0.0% stored with scheme | 76.9% | 13.7% | Does not repair HTTPS/HTTP representation. |
| URL-Phish legitimate | 34.0% | 23.6% | 2.1% | Structurally useful, but already development-informed. |
| URL-Phish phishing | 91.5% | 56.0% | 14.5% | Shows that scheme direction can reverse across sources. |

Additional P8 results show a source-conditioned development failure: P7 RF
false-positive rate was 60.16% for MURL legitimate rows (5,685/9,450) and
0.13% for PHIUSIIL legitimate rows (26/20,282). On that same split,
path-bearing legitimate URLs had a 56.93% false-positive rate and query-bearing
legitimate URLs 75.16%. These facts demonstrate insufficient coverage of
ordinary legitimate URL structures in the *admitted development sources*.

They do not prove that one feature, the source field, or a single collection
mechanism caused P9. Model errors can arise from correlated lexical structure,
label definitions, source sampling, collection time, and their interactions.
The evidence supports collecting labelled legitimate URLs with paths, queries,
fragments, subdomains, digits, hyphens, varied lengths, and both schemes while
preserving collection provenance.

## Candidate sources

| Candidate | Official source / legal status | Raw URLs and labels | Known structure or bias | Row-level overlap status | P10 disposition |
|---|---|---|---|---|---|
| **PhishVN v4 verified core** | [Mendeley DOI 10.17632/b97hxbxtpd.4](https://data.mendeley.com/datasets/b97hxbxtpd/4), CC BY 4.0, published 2026-08-18 | 18,997 verified-core records: 2,587 gold/silver phishing from NCSC Tin Nhiem Mang and 16,410 legitimate; labels and confidence tiers are supplied. | Vietnamese-focused. Legitimate rows combine trusted organisations, a `.vn` Tranco slice, and global Tranco. It explicitly strips schemes, so `is_https` is unusable and raw HTTP/HTTPS diversity cannot be measured. Bronze positives have reported 12.1% label-noise estimate; do not merge them into the verified core. | **Unverified:** no archive is local, so exact/normalized/domain/template overlap has not been computed. | Acquire and audit as a locale-specific secondary benchmark or development source, retaining verified core and bronze separately. |
| **New PhishTank + provenance-backed legitimate snapshot** | [PhishTank developer feed](https://phishtank.org/developer_info.php); current terms require review and snapshotting. Verified feed includes `verification_time`; its terms say its defined data is available for commercial use without charge. | Verified phishing URLs with submission/verification timestamps; no legitimate class. | Strong phishing provenance and temporal fields, but online-only data and feed survivorship can bias examples. | Not collected; no overlap result exists. | Preferred phishing arm for a new prospective benchmark after terms and snapshot are recorded. |
| **ROR organisation metadata as negative-source seed** | [ROR data dump](https://ror.readme.io/docs/data-dump), CC0 1.0 | Includes organisation website metadata, not a confirmed page-level benign label. | More diverse institutional organisations than bare top-domain lists, but website field is usually a homepage and can be stale or compromised. | Not collected; no overlap result exists. | Suitable only as a documented legitimate-source seed. Validate URL availability and exclude known phishing-feed/domain/template conflicts before assigning a benign label. |
| **OpenPhish academic feed** | [OpenPhish Academic Use Program](https://www.openphish.com/academic_use.html) requires approved, non-commercial access and prohibits third-party sharing. | Current phishing URLs and a 30-day archive after approval; no benign class. | Current but access is time-limited and terms constrain redistribution. | Not acquired. | Optional confirmed legal path for a prospectively timestamped phishing arm; do not download without approval. |
| **Tranco / Common Crawl** | [Tranco](https://tranco-list.eu/) aggregates provider rankings with mixed upstream terms; [Common Crawl index](https://index.commoncrawl.org/) is accessible/free. | Ranking/crawl presence is not a benign label. | Tranco is mainly domain-level; Common Crawl URLs can add realistic paths/queries but crawled presence does not prove legitimate. | Not acquired. | Do not use alone as labelled negative data. Use only for sampling candidates after a separate, documented benign-label rule. |

### Manual acquisition gate for PhishVN

Download the archive presented on the official PhishVN v4 Mendeley page above;
do not bypass its access controls. Preserve the original archive name and its
SHA-256. Required contents are `dataset_url.csv`, the supplied
`splits/url_train.csv`, `splits/url_val.csv`, `splits/url_test.csv`,
`docs/datasheet.md`, `docs/column_schema.*`, and `MANIFEST.txt`. Verify every
data-file hash against `MANIFEST.txt`, record version `v4`/DOI, CC BY 4.0, and
the retrieval date before parsing. The Mendeley documentation says v4 data files
are byte-identical to v3 and documents the label audit; that claim must be
preserved with the archive, not treated as a substitute for local verification.

## Current overlap evidence and required new audit

P6 already measured development-source overlap. PHIUSIIL/MURL share 10,218
registered domains and 4,159 templates; MURL has 402 exact conflicting labels
against LegitPhish. URL-Phish and LegitPhish informed development, so neither
can become a final test. CompPhish was now consumed by P9 and is permanently
excluded from future selection.

No row-level PhishVN, new PhishTank, ROR, OpenPhish, Tranco, or Common Crawl
candidate files exist locally. Thus no honest exact/normalized URL, hostname,
registered-domain, or template overlap count can yet be reported for them.
Before assigning any candidate role, run the existing P6 normalizer, offline-PSL
domain extractor, and project template rule against the complete *approved
development corpus*. Exclude a candidate record if any of its exact raw URL,
conservatively normalized URL, hostname, registrable domain, or template key
appears in development. Quarantine exact or normalized URL label conflicts;
never majority-vote them. Keep an exclusion ledger with source row ID and every
matching rule, using a nonexclusive flag set plus one documented primary reason.

## Proposed new boundary

1. **Development corpus:** retain PHIUSIIL and conflict-quarantined MURL only
   as the historical baseline. Do not silently expand it. After approval,
   consider adding an overlap-cleaned, documented PhishVN *verified-core
   training slice* only in a separate experiment; preserve its supplied group
   and temporal split. Bronze data is a label-noise sensitivity stratum.
2. **Internal development:** build connected groups from registrable domain and
   project-normalized template, split by group, and report all source/class/
   structure strata. Retain an explicit source column and do not balance away
   its distribution without reporting the effect.
3. **Secondary external benchmark:** if PhishVN verified-core test rows remain
   overlap-free after the development corpus is fixed, freeze them *before* the
   next model iteration. Report them only as a Vietnamese, scheme-stripped
   benchmark and never use their results to tune the model.
4. **Broad final benchmark:** collect a prospective, date-bounded snapshot:
   verified PhishTank positives after a declared cutoff and legitimate URLs from
   a versioned, legally reusable registry plus real path/query URLs under a
   documented benign-label policy. Hash the raw snapshots, record timestamps,
   source IDs, licence/terms, and URL provenance. Freeze it before any training.
5. **Isolation:** exclude all exact/normalized URL, hostname, domain, and
   template overlap with development. Do not reuse CompPhish, URL-Phish,
   LegitPhish, or any P9 outcome to select a candidate.

This plan will reduce sample size and may change class balance. Those effects
must be measured after candidate files are acquired; they must not be offset by
relaxing overlap rules.

## Next step

Obtain PhishVN v4 manually and a legally documented prospective phishing and
legitimate snapshot, then run a row-level qualification audit before any new
model development. No retraining, external model evaluation, feature change,
or promotion is authorized from P10.
