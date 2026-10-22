# Final Showcase URL Classifier

## Scope

This controlled development run uses PHIUSIIL, conflict-screened MURL, and a byte-identical development copy of the previously evaluated CompPhish subset. The original frozen CSV, P6 manifest, and P9 result remain unchanged. CompPhish is development data in this run and is not an independent test.

All features are static lexical features extracted from raw URL strings. No HTML-derived fields, DNS, HTTP requests, or URL visits are used.

## Split integrity

The deterministic seed is 42. Train, validation, and test groups are connected components formed by registered domain and normalized URL template. Every pairwise overlap count for raw URL, normalized URL, registered domain, and template is zero; see `showcase_split_manifest.json`.

## Candidates

Thresholds maximize F1 on validation only. The grouped test partition is scored once after the validation-based selection.

| Candidate | Features | Validation F1 | Test F1 | Test ROC-AUC | Test PR-AUC | Test FPR | Test FNR |
|---|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression_p7_all_features | p7_all_features | 0.8732 | 0.8664 | 0.9355 | 0.9039 | 19.00% | 7.50% |
| RandomForest_p7_all_features | p7_all_features | 0.9041 | 0.9186 | 0.9795 | 0.9777 | 9.29% | 6.34% |
| LogisticRegression_p8_without_brand | p8_without_brand | 0.8730 | 0.8656 | 0.9353 | 0.9037 | 17.46% | 8.97% |
| RandomForest_p8_without_brand | p8_without_brand | 0.9045 | 0.9181 | 0.9799 | 0.9782 | 9.59% | 6.14% |

## Selected candidate and integration gate

Validation selection chose **RandomForest_p8_without_brand** using highest validation F1, then lower FPR and FNR. The served baseline was evaluated on the same grouped test partition for comparison only.

- PASS: test F1 0.9181 >= 0.80
- PASS: test ROC-AUC 0.9799 >= 0.85
- PASS: test FPR 0.0959 <= 0.20
- PASS: test FNR 0.0614 <= 0.20
- PASS: test F1 improvement 0.0746 >= 0.05
- PASS: test FPR reduction 0.2361 >= 0.05

**Integrated:** True. All predeclared grouped internal acceptance criteria passed.

## Legitimate URLs with paths and queries

| Subgroup | Legitimate rows | False positives | FPR |
|---|---:|---:|---:|
| nonempty_path | 9118 | 2496 | 27.37% |
| query_parameter | 1216 | 291 | 23.93% |
| path_or_query | 9118 | 2496 | 27.37% |

## PhishVN diagnostic

PhishVN v3.1.0 is verified against its internal manifest and is accepted as the official v4 documentation-only-equivalent archive. It is **not scored here**: the raw URL field is scheme-stripped, and treating missing schemes as HTTP would invent a feature value for `is_https`. Its earlier structural analysis and verified-core/bronze separation are retained in `reports/p11/p11_phishvn_archive_qualification.json`.

## Limits

This is grouped internal development evidence, not a new independent benchmark. P9's historical external failure remains valid for its P7 model and cannot be compared directly with the new candidate. The URL classifier remains a college-project prototype rather than a production security control.
