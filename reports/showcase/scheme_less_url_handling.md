# Scheme-less URL handling audit

## Change

`normalize_url_for_parsing()` now represents a scheme-less input such as `www.example.com/path?q=1` as `//www.example.com/path?q=1` for parsing. This is deterministic and idempotent. It allows `urllib.parse` to extract the host, path, and query without inserting an HTTP scheme. The submitted URL remains the API response's `url` value and all lexical counts still use that submitted text.

`is_https` remains `1` only when the user explicitly supplied `https://`. An absent scheme is not represented as HTTPS or as an asserted HTTP transport. The existing 30-feature served model has no feature that distinguishes an absent scheme from explicit HTTP, so both retain `is_https = 0`.

## Measured served-model results

Threshold: `0.5143261183261184`; score: `P(phishing)`.

| Input | Parse representation | `is_https` | Score | Result |
|---|---|---:|---:|---|
| `www.google.com` | `//www.google.com` | 0 | 0.883486 | phishing |
| `google.com` | `//google.com` | 0 | 0.228345 | legitimate |
| `https://www.google.com/` | unchanged | 1 | 0.310936 | legitimate |
| `http://www.google.com/` | unchanged | 0 | 0.660485 | phishing |
| `https://www.google.com/search?q=cybersecurity` | unchanged | 1 | 0.355917 | legitimate |
| `https://www.wikipedia.org/` | unchanged | 1 | 0.079864 | legitimate |
| `https://www.wikipedia.org/wiki/Phishing` | unchanged | 1 | 0.518901 | phishing |
| `http://paypal-secure-login.example/verify?urgent=true` | unchanged | 0 | 0.874530 | phishing |

The false positive for `www.google.com` remains. SHAP attributes substantial risk to the no-HTTPS value; the same model also classifies explicit HTTP Google as phishing. This is a learned data/model limitation, not an incorrect Google or Wikipedia parsing result. A URL's host name alone is not proof of safety, and no domain allowlist or destination request was added.

## Isolated retraining candidate

`scripts/train_scheme_less_url_candidate.py` trained an unserved Random Forest candidate using the existing grouped showcase partitions. It keeps the existing P8 feature order as a prefix and appends `has_explicit_scheme`. It was limited to deterministic balanced diagnostic samples (12,000 fit, 4,000 validation, 4,000 test), so its results are not comparable to the final showcase's full test report.

- Validation: F1 0.9072; ROC-AUC 0.9716; FPR 8.30%; FNR 10.10%.
- Grouped test sample: F1 0.8994; ROC-AUC 0.9758; FPR 6.95%; FNR 12.60%.
- `www.google.com`: score 0.970792, phishing.

The candidate does not remedy the reported case and was **not integrated**. Its artifact and metadata are separate from the served and rollback artifacts; details are in `scheme_less_candidate_results.json`.

## Validation

Feature tests cover scheme-less domains, explicit HTTP/HTTPS, paths, queries, and idempotent normalization. Explanation tests verify that the scheme-less input's score, threshold label, feature values, class mapping, and SHAP additivity remain consistent.
