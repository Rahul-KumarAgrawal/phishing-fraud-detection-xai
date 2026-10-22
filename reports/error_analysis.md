# Error Analysis

**Project:** AI-Based Fraud and Phishing Detection with Explainable AI
**Course:** CSET312, Bennett University

---

## Overview

This document records the measured error patterns of the two deployed models.
Every number below comes from an actual evaluation run; re-running
`scripts/evaluate_message_model.py` and `scripts/evaluate_url_model.py`
reproduces them.

Positive class = **phishing / smishing**. Confusion matrices are written as
`[[TN, FP], [FN, TP]]`.

---

## Branch A – Message Classifier

### Status: measured

**Test set:** 820 messages (726 ham / 94 smishing) — 15% of the real Mendeley
SMS/Smishing dataset after excluding 489 `spam` rows and 20 duplicates.

**Model:** `LinearSVC` (wrapped in `CalibratedClassifierCV`), threshold 0.6181.

| Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|
| 0.9894 | 0.9894 | 0.9894 | 1.0000 |

Confusion matrix: `[[725, 1], [1, 93]]` — **1 false positive, 1 false negative**.

### Observed errors

| Error Type | Count | Example | Probability | Interpretation |
|---|---|---|---|---|
| False Negative (smishing → ham) | 1 | *"Dorothy@kiefer.com (Bank of Granite issues Strong-Buy) EXPLOSIVE PICK FOR OUR MEMBERS \*\*\*\*\*UP OVER 300%... Nasdaq Symbol CDGT"* | 0.3584 | This is a stock-promotion **spam** message. It is labelled `smishing` in the source data but contains no phishing mechanics (no credential harvesting, no link to a fake login). This is most likely a **label-noise** case in the source dataset, not a model failure. |
| False Positive (ham → smishing) | 1 | *"We have sent JD for Customer Service cum Accounts Executive to ur mail id, For details contact us"* | 0.6818 | A legitimate business message, but it shares vocabulary with smishing (`contact us`, `ur mail id`, a named colleague). The calibrated score (0.68) is only just above the 0.6181 threshold, so the model is appropriately uncertain here. |

### Caveats

- ROC-AUC of exactly 1.0000 on 820 samples is a small-sample effect; with only
  94 positives, a single reordered pair changes the third decimal.
- The source dataset is known to overlap the SMS Spam Collection and to contain
  label noise, so a small number of errors are expected to be irreducible without
  manual re-labelling.

### Binary Classification Policy — Exclusion Note

Generic **spam** messages (489 rows) are excluded from the binary experiment.
Reason: generic spam is not smishing. Treating all spam as the phishing positive
class would inflate the positive class with messages that contain no phishing
content — exactly the error the excluded row above illustrates. The `spam`
category is documented in `reports/dataset_inventory.md`.

---

## Branch B – URL Classifier

### Status: measured

Three test sets are reported, each scored by a model that is permitted to see it.
See `reports/url_metrics.json` and `reports/url_leakage_report.json`.

| Test set | Model | N | FP | FN | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Random | production (random split) | 35,306 | 11 | 72 | 0.9993 | 0.9952 | 0.9972 | 0.9982 |
| Domain-aware | domain-holdout | 38,591 | 6 | 77 | 0.9997 | 0.9958 | 0.9977 | 0.9986 |
| Template-clean | domain-holdout | 38,342 | 6 | 77 | 0.9997 | 0.9957 | 0.9977 | 0.9985 |

### The dominant error pattern: the model has learned a dataset shortcut

**Every one of the 249 misclassified URLs is a bare `https://host` with no path
and no query.** Measured over all false cases (`reports/url_false_cases.csv`):

| Group | n | `is_https` | `slash_count`=2 | `path_length`=0 | mean hostname length | mean `suspicious_token_count` |
|---|---|---|---|---|---|---|
| All false negatives (phishing missed) | 226 | 1.000 | 100.0% | 100.0% | 20.9 | 0.066 |
| All false positives (legit flagged) | 23 | 1.000 | 100.0% | 100.0% | 26.7 | 0.000 |
| Training: legitimate | 2,318 | 1.000 | 100.0% | 100.0% | 19.4 | 0.006 |
| Training: phishing | 1,682 | 0.483 | 40.7% | 40.7% | 23.9 | 0.115 |

In PHIUSIL **every legitimate URL is a bare `https://domain` with no path**, so
`is_https` (1.000 vs 0.483) and `slash_count` (2.000 vs 3.033) almost perfectly
separate the classes. Once a URL is a bare `https://host`, those features are
constant and the decision is effectively made by **hostname length and
digit/hyphen content** — which is why *both* error groups sit at the same
structural point and are separated only by hostname length (20.9 vs 26.7, training
means 19.4 vs 23.9).

### Observed errors

| Error Type | Cause | Examples | Mitigation |
|---|---|---|---|
| False Positive (legit → phishing) | Legitimate, digit-heavy or long hostnames look like phishing once `is_https`/`slash_count` are fixed | `https://www.180360.com` (0.988), `https://www.112112ww.com` (0.964), `https://www.centregranger.cnrs.fr` (0.965) | Train on data where legitimate URLs include paths/HTTP; add calibration features; stop relying on `is_https` |
| False Negative (phishing → legit) | Real typosquats with short hostnames and no path match the "bare HTTPS domain" signature of the legitimate class | `https://www.netflix-vn.com` (0.0099), `https://www.colis-dhl.com` (0.000), `https://www.amazon.ydoxt.xyz` (0.426), `https://www.icloud-mapfind.com` (0.004) | Brand-impersonation and lookalike-domain features (follow-up work); these URLs are branded but the model has no brand concept |
| URL shortener mis-classification | Shortener URLs have few extractable lexical features | e.g. `https://www.ur.link` misclassified as legitimate | `is_url_shortener` flag exists but the underlying signal is weak for bare shortener domains |
| Domain leakage | Previously, the "domain-aware" test set was scored with a random-split model sharing 32,826 URLs and 86.2% of templates | — | **Fixed**: unseen-domain figures now come from a model trained only on the domain-isolated pool |

### Domain Leakage Note

Random-split metrics and unseen-domain metrics are reported separately in
`reports/url_metrics.json` and **are not interchangeable**:

- **Random split** — in-distribution; domains and templates are shared with
  training. Estimates serving-time accuracy on familiar patterns.
- **Domain-aware** — registered domains disjoint from the training pool.
- **Template-clean** — additionally, no shared URL path template with training.

The high F1 **survives** the domain-isolated and template-clean tests, which
establishes that the score is not primarily a leakage artefact. It is an artefact
of the dataset's URL-format shortcut described above. That distinction matters:
the model is not memorising test URLs, but it is exploiting a shortcut that will
not transfer to real traffic.

---

## SHAP Explanation Limitations

- SHAP values show *model attribution*, not causal proof.
- A high `suspicious_token_count` does not prove a URL is phishing.
- Legitimate URLs (e.g. Google login, PayPal) contain suspicious tokens.
- Message explanations use `KernelExplainer` on the **calibrated**
  `predict_proba`, so `base + Σshap` equals the displayed `P(smishing)` exactly
  (measured gap `0.0`). The cost is that the values are a sampled approximation
  rather than an exact decomposition, and the SHAP baseline is an empty message
  (`P(smishing) = 0.0023` for the served model).
- URL explanations use `TreeExplainer` (exact, probability units, additive to
  `P(phishing)`) for the served Random Forest, or `LinearExplainer` (exact,
  **log-odds** units) for a logistic model. The payload states which, in `units`
  and `explained_output`; log-odds contributions are not probability
  contributions.
- No faithfulness/robustness evaluation of the explanations themselves
  (e.g. deletion curves) is included.
- For URLs, the top SHAP contributors include `is_https` and `slash_count`, which
  is the explanation surfacing the shortcut — an honest signal that the model's
  reasoning is degenerate on bare-URL inputs.

---

## Known Bugs / Open Issues

| Issue | Status | Notes |
|---|---|---|
| PHIUSIL labels read backwards (1=legitimate treated as 1=phishing) | **Fixed** | Every metric and API label was inverted. Now asserted in `prepare_urls.py`, guarded by `tests/test_url_split_leakage.py` |
| Domain-aware test scored with a random-split model (leakage) | **Fixed** | 32,826 shared URLs / 86.2% shared templates. Now quantified in `reports/url_leakage_report.json`; unseen-domain figures use `models/url_model_domain_holdout.joblib` |
| Message model trained on a 65-message synthetic placeholder | **Fixed** | Real Mendeley dataset now downloaded, SHA-256 verified, composition asserted; `dataset_provenance` recorded in metadata and exposed via `/api/health`; the trainer refuses synthetic data without `--allow-synthetic` |
| Mendeley download required browser auth | **Fixed** | The old hardcoded file UUID 404'd. The downloader now resolves the archive from the Mendeley public API and falls back to `urllib` when Cloudflare blocks `requests` |
| Message SHAP explains the uncalibrated SVM, not the calibrated probability | **Fixed** | The explainer now runs `KernelExplainer` on the calibrated `predict_proba` and resolves the class from `classes_`. Previously it explained fold 1 of 3 uncalibrated `LinearSVC`s: `base + Σshap` was off the displayed score by 1.1557 (ham sample) / 0.0662 (smishing sample). Now exact (gap 0.0). Regression test: `test_message_explanation_does_not_explain_the_uncalibrated_svm` |
| SHAP contributions are surfaced in a field named `value` | **Fixed** | `FeatureContribution` is now `{feature, shap_value, feature_value, direction, description}`; `value` is gone. The message path previously printed the SHAP value in the feature-value slot and rendered every contribution bar at width 0. See `reports/explainability_fixes.md` |
| `url_features.py` docstring says 28 features / Group 2 = 14 | **Fixed** | Docstring corrected; schema is now 33 features (30 + 3 brand-impersonation features) |
| Duplicated URL-shortener condition in the feature extractor | **Fixed** | The `any(regd_lower == s ...)` clause was redundant with the membership test and has been removed |
| No model-artifact feature-schema check on load | **Fixed** | `FEATURE_SCHEMA_VERSION` and a `feature_schema_fingerprint` (hash of the canonical schema) are written into every metadata file, and `_assert_schema()` is strict: empty metadata, missing required keys, unknown names, wrong order, duplicates, a `n_features` that disagrees with the listed names, an undeclared feature subset, a pipeline-width mismatch, a stale version or a stale fingerprint all raise. A feature *inserted* mid-schema — which the old order check missed, because it shifts every later column while leaving the recorded order intact — is caught by the subset rule. See `reports/p5_artifact_safeguards.md` |
| A model artifact could load with no metadata at all | **Fixed** | `load_model()` / `load_revised_model()` / `load_domain_holdout_model()` returned `(pipeline, {})` when the `*_metadata.json` was missing, so every caller silently fell back to `threshold = 0.5` and the full feature schema (the 24-feature revised candidate would have been fed the wrong columns). The metadata file is now required, and the API registry marks the model unloaded so `/api/analyze/*` returns 503 instead of serving it |
| Message artifacts had no load-time guard | **Fixed** | The metadata must now carry `model_name`, `threshold` (a probability in (0, 1)), `label_map` and `dataset_provenance`; `feature_type` must equal the shared `FEATURE_SPEC` constant; and the pipeline must expose `predict_proba` plus the `features` / `clf` steps |
| URL model learned a dataset-format shortcut ("HTTPS **and** a path -> phishing") | **Partially fixed** | The conjunction occurred *only* in the phishing class in training data (0 legitimate vs 32,780 phishing rows), because PHIUSIL's legitimate class is entirely bare `https://domain` and the MURL augmentation source carries no scheme. Diagnosed in `reports/url_diagnosis.md`, ablated in `reports/url_feature_ablation.md`. The revised candidate cuts the false-positive rate on the 445 new legitimate URLs from 100.0% to 4.0%, but is **not promoted** |
| URL model does not generalise to a genuinely independent corpus | **Open** | On the untouched external set (URL-Phish) ROC-AUC is 0.5496 (served) / 0.5329 (revised) - near chance - with false-positive rates of 94.07% / 66.06%. The dominant feature `is_https` is inverted between corpora (legitimate HTTPS rate 100% in PHIUSIL vs 17.9% in URL-Phish). See `reports/url_remediation.md` §4 and §6. **Not re-evaluated in P4; that corpus is now development-informed, so a fresh independent set is reserved for the final claim** |
| Word and character n-grams shared a raw feature name in explanations | **Fixed** | e.g. the word feature `no` and the char 2-gram `no` are different TF-IDF columns with the same reported name. Display names are now block-prefixed (`word__no` / `char__no`) and uniqueness is asserted in `tests/test_explanation_consistency.py` |
| Message SHAP fallback mis-unpacked the multi-class SHAP shape | **Fixed** | The old fallback indexed `sv[0]` on a `(1, n_features, 2)` array, producing a 2-D contribution vector. `shap_utils.select_class_shap_values()` now handles every documented shape and is unit-tested |
| Installed dependency versions drift from `requirements.txt` | **Open** | e.g. scikit-learn 1.8.0 vs pinned 1.5.1, pandas 3.0.1 vs 2.2.2, shap 0.51.0 vs 0.45.1. Reproducibility risk |
| PHIUSIL dataset requires ucimlrepo or manual download | Open | Automated via `ucimlrepo`; falls back to manual instructions |
| tldextract `registered_domain` deprecation | Fixed | Uses `top_domain_under_public_suffix` with fallback |
| FastAPI `on_event` deprecation | Fixed | Migrated to `lifespan` context manager |

---

*Last updated: 2026-10-19. Generated from evaluation runs on the real datasets.
The SHAP rows reflect the P4 fixes in `reports/explainability_fixes.md`; the
artifact-guard rows reflect the P5 work in `reports/p5_artifact_safeguards.md`.*
