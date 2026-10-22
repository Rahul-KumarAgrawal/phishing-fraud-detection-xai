# Implementation Checklist
**Project:** AI-Based Fraud and Phishing Detection with Explainable AI  
**Course:** CSET312, Bennett University  
**Repository:** https://github.com/Rahul-KumarAgrawal/phishing-fraud-detection-xai

---

## Stage 0 – Repository Audit
- [x] Inspected existing files (only stub README.md + .git found)
- [x] Verified Python version: 3.11.9
- [x] Verified Node version: 22.19.0
- [x] Git history reviewed (1 initial commit)
- [x] No existing models, datasets, or application code found

## Stage 1 – Project Structure & Environment
- [x] Directory tree created
- [x] `__init__.py` files created for all packages
- [x] `.gitignore` created
- [x] `requirements.txt` created with pinned versions
- [x] `pip install` completed (note: the development environment has since drifted from the pins — see `reports/limitations.md` → Reproducibility)

## Stage 2 – Dataset Acquisition & Inspection
- [x] `src/fraud_phishing/data/download_data.py` created
- [x] `src/fraud_phishing/data/inspect_data.py` created
- [x] SMS/smishing dataset downloaded (Mendeley, 5,971 rows, SHA-256 verified)
- [x] URL dataset downloaded (UCI PHIUSIL, 235,795 rows)
- [x] `reports/dataset_inventory.md` created

## Stage 3 – Message Data Preparation
- [x] `src/fraud_phishing/data/prepare_messages.py` created
- [x] Labels mapped: ham→0, smishing→1 (spam excluded; label casing normalised)
- [x] Duplicates removed before splitting
- [x] Train/val/test splits saved to `data/processed/`
- [x] Dataset provenance (`real` / `synthetic`) detected and recorded

## Stage 4 – URL Data Preparation
- [x] `src/fraud_phishing/data/prepare_urls.py` created
- [x] Raw URL strings extracted, labels normalized (legit→0, phishing→1)
- [x] Raw PHIUSIL label convention verified against the official dataset card
- [x] Random split saved to `data/processed/`
- [x] Domain-isolated and template-clean splits built, with a leakage report

## Stage 5 – URL Feature Extraction
- [x] `src/fraud_phishing/features/url_features.py` created
- [x] `tests/test_url_features.py` created
- [x] Feature schema: **33** named features, deterministic, immutable order
- [x] Edge cases handled: malformed, IP, Unicode, missing scheme

## Stage 6 – Message Classification Pipeline
- [x] `src/fraud_phishing/models/message_model.py` created
- [x] `scripts/train_message_model.py` created
- [x] `scripts/evaluate_message_model.py` created
- [x] 3 models compared: LR, LinearSVC, MultinomialNB
- [x] Model selected based on F1/AUC on val set
- [x] `models/message_model.joblib` saved

## Stage 7 – URL Classification Pipeline
- [x] `src/fraud_phishing/models/url_model.py` created
- [x] `scripts/train_url_model.py` created
- [x] `scripts/evaluate_url_model.py` created
- [x] 2 models compared: LR, Random Forest
- [x] Domain-aware split built with verified zero domain overlap
- [x] Domain-holdout model trained for honest unseen-domain evaluation
- [x] `models/url_model.joblib` saved

## Stage 8 – SHAP Explanations
- [x] `src/fraud_phishing/explain/message_explainer.py` created
- [x] `src/fraud_phishing/explain/url_explainer.py` created
- [x] `tests/test_explanations.py` created
- [x] SHAP values verified against actual fitted models
- [x] Message SHAP aligned to the calibrated probability (`KernelExplainer` on
      the fitted `CalibratedClassifierCV`'s `predict_proba`; stage 16)

## Stage 9 – FastAPI Backend
- [x] `src/fraud_phishing/api/main.py` created
- [x] `src/fraud_phishing/api/schemas.py` created
- [x] `src/fraud_phishing/api/services.py` created
- [x] `tests/test_api.py` created
- [x] GET /api/health works (now also reports training-data provenance)
- [x] POST /api/analyze/message works
- [x] POST /api/analyze/url works

## Stage 10 – React + Tailwind Frontend
- [x] Vite + React project bootstrapped in `frontend/`
- [x] Message analysis tab implemented
- [x] URL analysis tab implemented
- [x] Results rendered from live backend
- [x] `frontend/README.md` created
- [x] Prototype banner shown when the message model was trained on non-real data

## Stage 11 – Integration Testing
- [x] All pytest tests pass (`113 passed, 2 skipped, 1 xfailed`; the 2 skips are
      intentional and the xfail records a measured limitation)
- [x] Frontend builds successfully (`vite build` → 20 modules, 283ms)
- [x] End-to-end demo verified (health, message analysis, URL analysis via TestClient)

## Stage 12 – Documentation
- [x] `reports/implementation_checklist.md` (this file)
- [x] `reports/dataset_inventory.md`
- [x] `reports/model_comparison.csv`
- [x] `reports/error_analysis.md` (populated with measured results)
- [x] `reports/limitations.md`
- [x] `reports/url_leakage_report.json`
- [x] `reports/url_false_cases.csv`
- [x] `reports/url_remediation.md` (P3 findings, experiments and recommendation)
- [x] `reports/url_model_comparison.md` (original vs revised, all datasets)
- [x] `README.md` updated with full setup instructions and measured results

## Stage 13 – Evaluation Integrity Remediation

- [x] Inverted PHIUSIL labels corrected and asserted on every run
- [x] Message branch retrained on the real Mendeley dataset
- [x] Dataset provenance exposed via `/api/health` and the UI
- [x] Trainer refuses synthetic data unless `--allow-synthetic`
- [x] Evaluation leakage quantified (`reports/url_leakage_report.json`)
- [x] Split-integrity tests added (`tests/test_url_split_leakage.py`)
- [x] Unseen-domain evaluation moved to a domain-isolated model
- [x] Stale 0.998 claim replaced with per-split measured metrics
- [x] URL feature improvements: 33 features incl. brand impersonation, lookalike
      domains and a bounded Levenshtein check (`url_features.py`)
- [x] Message SHAP aligned to the calibrated probability
- [x] SHAP contributions renamed out of the misleading `value` field
      (`shap_value` + `feature_value`, plus `units`, `explained_output`,
      `base_value`, `additivity_gap`)
- [x] Model-artifact feature-schema version check on load (`_assert_schema`, which
      also compares the pipeline's `n_features_in_` against the metadata)

## Stage 14 – External Dataset Evaluation (LegitPhish)

- [x] LegitPhish added as Dataset C in `download_data.py` (SHA-256 verified)
- [x] `prepare_external_urls.py` created; label convention asserted, overlap with
      PHIUSIL measured; evaluation-only
- [x] `evaluate_url_model.py` scored both models on 100,872 external URLs
- [x] Recall on 62,482 unseen-domain phishing URLs measured (1.0000, incl. HTTPS-only)
- [x] Legitimate class split by byte-identity to expose memorisation
- [x] Precision weakness quantified: 100% FPR on the 445 new legitimate URLs
- [x] `reports/external_evaluation.md` written
- [x] `reports/dataset_inventory.md` regenerated with Dataset C
- [x] Retrained with the formatting shortcut removed — see Stage 15

## Stage 15 – URL Robustness (P3)

- [x] Step 1 — diagnosis reproduced and root cause identified:
      **"HTTPS AND a non-empty path" occurred only in the phishing class**
      (0 legitimate vs 32,780 phishing rows); 100.00% FPR on the 445 new
      legitimate URLs, mean P(phishing) 0.9998
- [x] `scripts/url_feature_diagnosis.py`; `reports/url_diagnosis.md`,
      `url_diagnosis_false_positives.csv`, `url_feature_distributions.csv`
- [x] Step 2 — controlled ablation (`scripts/ablate_url_features.py`,
      10 subsets × 2 pools × 2 model families). Data augmentation alone does
      **not** fix the defect (RF still 100%); path/query features must go
- [x] Scheme-normalisation experiment run and **rejected** (it fixes trees but
      breaks the linear model, and would create a "HTTP implies phishing"
      shortcut); no schemes were fabricated
- [x] Datasets D (MURL, augmentation) and E (URL-Phish, final external) wired into
      `download_data.py` with SHA-256 verification and composition assertions
- [x] Step 3 — domain- and template-aware splits; `prepare_urls_augmented.py`
      builds a domain-isolated augmented pool; URL-Phish reserved and untouched
- [x] Step 4 — `scripts/compare_url_models.py`; `reports/url_model_comparison.md`
      and `reports/url_remediation.md` (P/R/F1, confusion matrices, FPR, FNR,
      domain- and template-held-out, all four datasets, integrity strata)
- [x] Step 5 — regression tests added; suite is `113 passed, 2 skipped, 1 xfailed`
- [x] Honest negative finding recorded: on the untouched external set both models
      are near chance (ROC-AUC 0.5496 / 0.5329), so the revised candidate is
      **not promoted** and the URL classifier is documented as a research
      prototype with a measured generalisation failure
- [ ] Promote the revised URL model (awaiting approval)
- [ ] Obtain a legitimate URL corpus that mixes HTTP/HTTPS with realistic paths
      and subdomains (prerequisite for a real fix)

## Stage 17 – Correctness Cleanup & Artifact Safeguards (P5)

- [x] Verified the `url_features.py` docstring counts (already correct since P2)
      and pinned them with `test_docstring_feature_counts_match_the_schema`,
      which cross-checks the total, the per-group breakdown and the group markers
      in the tuple itself
- [x] Verified the redundant URL-shortener clause was already removed in P2
      (`is_url_shortener = int(regd_lower in URL_SHORTENERS)`); behaviour unchanged
- [x] Metadata is now **required** for every URL artifact — a missing or empty
      file no longer loads with a silent `threshold = 0.5` fallback
- [x] `_assert_schema` additionally rejects missing required keys, duplicate or
      unknown names, a `n_features` disagreeing with the name list, an
      undeclared feature subset, a pipeline-width mismatch, a stale schema
      version and a stale `feature_schema_fingerprint`
- [x] `feature_schema_fingerprint()` (SHA-256 of the canonical schema) is stamped
      by every `save_*` function, so a mid-schema feature insertion is caught even
      if `FEATURE_SCHEMA_VERSION` is not bumped
- [x] Message artifacts gain `_assert_artifact`: required metadata keys, a
      probability threshold in (0, 1), `feature_type == FEATURE_SPEC`, and the
      `features` / `clf` / `predict_proba` pipeline contract
- [x] Confirmed no backend, frontend, test or documented example depends on the
      removed explanation `value` field (only labelled "before" payloads and
      absence assertions remain)
- [x] Suite is `183 passed, 2 skipped, 1 xfailed`; `npm run lint` clean;
      `npm run build` succeeds; all `models/*` artifacts byte-identical
- [x] Documentation updated: `README.md`, `reports/limitations.md`,
      `reports/error_analysis.md`, `reports/url_remediation.md` (diagnosis-time
      schema clarified); `reports/p5_artifact_safeguards.md` written
- [ ] Promote the revised URL model (awaiting approval)

---

*Last updated: 2026-10-19*

## Stage 20 – College-project readiness (2026-10-20)

- [x] Scoped the project as an educational prototype and deferred P11's
      prospective benchmark collection; no new datasets or manually labelled
      records are required for the demonstration.
- [x] Preserved P9's frozen CompPhish files and report. Its external failure
      (P7 FPR 98.46%, ROC-AUC 0.3216) is shown in the README, limitations, and
      college report; no candidate was promoted.
- [x] Kept the served artifacts unchanged at this stage: `models/message_model.joblib`
      and `models/url_model.joblib`. Later showcase integration is recorded in
      Stage 21.
- [x] Added a persistent prototype warning in the frontend and OpenAPI
      description. Updated the URL API documentation to the actual 33-feature
      schema.
- [x] Fixed static suffix parsing to use tldextract's bundled suffix list and a
      repository-local cache. This removes network/cache-lock dependency from
      URL feature extraction without changing feature names, ordering, models,
      or thresholds.
- [x] Verification on 2026-10-20: `npm run lint` passed; `npm run build`
      passed (20 modules); `tests/test_url_features.py` passed (61); and
      `tests/test_model_artifact_guards.py` passed (47). The non-HTTP part of
      `tests/test_explanation_consistency.py` passed (20; two API payload tests
      are environment-limited by TestClient/lifespan execution in this managed
      session).
- [ ] Run the live local demo on the user's machine after launching it. This
      environment blocks localhost sockets, so browser/API smoke calls could not
      be completed here.

*Last updated: 2026-10-20*

## Stage 21 – Final showcase URL integration (2026-10-20)

- [x] Built a PHIUSIIL, conflict-screened MURL, and byte-identical CompPhish
      development corpus. It has deterministic train/validation/test connected
      domain/template groups with zero exact, normalized, domain, and template
      overlap across every split pair.
- [x] Preserved P9's frozen CSV, manifest, metrics, and report. The previous P9
      result remains historical evidence for P7; CompPhish is now development
      data and is not an independent result for the new model.
- [x] Compared Logistic Regression and Random Forest under P7's full 33-feature
      schema and P8's declared 30-feature `without_brand` subset. Thresholds
      were selected on validation only.
- [x] Selected RandomForest/P8-without-brand by validation F1. On the grouped
      internal test: precision 0.8985, recall 0.9386, F1 0.9181, ROC-AUC
      0.9799, PR-AUC 0.9782, FPR 9.59%, and FNR 6.14%.
- [x] Integrated the selected model after all predeclared internal acceptance
      criteria passed. Current served hash:
      `69b1476c670c07a38b3368285ea5e60fc2295d0bf84a3ee570eb87f64dba011e`.
      Rollback artifact: `models/url_model_pre_showcase.joblib` with hash
      `0366b8c528523e47bf447fa9c6051f4f6558317e6fc68d00ea30e88e850d834c`.
- [x] Did not score PhishVN verified core: scheme-stripped URLs would require
      invented HTTP/HTTPS feature values. Its verified-core/bronze audit and
      structural limitations remain in `reports/p11/`.
- [x] Focused integration, artifact, P6/P9 preservation, and URL-SHAP tests:
      112 passed, 27 deselected. Frontend lint and production build passed.

*Last updated: 2026-10-20*

## Stage 19 – Frozen external evaluation and dataset remediation (P9–P10)

- [x] Evaluated P7 Random Forest once at its saved threshold on the frozen,
      overlap-excluded CompPhish subset; P7 is not promotable (98.46% FPR)
- [x] Preserved the P9 report, metrics, frozen CSV, manifest, and hashes
- [x] Performed a read-only, aggregate-only P9 integrity review; no CompPhish
      rows, model predictions, or P8 evaluation were used in P10
- [x] Documented PHIUSIIL/MURL structural coverage gaps and P8 source disparity
- [x] Qualified PhishVN v4 verified core and prospective-source requirements;
      documented manual acquisition and overlap gates
- [ ] Acquire and row-level-audit a new source-separated benchmark before any
      further URL model development (awaiting approval)

## Stage 18 – Controlled URL baseline (P7)

- [x] Built PHIUSIL + conflict-quarantined MURL development partitions without
      reading the frozen CompPhish evaluation CSV
- [x] Enforced connected registered-domain and normalized-template isolation;
      exact URL, normalized URL, domain and template overlap are all zero
- [x] Trained separate LogisticRegression and RandomForest candidate artifacts;
      neither replaced the served URL model or revised candidate
- [x] Saved thresholds, feature-schema metadata, hashes and internal validation
      metrics in `reports/p7/p7_internal_validation.json`
- [x] Added P7 candidate/split regression tests; focused P6/P7 suite passed
- [ ] Approve any external evaluation or promotion (awaiting review)

## Stage 16 – Explanation Correctness (P4)

- [x] Message SHAP re-anchored to the **calibrated** `predict_proba`
      (`KernelExplainer`, `l1_reg=0`); additivity gap measured at `0.0` instead
      of 1.1557 off the displayed score
- [x] Positive class resolved from `classes_` (`explain/shap_utils.py`) instead of
      a hard-coded column index
- [x] `FeatureContribution` is `{feature, shap_value, feature_value, direction,
      description}`; the ambiguous `value` field is removed and the payload now
      states `units`, `explained_output`, `base_value`, `additivity_gap`
- [x] Feature names are block-prefixed (`word__…` / `char__…`) so word and
      character n-grams that share a raw name are distinguishable
- [x] `ExplanationPanel.jsx` shows the feature value and the SHAP contribution as
      separate, labelled numbers, with the units and baseline in the header
- [x] `tests/test_explanation_consistency.py` (22 tests): probability mapping for
      both classes, class alignment, additivity, direction signs, feature-value
      display, API payload has no `value` field
- [x] Suite is `135 passed, 2 skipped, 1 xfailed`; `npm run lint` clean;
      `npm run build` succeeds
- [x] `reports/explainability_fixes.md` written
- [x] P5 cleanups and artifact safeguards completed separately — see Stage 17
- [ ] Promote the revised URL model / further implementation phases (awaiting approval)

---

*Last updated: 2026-10-19*
