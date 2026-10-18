# Error Analysis

**Project:** AI-Based Fraud and Phishing Detection with Explainable AI
**Course:** CSET312, Bennett University

---

## Overview

This document records known error patterns, false positives, and false negatives
observed during model development. It will be updated after each training run.

> **Note:** This document is generated from actual model evaluation.  
> If models have not yet been trained, sections marked `[PENDING]` will be populated
> after running `scripts/evaluate_message_model.py` and `scripts/evaluate_url_model.py`.

---

## Branch A – Message Classifier

### Status: [PENDING training]

Expected error categories once trained:

| Error Type | Expected Cause | Mitigation |
|---|---|---|
| False Positive (ham → smishing) | Ham messages containing phishing-like words (e.g., "verify", "account") | Threshold tuning; recall/precision trade-off |
| False Negative (smishing → ham) | Novel phishing language not seen in training | Dataset diversity; re-labelling review |
| Short message misclassification | Very short messages (<5 tokens) have sparse TF-IDF representations | Minimum-length filter |

### Binary Classification Policy — Exclusion Note

Generic **spam** messages are excluded from the baseline binary experiment.  
Reason: Spam label definitions vary by dataset version; treating all spam as phishing
would inflate the positive class and produce misleading metrics.  
The `spam` category is documented in `reports/dataset_inventory.md`.

---

## Branch B – URL Classifier

### Status: [PENDING training]

Expected error categories:

| Error Type | Expected Cause | Mitigation |
|---|---|---|
| False Positive (legit → phishing) | Legitimate URLs with many query parameters, hyphens, or long paths | Threshold tuning; domain-aware evaluation |
| False Negative (phishing → legit) | Short, simple phishing URLs that mimic legit structure | Feature enrichment; future: DNS/WHOIS features (not in baseline) |
| URL shortener mis-classification | Shortener URLs have few extractable features | `is_url_shortener` flag added to feature set |
| Domain leakage | Repeated domains across train/test skews random-split metrics | Domain-aware split reported separately |

### Domain Leakage Note

Random-split metrics and domain-aware test metrics are reported separately
in `reports/url_metrics.json`. They are **not interchangeable**:

- Random split: measures in-distribution performance
- Domain-aware split: measures generalisation to unseen registered domains

---

## SHAP Explanation Limitations

- SHAP values show *model attribution*, not causal proof.
- A high `suspicious_token_count` does not prove a URL is phishing.
- Legitimate URLs (e.g., Google login, PayPal) contain suspicious tokens.
- LinearExplainer is used for linear models; TreeExplainer for Random Forest.
- KernelExplainer fallback is approximate and slower.

---

## Known Bugs / Open Issues

| Issue | Status | Notes |
|---|---|---|
| Mendeley dataset requires browser auth for direct download | Open | Manual download required; instructions in `download_data.py` |
| PHIUSIL dataset requires ucimlrepo or manual download | Open | Automated via `ucimlrepo`; falls back to manual instructions |
| tldextract `registered_domain` deprecation | Fixed | Uses `top_domain_under_public_suffix` with fallback |
| FastAPI `on_event` deprecation | Fixed | Migrated to `lifespan` context manager |

---

*Last updated: 2026-10-18. Re-run evaluation scripts to update with measured results.*
