# Limitations

**Project:** AI-Based Fraud and Phishing Detection with Explainable AI
**Course:** CSET312, Bennett University

---

## Dataset Limitations

### Dataset A – SMS/Smishing (Mendeley)

- ~5,971 messages total; a small dataset by NLP standards.
- Requires manual browser download (Mendeley authentication wall).
- Generic `spam` category excluded from binary experiment to avoid label ambiguity.
- No temporal split; model may overfit to the vocabulary of a specific collection period.
- Limited language diversity (primarily English).

### Dataset B – PHIUSIL URL Dataset (UCI)

- ~235,795 records, but many may share registered domains.
- The domain-aware split reduces but does not eliminate domain leakage.
- URLs were collected at a specific point in time; phishing domains change rapidly.
- The dataset's existing feature columns (webpage-derived) are **not** used;
  only the raw URL string and verified label are consumed.
- Label mapping (0/1 or -1/1) verified programmatically in `prepare_urls.py`.

---

## Model Limitations

### Message Classifier

- TF-IDF is a bag-of-words representation; it ignores word order and sentence structure.
- The model does not understand semantic meaning or context.
- New phishing vocabulary not seen during training will not be detected.
- MultinomialNB can produce poorly calibrated probabilities on imbalanced data.
- Class imbalance is addressed via `class_weight="balanced"` and threshold tuning.

### URL Classifier

- Features are purely lexical/static; no DNS, WHOIS, or page content is used.
- A clever attacker can craft a long URL with low suspicious-token count.
- URL shorteners reduce feature information significantly.
- The model cannot detect zero-day phishing domains with clean URLs.
- Random Forest can memorise training examples (mitigated by `min_samples_split`).

---

## SHAP / Explainability Limitations

- SHAP values indicate correlation between features and predictions, not causation.
- LinearExplainer is exact for linear models but approximate for calibrated classifiers.
- KernelExplainer (fallback) is a sampling-based approximation.
- Explanations do not prove that a message or URL is malicious.
- A feature flagged as suspicious (e.g., `login` in URL) also appears in legitimate URLs.

---

## Scope Limitations (Baseline)

- No authentication, user accounts, or persistent storage.
- No real-time URL reputation lookup (by design – URL destination is never visited).
- No email header analysis; only message body text.
- No image/attachment analysis.
- No multi-language support.
- No model monitoring or drift detection.
- No adversarial robustness evaluation.
- API rate limiting and production hardening are out of scope for this baseline.

---

## Reproducibility

- Dataset access may change over time (Mendeley / UCI URLs can break).
- Model performance depends on the exact dataset version downloaded.
- All random seeds are fixed (`RANDOM_SEED = 42`).
- Dependency versions are pinned in `requirements.txt`.

---

*This baseline is an educational prototype for CSET312. It is not production-ready.*
