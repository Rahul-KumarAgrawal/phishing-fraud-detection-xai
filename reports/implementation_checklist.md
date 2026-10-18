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
- [ ] `pip install` completed

## Stage 2 – Dataset Acquisition & Inspection
- [ ] `src/fraud_phishing/data/download_data.py` created
- [ ] `src/fraud_phishing/data/inspect_data.py` created
- [ ] SMS/smishing dataset downloaded (Mendeley)
- [ ] URL dataset downloaded (UCI PHIUSIL)
- [ ] `reports/dataset_inventory.md` created

## Stage 3 – Message Data Preparation
- [ ] `src/fraud_phishing/data/prepare_messages.py` created
- [ ] Labels mapped: ham→0, smishing→1
- [ ] Duplicates removed before splitting
- [ ] Train/val/test splits saved to `data/processed/`

## Stage 4 – URL Data Preparation
- [ ] `src/fraud_phishing/data/prepare_urls.py` created
- [ ] Raw URL strings extracted, labels normalized (legit→0, phishing→1)
- [ ] Split saved to `data/processed/`

## Stage 5 – URL Feature Extraction
- [x] `src/fraud_phishing/features/url_features.py` created
- [ ] `tests/test_url_features.py` created
- [ ] Feature schema: 28 named features, deterministic, immutable order
- [ ] Edge cases handled: malformed, IP, Unicode, missing scheme

## Stage 6 – Message Classification Pipeline
- [ ] `src/fraud_phishing/models/message_model.py` created
- [ ] `scripts/train_message_model.py` created
- [ ] `scripts/evaluate_message_model.py` created
- [ ] 3 models compared: LR, LinearSVC, MultinomialNB
- [ ] Model selected based on F1/AUC on val set
- [ ] `models/message_model.joblib` saved

## Stage 7 – URL Classification Pipeline
- [ ] `src/fraud_phishing/models/url_model.py` created
- [ ] `scripts/train_url_model.py` created
- [ ] `scripts/evaluate_url_model.py` created
- [ ] 2 models compared: LR, Random Forest
- [ ] Domain-aware split attempted
- [ ] `models/url_model.joblib` saved

## Stage 8 – SHAP Explanations
- [ ] `src/fraud_phishing/explain/message_explainer.py` created
- [ ] `src/fraud_phishing/explain/url_explainer.py` created
- [ ] `tests/test_explanations.py` created
- [ ] SHAP values verified against actual fitted models

## Stage 9 – FastAPI Backend
- [ ] `src/fraud_phishing/api/main.py` created
- [ ] `src/fraud_phishing/api/schemas.py` created
- [ ] `src/fraud_phishing/api/services.py` created
- [ ] `tests/test_api.py` created
- [ ] GET /api/health works
- [ ] POST /api/analyze/message works
- [ ] POST /api/analyze/url works

## Stage 10 – React + Tailwind Frontend
- [ ] Vite + React project bootstrapped in `frontend/`
- [ ] Message analysis tab implemented
- [ ] URL analysis tab implemented
- [ ] Results rendered from live backend
- [ ] `frontend/README.md` created

## Stage 11 – Integration Testing
- [ ] All pytest tests pass
- [ ] Frontend builds successfully
- [ ] End-to-end demo verified

## Stage 12 – Documentation
- [x] `reports/implementation_checklist.md` (this file)
- [ ] `reports/dataset_inventory.md`
- [ ] `reports/model_comparison.csv`
- [ ] `reports/error_analysis.md`
- [ ] `reports/limitations.md`
- [ ] `README.md` updated with full setup instructions

---

*Last updated: 2026-10-18*
