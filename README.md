# Phishing & Fraud Detection with Explainable AI (XAI)

**Course:** CSET312 – AI Applications, Bennett University  
**Repository:** https://github.com/Rahul-KumarAgrawal/phishing-fraud-detection-xai

An AI-based baseline for detecting phishing in SMS/email messages and URLs,
with SHAP-powered explanations, a FastAPI backend, and a React frontend.

> **College-project prototype.** This application is for an educational
> demonstration, not a dependable production security control. The served URL
> model passed a grouped internal showcase evaluation, while P9's previous P7
> candidate failed its one-time CompPhish evaluation (98.46% false-positive
> rate). CompPhish is now development data, so the new served model has no new
> independent external result. Do not use its URL output to make security
> decisions.

---

## Quick Start

### Option A — 1-Click Launch (Windows)
Double-click `START_APP.bat` in the project root (or run `python launcher.py`).
This checks prerequisites, trains models if missing, installs frontend packages if needed, and launches both the backend and frontend in browser.

### Option B — Step-by-Step Manual Setup

```powershell
# 1. Clone & enter repository
git clone https://github.com/Rahul-KumarAgrawal/phishing-fraud-detection-xai.git
cd phishing-fraud-detection-xai

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the supplied artifacts for the demonstration.
# launcher.py trains only if a required served artifact is missing.
python launcher.py

# Reproduce the final-showcase URL pipeline. It uses PHIUSIIL,
# conflict-screened MURL, and a byte-identical CompPhish development copy.
# It changes the served URL model only if its documented acceptance gate passes,
# while writing models/url_model_pre_showcase.* as rollback artifacts.
python scripts/prepare_showcase_url_development.py
python scripts/train_showcase_url_candidates.py

# Message retraining overwrites the served message artifact; run it only when
# intentionally reproducing that branch.
# python scripts/train_message_model.py

# 5. Run tests
pytest tests/ -v

# 6. Start backend (Terminal 1)
python -m uvicorn fraud_phishing.api.main:app --reload --port 8000

# 7. Start frontend (Terminal 2)
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## Project Structure

```
project-root/
+-- data/
|   +-- raw/           # Raw downloaded datasets (not committed)
|   +-- processed/     # Train/val/test splits
+-- models/            # Saved model artefacts
+-- reports/
|   +-- figures/       # Confusion matrices, plots
|   +-- dataset_inventory.md
|   +-- implementation_checklist.md
|   +-- error_analysis.md
|   +-- limitations.md
|   +-- message_metrics.json
|   +-- url_metrics.json
|   +-- url_leakage_report.json
|   +-- url_false_cases.csv
|   +-- model_comparison.csv
|   +-- explainability_fixes.md
|   +-- p5_artifact_safeguards.md
+-- src/
|   +-- fraud_phishing/
|       +-- data/           # Download, inspect, prepare scripts
|       +-- features/       # url_features.py (canonical 33-feature extractor)
|       +-- models/         # message_model.py, url_model.py
|       +-- explain/        # message_explainer.py, url_explainer.py, shap_utils.py
|       +-- api/            # FastAPI: main.py, schemas.py, services.py
+-- frontend/              # React + Vite frontend
+-- scripts/               # train_*.py, evaluate_*.py
+-- tests/                 # pytest test suite
+-- requirements.txt
+-- .gitignore
+-- README.md
```

---

## Datasets

### Dataset A – SMS/Smishing Messages

- **Source:** https://data.mendeley.com/datasets/f45bkkt8pr/1
- **License:** CC BY 4.0
- **Citation:** Mishra, S. & Soni, D. (2019). *SMS Phishing Dataset for Machine Learning and Pattern Recognition.* Mendeley Data, V1.
- **Rows:** 5,971 — 4,844 ham · 638 smishing · 489 spam
- **Label policy:** ham → 0, smishing → 1, spam → **excluded** (generic spam is not
  smishing, so it is kept out of the binary positive class)
- **Download:** automatic — `python src/fraud_phishing/data/download_data.py`

The downloader resolves `Dataset_5971.zip` from the Mendeley public API, verifies
the SHA-256 of the archive
(`9bbf3188fdad81495d8e82825648b9b63b53fc86841a3d26c02629990b233cc3`), extracts the
CSV, and asserts the documented row/label composition before writing
`data/raw/SMS_dataset.csv`. Mendeley sits behind Cloudflare, which rejects the
`requests` TLS fingerprint on some networks, so the downloader falls back to the
stdlib `urllib` client; if both are blocked it prints manual steps.

Note: the published labels are **inconsistently cased** (`Smishing` *and*
`smishing`, `spam` *and* `Spam`). The prepare script lowercases before mapping, so
all 638 smishing and 489 spam rows are matched.

**Manual fallback:**
1. Go to https://data.mendeley.com/datasets/f45bkkt8pr/1
2. Download `Dataset_5971.zip` and extract `Dataset_5971.csv`
3. Save it as `data/raw/SMS_dataset.csv`

### Dataset B – PHIUSIL Phishing URL Dataset

- **Source:** https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset
- **License:** CC BY 4.0
- **Citation:** Hannousse, A. (2023). *PHIUSIL: A phishing URL dataset.* UCI Machine Learning Repository. https://doi.org/10.24432/C58S7F
- **Rows:** 235,795 — 134,850 legitimate · 100,945 phishing
- **Label policy:** legitimate → 0, phishing → 1

> **PHIUSIL's raw labels are inverted relative to intuition.** The official
> dataset card states *"Label 1 corresponds to a legitimate URL, label 0 to a
> phishing URL"*. `prepare_urls.py` maps `1 → 0` (legitimate) and `0 → 1`
> (phishing) and **fails loudly** if the raw counts do not match the documented
> 134,850 / 100,945 split. Reading it backwards silently inverts every metric and
> every label the API returns, while still producing excellent-looking numbers.

Programmatic download is attempted via `ucimlrepo`. If it fails, manual download steps are printed.

---

### Dataset C – LegitPhish URLs (development benchmark)

- **Source:** https://data.mendeley.com/datasets/hx4m73v2sf/2 (DOI `10.17632/hx4m73v2sf.2`)
- **License:** CC BY 4.0
- **Citation:** Potpelwar, R., Kulkarni, U. & Waghmare, J. (2025). *LegitPhish Dataset.* Mendeley Data, V2.
- **Rows:** 101,219 raw → 100,872 after dropping 1 unlabelled row and 346 duplicate URLs
- **Composition:** 63,678 phishing · 37,540 legitimate (SHA-256 verified on download)
- **Label policy:** legitimate → 0, phishing → 1
- **Role:** development / diagnostic benchmark — no model is trained on it, but it
  revealed the false-positive defect and informed the revised feature subset

> **This dataset uses the same inverted label convention as PHIUSIL** (`0` = phishing,
> `1` = legitimate). `prepare_external_urls.py` maps it to the project convention and
> **fails hard** if the raw counts ever change, so the labels cannot silently invert.

> **LegitPhish is only half independent.** 98.8% of its *legitimate* URLs are
> byte-identical to PHIUSIL's, so precision on that class is mostly a memorisation
> check. The 445 legitimate URLs that are genuinely new are the ones the URL
> diagnosis uses. Its *phishing* class (63,492 URLs from URLHaus/PhishTank) has
> **zero** PHIUSIL overlap.
>
> Because results on it guided development, it is **no longer a clean
> generalisation claim** — Dataset E is. See `reports/url_remediation.md`.

---

### Dataset D – MURL URLs (training augmentation only)

- **Source:** https://data.mendeley.com/datasets/5fk24y4xmg/2
- **License:** CC BY 4.0
- **Rows kept:** 120,000 (60,000 benign/legitimate · 60,000 phishing, capped with a
  fixed seed). `defacement`, `malware`, `spam` and `Exploit` rows are dropped.
- **Label policy:** benign/legitimate → 0, phishing → 1
- **Role:** training augmentation only — it supplies path-bearing URLs that
  PHIUSIL lacks. Never used for evaluation.
- **Caveat:** its URLs are stored **without a scheme**, so `is_https` reads 0 for
  every MURL row. That is a property of the file, and it is why MURL cannot fill
  the "HTTPS + path" region in training.

---

### Dataset E – URL-Phish URLs (development-informed diagnostic set)

- **Source:** https://data.mendeley.com/datasets/65z9twcx3r/2
- **License:** CC BY 4.0
- **Rows:** 116,600 raw → 115,231 after de-duplication
  (98,641 benign · 16,590 phishing; SHA-256 verified and composition asserted)
- **Label policy:** already intuitive — `0` = benign, `1` = phishing
- **Provenance:** phishing from PhishTank; benign from the Research Organization
  Registry list of legitimate institutions
- **Role:** development-informed diagnostic data. It has already influenced URL
  model-development decisions and has measurable overlap with the augmented
  training set, so it is not an untouched final evaluation set. See
  `reports/p6/dataset_qualification.md`.

> **Its scheme distribution is inverted relative to PHIUSIL**: after exact-URL
> de-duplication, 34.94% of benign URLs are HTTPS and 91.43% of phishing URLs
> are HTTPS, so "HTTPS implies
> legitimate" fails on it. This is the property that exposes the URL model's
> generalisation failure.

---

## ML Pipeline

### Branch A – Message Classification

| Component | Choice |
|---|---|
| Representation | TF-IDF word (1-3gram) + char_wb (2-5gram) via `FeatureUnion` |
| Candidates | Logistic Regression, LinearSVC (calibrated), MultinomialNB |
| Selection | Best validation F1 |
| Split | 70 / 15 / 15 stratified |
| Threshold | Tuned on validation set (maximises F1) |
| Artifact | `models/message_model.joblib` |
| Data guard | Refuses to train on the synthetic placeholder unless `--allow-synthetic` is passed |

### Branch B – URL Classification

| Component | Choice |
|---|---|
| Features | 33 static lexical features, including 3 brand-impersonation features (see `url_features.py`) |
| Candidates | Logistic Regression, Random Forest |
| Selection | Best validation F1 for the served model |
| Split | 70/15/15 random, plus domain-isolated and template-clean test sets |
| Threshold | Tuned on validation set; the domain-holdout model tunes on held-out domains |
| Artifact | `models/url_model.joblib` — final-showcase Random Forest using the declared 30-feature P8 subset |
| Eval artifact | `models/url_model_domain_holdout.joblib` — trained on the domain-isolated pool, used only for unseen-domain evaluation |
| Candidate | `models/url_model_revised.joblib` — Logistic Regression on 24 of the 33 features, trained on PHIUSIL + MURL. **Not promoted**; see `reports/url_remediation.md` |
| Development benchmark | LegitPhish (`data/processed/legitphish_external.csv`) |
| URL-Phish role | Development-informed diagnostic dataset; it is not an untouched final test |
| CompPhish role | P9 is preserved historical evidence for the failed P7 Random Forest. A byte-identical development copy is used by the final-showcase pipeline. |

### XAI

Every explanation payload states what it explains and in which units, and each
contribution carries the SHAP value and the feature's real value as separate
fields.

| Branch | Explainer | `units` | `explained_output` |
|---|---|---|---|
| Message (`LinearSVC`, calibrated) | SHAP `KernelExplainer` on `predict_proba` | `probability` | `P(smishing)` |
| URL (`RandomForestClassifier`, served) | SHAP `TreeExplainer` | `probability` | `P(phishing)` |
| URL (`LogisticRegression`, candidate) | SHAP `LinearExplainer` | `log_odds` | `log-odds of P(phishing)` |

`base_value + sum(shap_value)` equals `explained_output` for the input;
the measured residual is returned as `additivity_gap`. Message contributions are
additive to the calibrated probability (gap `0.0` on the samples in
`reports/explainability_fixes.md`), and the SHAP baseline is an empty message.
No field is named `value`. Details, before/after measurements and the remaining
limitations: `reports/explainability_fixes.md`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Model load status + message-training-data provenance |
| POST | `/api/analyze/message` | Analyse message text |
| POST | `/api/analyze/url` | Analyse a URL |

Docs at: http://localhost:8000/docs

---

## Running Tests

```powershell
# From repo root
pytest tests/ -v
# or: python -m pytest tests/ -v

# Results are recorded with their execution date in
# reports/implementation_checklist.md. A documented xfail records a known
# revised-candidate limitation; some checks intentionally skip when optional
# source data is absent.
```

Explanation-specific tests live in `tests/test_explanation_consistency.py`
(22 tests). They assert the probability mapping for both classes, SHAP class
alignment, `base_value + Σshap == explained_output`, direction signs,
`feature_value` correctness, and that no API field is named `value`.

Artifact-guard tests live in `tests/test_model_artifact_guards.py` (47 tests).
They assert that the shipped artifacts still load and that missing metadata,
stale schema versions, stale fingerprints, undeclared feature subsets, reordered
or inserted features, and message artifacts with a bad threshold or TF-IDF spec
all fail with a clear diagnostic. `tests/test_url_features.py` additionally pins
the feature counts in the `url_features.py` docstring to the schema itself.

---

## Model Artifacts

After training, `models/` contains:

```
models/
+-- message_model.joblib
+-- message_model_metadata.json
+-- url_model.joblib                          # served by the API
+-- url_model_metadata.json
+-- url_model_domain_holdout.joblib           # domain-isolated eval model
+-- url_model_domain_holdout_metadata.json
+-- url_model_revised.joblib                  # candidate, NOT promoted
+-- url_model_revised_metadata.json
```

`model_metadata.json` records: dataset source, label map, features, random seed,
split strategy, model parameters, training date, dependency versions, eval results.
URL metadata additionally records `feature_schema_version`, a
`feature_schema_fingerprint` (a hash of the canonical feature schema, so a schema
edit is caught even if the version is not bumped) and the model's own
`feature_names`; the loaders refuse to load a model whose feature list does not
line up with the current extractor, whose metadata is missing or stale, or whose
pipeline width disagrees with its metadata. A refused model is reported as
unloaded, so `/api/analyze/*` returns `503` instead of serving it. See
`reports/p5_artifact_safeguards.md`.
The message metadata also records `dataset_provenance` (`real` / `synthetic` /
`unknown`) and a `prototype` flag, which `/api/health` exposes and the UI surfaces
as a banner; the message loader requires those keys and the `feature_type` it was
trained with.

---

## Results

Metrics use **phishing as the positive class**. All URL figures were produced with
the corrected label mapping.

### Messages (real Mendeley dataset, 5,971 → 5,462 after spam exclusion and dedup)

Test set: 820 messages (726 ham / 94 smishing), model `LinearSVC` (calibrated),
threshold 0.6181.

| Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|
| 0.9894 | 0.9894 | 0.9894 | 1.0000 |

Confusion matrix (`[[TN, FP], [FN, TP]]`) = `[[725, 1], [1, 93]]`.

### URLs (PHIUSIL, 235,370 after dedup)

Three test sets are reported, each scored by a model allowed to see it. They
measure different things and **must not be averaged**.

| Test set | Model | N | Precision | Recall | F1 | ROC-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|
| Random | production (random split) | 35,306 | 0.9993 | 0.9952 | 0.9972 | 0.9982 | 11 | 72 |
| Domain-aware | domain-holdout | 38,591 | 0.9997 | 0.9958 | 0.9977 | 0.9986 | 6 | 77 |
| Template-clean | domain-holdout | 38,342 | 0.9997 | 0.9957 | 0.9977 | 0.9985 | 6 | 77 |

The URL scores are very high and **survive domain isolation**, which means they are
not primarily a leakage artefact: they are driven by a genuine shortcut in the
dataset. In PHIUSIL every legitimate URL is a bare `https://domain` with no path,
so `is_https` and `slash_count` almost perfectly separate the classes. The
practical consequence is visible in the misclassifications: typosquats with no
path (e.g. `https://www.netflix-vn.com`, `https://www.colis-dhl.com`) are scored
as legitimate, while legitimate digit-heavy domains
(e.g. `https://www.180360.com`) are flagged as phishing. Per-URL evidence is in
`reports/url_false_cases.csv`. Brand-impersonation and lookalike-domain features
have since been added, and the false-positive defect on this dataset is analysed
and partly remediated in `reports/url_remediation.md`; the remaining
generalisation failure is documented in `reports/limitations.md`.

### URLs — external

Full detail in `reports/url_remediation.md` and `reports/url_model_comparison.md`.

**LegitPhish (development benchmark, 100,872 URLs):**

| Model | P | R | F1 | FPR on the 445 newly-seen legitimate URLs | recall, unseen phishing |
|---|---|---|---|---|---|
| production (random split) | 0.9930 | 1.0000 | 0.9965 | **100.0%** | 100.0% |
| domain-holdout | 0.9930 | 1.0000 | 0.9965 | **100.0%** | 100.0% |
| revised candidate | 0.9952 | 0.9625 | 0.9786 | **4.0%** | 96.3% |

The defect: every legitimate URL that is genuinely new to the model was flagged
(as before, the 1.2% headline false-positive rate was memorisation of URLs the
model had already seen). The measured cause is not "path implies phishing" but
the conjunction **"HTTPS *and* a non-empty path"**, which occurred **only in the
phishing class** in training data (0 legitimate vs 32,780 phishing rows).
PHIUSIL's legitimate URLs are all bare `https://domain` homepages and the MURL
augmentation source stores **no scheme**, so the region was empty. Removing the
path/query-encoding features and switching to Logistic Regression cuts the
false-positive rate on those 445 URLs from 100.0% to 4.0%.

**URL-Phish (development-informed diagnostic set, 115,231 URLs):**

| Model | P | R | F1 | ROC-AUC | FP | FN |
|---|---|---|---|---|---|---|
| production (random split) | 0.1511 | 0.9955 | 0.2624 | **0.5496** | 92,790 | 75 |
| domain-holdout | 0.1510 | 0.9954 | 0.2621 | 0.5675 | 92,880 | 77 |
| revised candidate | 0.1412 | 0.6459 | 0.2318 | **0.5329** | 65,165 | 5,875 |

**Neither model generalises.** Both rank benign above phishing at barely better
than chance, and both flag most legitimate URLs. The two corpora disagree about
the scheme signal — legitimate HTTPS rate is 100% in PHIUSIL but 17.9% in
URL-Phish, while phishing HTTPS rate is ~49% in PHIUSIL but 86.2% in URL-Phish —
and `is_https` is the model's most important feature. PHIUSIL and LegitPhish were
collected when phishing largely used plain HTTP; URL-Phish is modern and inverts
that. **The revised model is therefore not promoted to production**, and the URL
classifier should not be presented as a working phishing control. See the
recommendation in `reports/url_remediation.md` §6.

### URLs — historical P9 evaluation (CompPhish)

P9 evaluated the **P7 Random Forest candidate once**, using its already-saved
threshold (0.2933606675557894), on the frozen, group-excluded CompPhish subset.
This candidate was never served. Of 6,443 retained records (4,490 legitimate;
1,953 phishing), its confusion matrix was `[[69, 4421], [6, 1947]]`.

| Precision | Recall | F1 | Accuracy | ROC-AUC | FPR | FNR |
|---|---|---|---|---|---|---|
| 0.3057 | 0.9969 | 0.4680 | 0.3129 | 0.3216 | 98.46% | 0.31% |

The result is a failure for practical use: it marked 4,421 legitimate URLs as
phishing. The frozen file, manifest, P9 metrics and report are preserved in
`data/processed/compphish_external_frozen.csv`, `reports/p6/`, and
`reports/p9/`. P9 remains the fixed historical result for P7. The final
showcase intentionally repurposes a verified byte-identical copy of this
subset as development data, so it cannot claim a new independent CompPhish
evaluation.

### URLs — final showcase grouped internal evaluation

The final showcase pipeline combines PHIUSIIL, conflict-screened MURL, and the
CompPhish development copy. Its deterministic train/validation/test partitions
are connected components of registered domain and normalized URL template, with
zero raw-URL, normalized-URL, domain, and template overlap across every pair.

The served model is a Random Forest using the P8 `without_brand` 30-feature
subset. Its threshold, selected only on validation, is **0.514326**. On the
one-time grouped internal test partition (58,026 URLs: 30,471 legitimate,
27,555 phishing), it measured:

| Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---|---|---|---|---|---|
| 0.8985 | 0.9386 | 0.9181 | 0.9799 | 0.9782 | 9.59% | 6.14% |

The preceding served baseline had F1 0.8435 and FPR 33.20% on the same grouped
test partition. The new model reduces the FPR by 23.61 percentage points, but
legitimate URLs with a non-empty path still have a 27.37% FPR and URLs with a
query have a 23.93% FPR. See `reports/showcase/` for all four candidate results,
hashes, split evidence, rollback artifact, and acceptance criteria.

### Evaluation integrity

Two silent defects previously made the reported numbers meaningless. Both are now
fixed and guarded:

1. **Inverted URL labels.** PHIUSIL's `1 = legitimate` convention was read as
   `1 = phishing`, which flipped every metric and every API label. Guarded by a
   composition assertion in `prepare_urls.py` and
   `tests/test_url_split_leakage.py`.
2. **Evaluation leakage.** The "domain-aware" test set was scored with a model
   trained on a random split that shared **32,826 identical URLs and 86.2% of
   path templates** with it. The domain-aware claim was therefore measuring
   memorisation. The leakage is now quantified in
   `reports/url_leakage_report.json`, the isolation is asserted in tests, and the
   unseen-domain figures come from a model trained only on the domain-isolated
   pool.

---

## Security Notes

- The URL destination is **never** visited, fetched, or resolved.
- Submitted messages and URLs are **not stored** by default.
- CORS is restricted to `localhost:5173` (dev only).
- Do not deploy to production without proper authentication and rate limiting.

---

## Disclaimer

> This application produces **model-based risk assessments**, not guarantees.
> Verify important requests through official channels.
> SHAP explanations indicate feature correlation, not causation.
> See `reports/limitations.md` for full scope and limitation details.

---

## Demo Script (Presentation)

```powershell
# Terminal 1 – start backend
python -m uvicorn fraud_phishing.api.main:app --port 8000

# Terminal 2 – start frontend
cd frontend && npm run dev

# Terminal 3 – quick API smoke test
Invoke-RestMethod http://localhost:8000/api/health | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/analyze/message `
  -ContentType "application/json" `
  -Body '{"message": "URGENT: Your bank account will be suspended. Verify at http://bit.ly/secure-now"}' |
  ConvertTo-Json -Depth 10

Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/analyze/url `
  -ContentType "application/json" `
  -Body '{"url": "http://192.168.1.1/login?verify=true&account=update"}' |
  ConvertTo-Json -Depth 10
```
