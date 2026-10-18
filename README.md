# Phishing & Fraud Detection with Explainable AI (XAI)

**Course:** CSET312 – AI Applications, Bennett University  
**Repository:** https://github.com/Rahul-KumarAgrawal/phishing-fraud-detection-xai

An AI-based baseline for detecting phishing in SMS/email messages and URLs,
with SHAP-powered explanations, a FastAPI backend, and a React frontend.

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

# 4. (Optional) Re-prepare datasets or train models if needed
# Datasets are pre-split in data/processed and models are saved in models/
python scripts/train_message_model.py
python scripts/train_url_model.py

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
|   +-- model_comparison.csv
+-- src/
|   +-- fraud_phishing/
|       +-- data/           # Download, inspect, prepare scripts
|       +-- features/       # url_features.py (canonical 30-feature extractor)
|       +-- models/         # message_model.py, url_model.py
|       +-- explain/        # message_explainer.py, url_explainer.py
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
- **Expected rows:** ~5,971
- **Label policy:** ham → 0, smishing → 1, spam → excluded
- **Manual download required** (Mendeley requires browser authentication)

**Manual download steps:**
1. Go to https://data.mendeley.com/datasets/f45bkkt8pr/1
2. Download the CSV file
3. Place it at `data/raw/SMS_dataset.csv`

### Dataset B – PHIUSIL Phishing URL Dataset

- **Source:** https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset
- **License:** CC BY 4.0
- **Citation:** Hannousse, A. (2023). *PHIUSIL: A phishing URL dataset.* UCI Machine Learning Repository. https://doi.org/10.24432/C58S7F
- **Expected rows:** ~235,795
- **Label policy:** legitimate → 0, phishing → 1

Programmatic download is attempted via `ucimlrepo`. If it fails, manual download steps are printed.

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

### Branch B – URL Classification

| Component | Choice |
|---|---|
| Features | 30 static lexical features (see `url_features.py`) |
| Candidates | Logistic Regression, Random Forest |
| Selection | Best validation F1 |
| Split | 70/15/15 random + domain-aware test |
| Threshold | Tuned on validation set |
| Artifact | `models/url_model.joblib` |

### XAI

- **Message:** SHAP LinearExplainer (or KernelExplainer fallback)
- **URL:** SHAP TreeExplainer (Random Forest) or LinearExplainer

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Model load status |
| POST | `/api/analyze/message` | Analyse message text |
| POST | `/api/analyze/url` | Analyse a URL |

Docs at: http://localhost:8000/docs

---

## Running Tests

```powershell
# From repo root
pytest tests/ -v
# or: python -m pytest tests/ -v

# Expected (without trained models): 61 passed, 2 skipped
# Expected (with trained models):    63 passed, 0 skipped
```

---

## Model Artifacts

After training, `models/` contains:

```
models/
+-- message_model.joblib
+-- message_model_metadata.json
+-- url_model.joblib
+-- url_model_metadata.json
```

`model_metadata.json` records: dataset source, label map, features, random seed,
split strategy, model parameters, training date, dependency versions, eval results.

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
