# College Project Report — AI-Based Fraud and Phishing Detection

**Course:** CSET312 — AI Applications, Bennett University  
**Project scope:** educational prototype and reproducible demonstration  
**Status:** ready for demonstration subject to the limitations below

## Aim and scope

The application accepts either a message or a URL and returns a binary risk
classification with a probability and an explanation. It is a course prototype,
not a production security control. It never visits, resolves, or fetches a
submitted URL.

The project deliberately stops the prospective P11 benchmark-collection work.
That work requires a timestamped phishing feed and individual, page-level
evidence for legitimate URLs. It is deferred because it is unnecessary for the
college demonstration; no substitute records have been created.

## System design

| Workflow | Input | Method | Output |
|---|---|---|---|
| Message analysis | Raw SMS/email text | TF-IDF word and character n-grams with calibrated LinearSVC | `P(smishing)`, label, SHAP token contributions |
| URL analysis | Raw URL string only | 33 static lexical features with Random Forest | `P(phishing)`, label, feature values, SHAP contributions |

The FastAPI service checks model metadata and feature schemas when loading an
artifact. The React frontend displays the classification, probability, and
explanation. Explanations declare their unit and explained class. Tests verify
that the positive-class mapping is correct and that SHAP additivity agrees with
the displayed output.

## Data and preparation

| Dataset | Use in this college scope | Label convention |
|---|---|---|
| Mendeley SMS Phishing Dataset | Message-model training and internal testing | ham = 0; smishing = 1; generic spam excluded |
| PHIUSIIL | URL development/training | raw 1 = legitimate and raw 0 = phishing; mapped to project 0 = legitimate, 1 = phishing |
| MURL | Conflict-screened URL augmentation in the P7 automated development pipeline | legitimate/benign = 0; phishing = 1; other classes excluded |
| LegitPhish and URL-Phish | Development diagnostics only | mapped to project convention |
| CompPhish frozen subset | Historical P9 evaluation of P7; byte-identical development copy used for the final showcase | 0 = legitimate; 1 = phishing |

The final showcase pipeline combines PHIUSIIL, conflict-screened MURL, and a
byte-identical CompPhish development copy. The original frozen CompPhish file,
manifest, and P9 result are preserved. Train, validation, and test components
are isolated by registered domain and normalized URL template.

## Measured results

### Message model

The saved calibrated LinearSVC reported the following held-out test result on
820 messages (726 ham and 94 smishing): precision 0.9894, recall 0.9894, F1
0.9894, ROC-AUC 1.0000; confusion matrix `[[725, 1], [1, 93]]`. These results
are internal to the Mendeley collection and do not establish performance on new
message sources.

### URL models

The served Random Forest's PHIUSIIL internal scores are high (random-split F1
0.9972, ROC-AUC 0.9982), but PHIUSIIL has a structural shortcut: legitimate
examples are bare HTTPS hostnames without paths. Therefore those internal scores
are not a claim of practical phishing-detection accuracy.

P9 performed one fixed external evaluation of the **unserved P7 Random Forest**
on 6,443 frozen CompPhish records. At the metadata-loaded threshold
0.2933606675557894, the measured confusion matrix was `[[69, 4421], [6, 1947]]`:

| Precision | Recall | F1 | Accuracy | ROC-AUC | False-positive rate | False-negative rate |
|---|---|---|---|---|---|---|
| 0.3057 | 0.9969 | 0.4680 | 0.3129 | 0.3216 | 98.46% | 0.31% |

This is an external generalization failure for P7. Its historical record is
preserved; the final showcase uses CompPhish as development data and does not
claim a new independent external result.

### Final showcase served URL model

The served URL artifact is now a 30-feature Random Forest selected from four
P7/P8 configurations on grouped validation and then scored once on a grouped
internal test partition. Its threshold is 0.514326. The test metrics are
precision 0.8985, recall 0.9386, F1 0.9181, ROC-AUC 0.9799, PR-AUC 0.9782,
FPR 9.59%, and FNR 6.14%. The preceding served baseline had F1 0.8435 and FPR
33.20% on the same test partition. Legitimate paths and queries remain a
weakness: their FPRs are 27.37% and 23.93%.

## Reproducibility and demonstration

From the repository root:

```powershell
# Install Python dependencies once
python -m pip install -r requirements.txt

# Start both application components; existing served artifacts are used.
python launcher.py

# Or start separately
python -m uvicorn fraud_phishing.api.main:app --port 8000
Set-Location frontend
npm run dev
```

The one-click launcher trains only if a required served artifact is absent. To
reproduce the automated final-showcase URL pipeline, run:

```powershell
python scripts/prepare_showcase_url_development.py
python scripts/train_showcase_url_candidates.py
```

These scripts use PHIUSIIL, conflict-screened MURL, and the CompPhish
development copy. They save candidate artifacts separately and promote only
when the predeclared grouped internal acceptance criteria pass; the prior served
artifact is retained as a rollback. See the root README for tests and API use.

## Limitations and deferred work

- The URL classifier's current evidence is inadequate for security use. It is
  intentionally shown only as a prototype demonstration.
- URL-Phish informed development and CompPhish is now development data; neither
  is a clean final test for the served showcase model.
- The prospective P11 benchmark is deferred. Completing it requires new,
  timestamped and provenance-verified phishing records plus page-level evidence
  for legitimate records, followed by the existing overlap gates.
- Static URL features cannot observe destination content, DNS, certificates, or
  the changing behavior of a live phishing campaign.
- SHAP explains model behavior, not whether a message or URL is truly malicious.

Detailed provenance, audit records, and immutable P9 evidence remain under
`reports/p6/`, `reports/p9/`, `reports/p10/`, and `reports/p11/`.
