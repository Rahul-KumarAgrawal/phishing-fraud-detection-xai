# P3 — URL classifier robustness: diagnosis, remediation, and honest results

**Scope.** The URL classifier only. The 0.42 message threshold, the SMS model and
the UI were not touched. Every number below is reproduced by a script in
`scripts/`; none is estimated.

**Bottom line up front.** The specific defect that was diagnosed — every
legitimate URL with a path being classified as phishing — is real and is fixed
(false-positive rate on those URLs drops from **100.0% to 4.0%**). But the
untouched external evaluation shows that **neither the original nor the revised
model generalises**: on URL-Phish (Dataset E, never used for any decision) both
models rank legitimate and phishing URLs at close to chance (ROC-AUC 0.55 and
0.53). The revised model is therefore **not promoted to production**; it is
delivered as a candidate artifact and its limitations are documented below.

---

## 1. Diagnosis (Step 1)

Reproduce: `python scripts/url_feature_diagnosis.py`
Evidence: `reports/url_diagnosis.md`, `reports/url_diagnosis_false_positives.csv`,
`reports/url_feature_distributions.csv`.

* Labels and the mapping were verified: all 445 flagged URLs carry raw
  `ClassLabel == 1.0`, which is LegitPhish's code for *legitimate*.
* Feature extraction raises no exceptions on them (0 failures).
* **Reproduction:** FPR on the 445 byte-distinct legitimate URLs was
  **100.00% (445/445), mean P(phishing) 0.9998** — while the same model scored
  **0.02% (8/36,935)** on legitimate URLs it had already seen. The apparent
  accuracy on PHIUSIL test data was memorisation.
* The false positives are ordinary pages: `wikipedia.org`, `stackoverflow.com`,
  `youtube.com`, `forbes.com`, `nytimes.com`.
* 16 of the 30 features that existed when this diagnosis ran had a *reversed*
  direction between PHIUSIL and LegitPhish. (The extractor has 33 features today:
  the 3 brand-impersonation features were added by the P3 remediation *after*
  this measurement, which is why `reports/url_feature_distributions.csv` has 30
  feature columns.)

### Root cause

The conjunction **"HTTPS scheme AND a non-empty path" occurred only in the
phishing class in training data** (augmented fitting slice, 268,577 rows):

| shape | legitimate | phishing |
|---|---:|---:|
| `is_https == 1` and `path_length > 0` | **0** | **32,780** |
| `path_length > 0` | 42,424 | 81,539 |
| legitimate rows with `is_https == 1` | 97,325 / 146,209 | — |

Every one of the 445 false-positive URLs has a path (440/445) and HTTPS
(442/445), so a tree model that learns that conjunction flags them all. Two
dataset properties create the gap:

1. PHIUSIL's legitimate URLs are *all* bare `https://domain` homepages — no
   paths, and `www.` present.
2. MURL (the augmentation source, Dataset D) supplies path-bearing URLs but
   stores **no scheme at all**, so its 102k benign rows do not fill the
   "HTTPS + path" region.

---

## 2. Controlled experiments (Step 2)

`is_https`, `path_length`, `slash_count` and `query_param_count` were **not**
removed on intuition. Three experiments were run and the results decided what to
keep.

### 2.1 Feature-subset ablation

Reproduce: `python scripts/ablate_url_features.py` →
`reports/url_feature_ablation.md` / `.json`. 10 subsets × 2 training pools × 2
model families; thresholds selected on a domain-held-out 15% slice of each pool.
LegitPhish is the development benchmark here, so figures in the *new-legit FPR*
column are in-sample for development purposes only.

Augmented pool (PHIUSIL + MURL), LogisticRegression:

| subset | features | PHIUSIL-test P | R | F1 | dev FPR (all legit) | FPR on the 445 new legit | recall, byte-distinct phishing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all` | 33 | 0.9983 | 0.9552 | 0.9763 | 0.66% | **42.9%** | 98.1% |
| `no_path_features` | 30 | 0.9940 | 0.8665 | 0.9259 | 0.54% | **3.6%** | 96.0% |
| **`no_path_no_query`** | **24** | **0.9880** | **0.9017** | **0.9429** | **1.12%** | **4.9%** | **96.3%** |
| `no_path_query_length` | 20 | 0.9685 | 0.5472 | 0.6993 | 1.74% | 3.6% | 96.9% |
| `consistent_only` | 16 | 0.9700 | 0.5544 | 0.7056 | 1.70% | 4.9% | 97.0% |
| `no_https` | 32 | 0.8711 | 0.8517 | 0.8613 | 11.99% | 57.8% | 97.6% |
| `no_brand` | 30 | 0.9982 | 0.9585 | 0.9779 | 0.69% | 43.8% | 98.2% |

Findings:

* **Data augmentation alone does not fix it.** `augmented / all / RandomForest`
  still has a **100.0%** false-positive rate on the 445 URLs.
* **Removing the path-encoding features does.** `no_path_features` →
  3.6%, `no_path_no_query` → 4.9%.
* **RandomForest never fixes it** — its best subset (`consistent_only`) is still
  62.7%, because a tree can express the conjunction. The model family, not just
  the feature set, matters.
* **Removing `is_https` makes things worse everywhere** (`no_https` 57.8%,
  `consistent_no_https` 43.6%), so the scheme feature was kept.
* **The brand features neither help nor hurt** on this benchmark (`no_brand`
  43.8% vs `all` 42.9%); they are retained for the brand-impersonation cases and
  are covered by tests, not by this metric.

### 2.2 Should the missing scheme in MURL be normalised? No.

MURL stores host+path with no scheme, so its 120k rows are forced to `http` by
the extractor. Controlled experiment: prefix `https://` to every MURL row
(label-independent) and re-run.

| pool | subset | model | new-legit FPR | PHIUSIL-test F1 |
|---|---|---|---:|---:|
| A: MURL scheme-less (current) | `no_path_features` | LR | **3.8%** | 0.9218 |
| B: MURL forced to `https` | `no_path_features` | LR | **34.2%** | 0.9798 |
| A | `all` | RF | 100.0% | 0.9972 |
| B | `all` | RF | 53.3% | 0.9953 |

Normalising the scheme helps the tree model but **breaks the linear model**, and
it would create a *new* perfect shortcut in the other direction — with all MURL
rows moved to HTTPS, every remaining HTTP row is phishing, which fails
catastrophically on URL-Phish's mostly-HTTP benign traffic. **Rejected as data
fabrication that trades one artifact for another.**

### 2.3 What was added instead

Three brand-impersonation features (`brand_token_count`,
`brand_domain_mismatch`, `brand_lookalike_domain`) were added — brand token in
the hostname, brand token on a different registered domain, and registered
domain within edit distance 1–2 of a known brand. They are implemented with a
fixed brand list and a bounded Levenshtein distance and are unit-tested. Their
measured contribution to the 445-URL false-positive rate is nil; they are
justified by the impersonation cases in §5, not by that metric.

Known accepted false positive: `apple.stackexchange.com` sets
`brand_domain_mismatch = 1` because `apple` appears as a subdomain of a
different registered domain.

---

## 3. Evaluation integrity (Step 3)

| ID | dataset | role |
|---|---|---|
| A | Mendeley SMS/Smishing | message branch; untouched by P3 |
| B | PHIUSIL URLs | training (random split **and** domain-isolated split) |
| C | LegitPhish URLs | **development / diagnostic benchmark** |
| D | MURL URLs | training augmentation only |
| E | URL-Phish URLs | **final untouched external test** |

* Training/validation/test remain separate. The augmented pool is split with
  `GroupShuffleSplit` on the registered domain, and the fit and validation
  domains are asserted disjoint.
* The revised model's threshold is selected only on `urls_aug_val.csv`, a
  domain-held-out slice of the augmented pool. No benchmark or external set ever
  sets a threshold.
* Dataset E was downloaded, its composition asserted, and then not inspected
  until the single evaluation in §4. No feature, model or threshold decision
  used it. Datasets C and D influenced development; §4 says so explicitly.

---

## 4. Original vs revised, side by side (Step 4)

Reproduce: `python scripts/compare_url_models.py` →
`reports/url_model_comparison.md` / `.json` (full confusion matrices there).

| model | subset | features | threshold | trained on |
|---|---|---:|---:|---|
| baseline (production) | `all` | 33 | 0.6294 | PHIUSIL |
| baseline (domain-heldout) | `all` | 33 | 0.6545 | PHIUSIL, domain-isolated |
| revised | `no_path_no_query` | 24 | 0.3066 | PHIUSIL + MURL (augmented) |

### PHIUSIL random test (in-distribution — not a generalisation claim)

| model | P | R | F1 | FP | FN | confusion `[[tn,fp],[fn,tp]]` |
|---|---:|---:|---:|---:|---:|---|
| baseline (production) | 0.9997 | 0.9948 | 0.9973 | 4 | 78 | `[[20224,4],[78,15000]]` |
| baseline (domain-heldout) | 0.9998 | 0.9954 | 0.9976 | 3 | 69 | `[[20225,3],[69,15009]]` |
| revised | 0.9889 | 0.8836 | 0.9333 | 150 | 1,755 | `[[20078,150],[1755,13323]]` |

### PHIUSIL template-clean test (unseen domains **and** unseen templates)

| model | P | R | F1 | FP | FN | confusion |
|---|---:|---:|---:|---:|---:|---|
| baseline (production) | 0.9998 | 0.9962 | 0.9980 | 4 | 69 | `[[20293,4],[69,17976]]` |
| baseline (domain-heldout) | 0.9997 | 0.9959 | 0.9978 | 6 | 74 | `[[20291,6],[74,17971]]` |
| revised | 0.9912 | 0.9032 | 0.9452 | 144 | 1,746 | `[[20153,144],[1746,16299]]` |

### LegitPhish — development benchmark

| model | P | R | F1 | FPR (all legit) | FNR | FPR on the 445 new legit | recall, unseen phishing |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (production) | 0.9930 | 1.0000 | 0.9965 | 1.20% | 0.00% | **100.0%** | 100.0% |
| baseline (domain-heldout) | 0.9930 | 1.0000 | 0.9965 | 1.20% | 0.00% | **100.0%** | 100.0% |
| revised | 0.9952 | 0.9625 | 0.9786 | 0.78% | 3.75% | **4.0%** | 96.3% |

The 1.20% headline FPR of the baselines is almost entirely memorised URLs; the
445 URLs that are genuinely new are the number that matters, and that is the
defect this work fixed.

### URL-Phish — final untouched external set

| model | P | R | F1 | ROC-AUC | FP | FN | FPR | FNR | confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline (production) | 0.1511 | 0.9955 | 0.2624 | **0.5496** | 92,790 | 75 | 94.07% | 0.45% | `[[5851,92790],[75,16515]]` |
| baseline (domain-heldout) | 0.1510 | 0.9954 | 0.2621 | 0.5675 | 92,880 | 77 | 94.16% | 0.46% | `[[5761,92880],[77,16513]]` |
| revised | 0.1412 | 0.6459 | 0.2318 | **0.5329** | 65,165 | 5,875 | 66.06% | 35.41% | `[[33476,65165],[5875,10715]]` |

Integrity strata (595 benign URLs are byte-identical to training data; 10,112
phishing URLs sit on a domain seen in PHIUSIL):

| model | P (excl. memorised) | R (excl. memorised) | P (unseen domain only) | R (unseen domain only) |
|---|---:|---:|---:|---:|
| baseline (production) | 0.1511 | 0.9955 | 0.0730 | 0.9895 |
| revised | 0.1412 | 0.6459 | 0.0511 | 0.4822 |

**The threshold-free view is the important one: ROC-AUC is 0.55 for the original
model and 0.53 for the revised one.** Both are barely above chance at *ranking*
benign above phishing on this corpus, so the revised model's lower headline FPR
comes with a third of the phishing recall lost and does not represent a
transferable improvement.

Why: the two corpora disagree about the scheme signal, which is the model's
most important feature.

| corpus | legitimate HTTPS | phishing HTTPS |
|---|---:|---:|
| PHIUSIL (training) | 100% | ~49% |
| URL-Phish (external) | **17.9%** | **86.2%** |

PHIUSIL and LegitPhish were collected when phishing mostly used plain HTTP, so
they reward "HTTP implies phishing". URL-Phish is modern, where phishing is
almost all HTTPS. A model selected with the available development data
necessarily learns the legacy direction. Note also that URL-Phish's benign URLs
are 98.8% path-bearing, so the path fix does not help there — that dataset is
failing on the scheme, not the path.

### Which datasets influenced development

* **PHIUSIL (B)** — training; also the source of the formatting shortcut.
* **MURL (D)** — training augmentation only.
* **LegitPhish (C)** — development benchmark; the diagnosis and the choice of
  `no_path_no_query` came from it. Results on it are therefore *not* a clean
  generalisation claim.
* **URL-Phish (E)** — untouched until the single evaluation above.

No model was trained on excerpts from C or E.

---

## 5. Tests and artifacts (Step 5)

* `python -m pytest tests/ -q` → **113 passed, 2 skipped, 1 xfailed**.
* New/updated tests: legitimate URLs with paths and query strings, bare-HTTPS
  URLs, IP-host phishing, brand impersonation, lookalike-domain detection,
  `brand_lookalike_domain` bounded-distance behaviour, probability-is-phishing
  sanity, and a metadata/feature-name schema contract.
* The xfail is deliberate: `https://www.forbes.com/sites/forbestechcouncil/2024/01/01/example/`
  scores 0.3833 against a 0.3066 threshold. A long legitimate path full of
  digits still looks phishing-like. The case is recorded rather than hidden.
* `models/url_model_revised.joblib` (+ `_metadata.json`) is the candidate. The
  served model is unchanged — still the original PHIUSIL model, retrained only
  so its feature schema matches the 33-feature extractor.

Files changed: `src/fraud_phishing/features/url_features.py` (33 features, brand
features, corrected docstring, duplicate shortener clause removed),
`src/fraud_phishing/features/feature_subsets.py` (new),
`src/fraud_phishing/models/url_model.py` (schema guard, revised save/load,
subset matrices), `src/fraud_phishing/explain/url_explainer.py` (respects the
model's own feature list, descriptions for the new features),
`src/fraud_phishing/data/download_data.py` (Datasets D and E),
`src/fraud_phishing/data/prepare_urls_augmented.py` (new),
`src/fraud_phishing/data/prepare_external_urls.py` (Datasets C and E),
`src/fraud_phishing/data/inspect_data.py` (inventory for D and E),
`scripts/url_feature_diagnosis.py`, `scripts/ablate_url_features.py`,
`scripts/compare_url_models.py` (all new), `scripts/train_url_model.py`.

---

## 6. Recommendation

1. **Do not promote the revised model on the strength of the development
   benchmark.** On the only clean external evidence it is not better than what
   is already served, and it trades recall for precision.
2. **Do not deploy the URL classifier as a real phishing control.** At ROC-AUC
   0.55 it is close to a coin flip on an independent corpus; its 99.7% F1 on
   PHIUSIL test data is a memorisation artifact.
3. To fix it properly, the input features must stop being dominated by transport
   and formatting facts that drift between corpora (`is_https` above all), and
   training needs a legitimate corpus that mixes HTTP and HTTPS with realistic
   paths and subdomains. A scheme-bearing legitimate corpus was not available
   within this pass; obtaining one is the prerequisite, not a tuning exercise.
4. Until then the honest description of this component is "research prototype
   with a measured generalisation failure", and the metric to quote is ROC-AUC
   on a held-out external corpus, never F1 on PHIUSIL.
