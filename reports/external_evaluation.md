# External URL Evaluation — LegitPhish

**Status:** complete · **Verdict:** recall generalises; **precision does not**.
**Artifacts:** `reports/url_metrics.json` → `external_legitphish`,
`reports/legitphish_external_stats.json`, `data/processed/legitphish_external.csv`.

> **Superseded in part — read §7 first.** LegitPhish was later re-classified as a
> *development benchmark*, and a genuinely untouched external set (URL-Phish)
> showed that neither the original nor the revised URL model generalises
> (ROC-AUC 0.55 / 0.53). Sections 1–6 below are kept as the historical record.

This is the independent check that PHIUSIL alone cannot provide: a second URL
dataset, from a different collection pipeline, that the model has **never** been
trained on.

---

## 1. The dataset

| Property | Value |
|---|---|
| Name | LegitPhish Dataset (Potpelwar, Kulkarni & Waghmare, 2025) |
| Source | `https://data.mendeley.com/datasets/hx4m73v2sf/2` (DOI `10.17632/hx4m73v2sf.2`) |
| File | `url_features_extracted1.csv`, 8,963,838 bytes |
| SHA-256 | `8685e7901c3b4e73eb43c106cb166cc932dd14f76960a032c713c2c32f09dbcd` (verified) |
| Rows | 101,219 raw → 100,872 after dropping 1 unlabelled row and 346 duplicate URLs |
| Phishing source | URLHaus and other threat repositories |
| Legitimate source | Wikipedia, Stack Overflow, high-authority `.gov`/`.edu`/`.mil` sites |
| License | CC BY 4.0 |

**Label convention.** The raw `ClassLabel` is `0 = phishing`, `1 = legitimate` —
the *same inverted convention as PHIUSIL*. This was verified two ways before use:

1. the dataset card states "1 = Legitimate, 0 = Phishing";
2. the observed counts match exactly (0 → 63,678, 1 → 37,540) and the raw label-1
   URLs resolve to `wikipedia.org`, `senate.gov`, `af.mil`, `ox.ac.uk`.

`prepare_external_urls.py` maps this to the project convention (0 = legitimate,
1 = phishing) and **fails hard** if the counts ever change, so the labels cannot
silently invert.

## 2. Overlap with PHIUSIL — LegitPhish is only *half* independent

This matters more than any metric below, and it was not obvious from the dataset
card. Comparing byte-for-byte against all 235,370 PHIUSIL URLs:

| LegitPhish class | Rows | Byte-identical to PHIUSIL | Genuinely new |
|---|---:|---:|---:|
| legitimate | 37,380 | **36,935 (98.81%)** | 445 (1.19%) |
| phishing | 63,492 | **0 (0.00%)** | 63,492 (100%) |

**LegitPhish's legitimate class is essentially a copy of PHIUSIL's legitimate
URLs.** Its phishing class is completely new (URLHaus/PhishTank), with 63% of all
rows on registered domains never seen in PHIUSIL training.

Consequence: the legitimate half of this set is *not* an out-of-distribution
sample. Only the 445 byte-distinct legitimate URLs (and the 63,492 phishing URLs)
test anything new. Precision measured on the full set is therefore optimistic and
is reported with that caveat rather than hidden.

## 3. Results

Both models were scored on the full 100,872-URL external set. Neutral thresholds
were the ones already tuned on PHIUSIL validation.

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production (random split) | 0.4612 | 0.9929 | 1.0000 | 0.9964 | 0.9941 | 453 | 0 |
| Domain-holdout | 0.6638 | 0.9930 | 1.0000 | 0.9965 | 0.9941 | 449 | 0 |

**Recall generalises completely.** On the 62,482 phishing URLs whose registered
domains never appear in PHIUSIL training, recall is **1.0000** (mean
phishing-probability 0.9999). Broken down by scheme:

| Phishing subset | n | Recall | Mean p |
|---|---:|---:|---:|
| `http://` | 58,677 | 1.0000 | 1.0000 |
| `https://` | 3,514 | 1.0000 | 0.9999 |
| no scheme | 1,301 | 1.0000 | 0.9949 |
| **unseen-domain, `https://` only** | **3,159** | **1.0000** | **0.9999** |

That last row is the important one: it removes the `is_https` shortcut entirely
(HTTPS is *not* the phishing signal in training) and the model still catches
100% of unseen HTTPS phishing URLs from a different dataset. **Recall is real.**

## 4. The failure: precision collapses the moment the legitimate URL is new

| Legitimate subset | n | False-positive rate | Mean p |
|---|---:|---:|---:|
| All LegitPhish legitimate | 37,380 | 1.21% (453) | 0.0165 |
| Byte-identical to PHIUSIL (memorised) | 36,935 | **0.02%** (8) | 0.0047 |
| **Byte-distinct (genuinely new)** | **445** | **100.00%** (445) | **0.9998** |

Every single legitimate URL that is new to the model is flagged as phishing.
Among them: `https://www.youtube.com/watch?v=…`, `https://stackoverflow.com/questions/tagged/php`,
`https://stackoverflow.com/questions/79030270/…`, `https://forbes.com`.

The 1.2% headline false-positive rate is an illusion created by memorisation: 98.8%
of the "legitimate" URLs were literally in the training pool.

### Root cause

The crosstab makes it unambiguous:

| | no path in URL | has a path |
|---|---:|---:|
| byte-identical (memorised) legit | 36,935 | 0 |
| byte-distinct legit | 5 | 440 |

Every PHIUSIL legitimate URL is a bare `https://domain` with no path
(`path_length = 0`). The model therefore learned the shortcut **"URL has a path →
phishing"** (plus "`http://` → phishing"). On real legitimate URLs — which
essentially always have a path — that heuristic fires every time.

This is the same defect the internal analysis found
(`reports/error_analysis.md`), now independently reproduced on data from a
different source: over 99% of the internal test F1 is carried by URL *formatting*
(`slash_count`, `path_length`, `is_https`), not by lexical similarity to real
phishing.

## 5. What this means for the recommendation

- **Recall (catching phishing) is trustworthy.** High, and it survives both
  domain isolation and a change of dataset, including HTTPS phishing.
- **Precision (avoiding false alarms) is not trustworthy.** Any legitimate URL
  with a path may be flagged. In production this is the more damaging error.
- **LegitPhish cannot fix the precision problem by itself.** Its legitimate
  class is 98.8% identical to PHIUSIL's, so folding it into training adds only
  445 genuinely new legitimate URLs. The training corpus still contains no
  realistic path-bearing legitimate URLs, which is *the* missing signal.

The fix has to be at the feature/model level: stop letting URL *formatting*
(`path_length`, `slash_count`, `is_https`, `query_param_count`) separate the
classes, and keep the lexical signals (hostname digits/hyphens, brand
impersonation, lookalike domains) that actually generalise. That is the P3 work
in `reports/implementation_checklist.md`.

## 6. Reproduce

```powershell
python src/fraud_phishing/data/download_data.py           # Datasets A + B + C + D + E
python src/fraud_phishing/data/prepare_external_urls.py   # LegitPhish + URL-Phish splits
python src/fraud_phishing/data/prepare_urls.py            # PHIUSIL splits
python src/fraud_phishing/data/prepare_urls_augmented.py  # PHIUSIL + MURL pool
python scripts/train_url_model.py                         # served, domain-holdout, revised
python scripts/evaluate_url_model.py                      # PHIUSIL test sets
python scripts/compare_url_models.py                      # all models, all four sets
```

The external datasets are **evaluation-only** and are never passed to
`train_url_model.py`.

---

## 7. Update after the P3 remediation — this verdict is superseded

LegitPhish was re-classified during P3 from "external evaluation" to
**development benchmark**, because results on it drove the diagnosis and the
choice of the revised feature subset. It is no longer a clean generalisation
claim. Two things changed the picture:

1. **The false-positive defect is diagnosed and partly fixed.** The 445
   legitimate URLs that are genuinely new to the model are flagged at **100.0%
   by the original model and 4.0% by the revised candidate** (mean P(phishing)
   0.9998 → below threshold). The cause is the conjunction **"HTTPS and a
   non-empty path"**, which occurred *only in the phishing class* in training
   data (0 legitimate vs 32,780 phishing rows), because PHIUSIL's legitimate
   URLs are all bare `https://domain` homepages and the MURL augmentation source
   stores no scheme. See `reports/url_remediation.md` §1.
2. **A truly untouched external set (URL-Phish, Dataset E) shows that neither
   model generalises.** ROC-AUC is **0.5496** for the served model and
   **0.5329** for the revised candidate — near chance — with false-positive
   rates of 94.07% and 66.06%. The survived-domain-isolation scores on PHIUSIL,
   and the LegitPhish figures in §2–§5 above, are therefore **not** evidence of
   working phishing detection.

| Set | Model | P | R | F1 | ROC-AUC | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|
| URL-Phish (final external) | served (random split) | 0.1511 | 0.9955 | 0.2624 | 0.5496 | 94.07% | 0.45% |
| URL-Phish (final external) | domain-holdout | 0.1510 | 0.9954 | 0.2621 | 0.5675 | 94.16% | 0.46% |
| URL-Phish (final external) | revised candidate | 0.1412 | 0.6459 | 0.2318 | 0.5329 | 66.06% | 35.41% |

The reason is a distribution shift in the scheme signal: legitimate HTTPS rate is
100% in PHIUSIL but 17.9% in URL-Phish, while phishing HTTPS rate is ~49% in
PHIUSIL but 86.2% in URL-Phish — and `is_https` is the model's most important
feature. LegitPhish's phishing class, being 92.4% HTTP, agrees with PHIUSIL and
not with the modern web.

Full numbers, stratified results and the recommendation are in
`reports/url_remediation.md` and `reports/url_model_comparison.md`. The revised
candidate is **not promoted** to production.
