# P9 — Frozen CompPhish External Evaluation

## Fixed evaluation protocol

This is the first and only P9 evaluation of the frozen CompPhish candidate. It
uses only `models/p7_url_randomforest.joblib`, loaded with its saved metadata,
full 33-feature schema, class order `[0, 1]`, and fixed saved threshold
`0.2933606675557894`. Label mapping is `0 = legitimate`, `1 = phishing`.

No model was fit, refit, recalibrated, threshold-tuned, or compared with a
second model. In particular, the P8 candidate was not read for this evaluation.
URLs were parsed statically; no URL destination, DNS record, or web resource
was accessed.

## Input verification

| Input | Verification |
|---|---|
| Frozen CSV | SHA-256 before and after: `56f6e8545a7df5f675b740d0858ee0fb4b2fa920a085d4aceae2838bfbc0e377` |
| Frozen source | CompPhish v4, from `Mapping_File.xlsx`; source mapping hash `e6dda6a21d925f8aba36090f295b08e9808a5dfe8dd3276bf8cd82022855f14a` |
| Frozen labels | Verified binary labels: `0 = legitimate`, `1 = phishing` |
| Frozen rows | 6,443 total: 4,490 legitimate and 1,953 phishing, matching the manifest |
| P7 RF model | SHA-256 `57a053267dc5f0e08c64508593a2c165cddb74a2f6acefa1407a46c121c11957` |
| P7 RF metadata | SHA-256 `c021014bc8dbcfec1cbb277d6295fd6194b32be272d806d278ef2942f29d1c22` |
| Feature compatibility | Metadata declares the canonical ordered 33 features, schema fingerprint `1de7363ea0a84a41`; compatibility guard loaded successfully |

The frozen CSV retained its source `serial_number` column with `url` and
`label`; that provenance column was not a model input.

## Measured external metrics

| Metric | Result |
|---|---:|
| Precision | 0.3057 |
| Recall | 0.9969 |
| F1 | 0.4680 |
| Accuracy | 0.3129 |
| Balanced accuracy | 0.5061 |
| ROC-AUC | 0.3216 |
| PR-AUC | 0.2260 |
| False-positive rate | 4,421 / 4,490 = **98.46%** |
| False-negative rate | 6 / 1,953 = **0.31%** |

Confusion matrix, rows actual and columns predicted:

| Actual class | Predicted legitimate | Predicted phishing |
|---|---:|---:|
| Legitimate | TN = 69 | FP = 4,421 |
| Phishing | FN = 6 | TP = 1,947 |

The model predicted 75 URLs as legitimate and 6,368 as phishing. Its phishing
probabilities are concentrated near one: mean 0.9762, median 1.0000, 5th
percentile 0.9132, and 95th percentile 1.0000. For legitimate rows alone, the
mean is 0.9761 and median is 1.0000; for phishing rows, the mean is 0.9765 and
median is 0.9950. This explains the extreme false-positive rate at the fixed
P7 threshold.

## Interpretation and limits

This result does **not** support promotion of P7 Random Forest. It shows severe
external misclassification of retained CompPhish legitimate URLs despite near
complete phishing recall. P9 did not change the threshold after observing this
outcome, because that would turn the frozen evaluation into tuning data.

The frozen subset was intentionally strict: records sharing an exact URL,
conservatively normalized URL, hostname, registered domain, or normalized URL
template with the P6 development boundary were excluded. The recorded
post-filter overlap is zero for all five levels. These exclusions changed the
population from 15,351 unique URLs (8,154 legitimate; 7,197 phishing) to 6,443
retained URLs (4,490 legitimate; 1,953 phishing). Therefore the result applies
to this filtered CompPhish subset, not the original source as a whole.

CompPhish structural distributions were examined in P6.1 before model fitting
or model-specific selection, and the frozen subset was not used for fitting,
feature selection, threshold selection, or debugging. That prior structural
exposure means P9 is not fully blind external validation. In addition, the
CompPhish v4 mapping source and the strict exclusions can impose selection bias;
neither this dataset nor its retained subset is perfectly representative of
real-world traffic.

The proper next step is to collect or qualify a new independently sourced
benchmark before any further final evaluation. Do not tune P7 against this
frozen result or evaluate P8 against it.

## Validation

The evaluator verifies input hashes, manifest counts and zero-overlap claims,
label mapping, class order, model/schema compatibility, and the frozen hash
again after inference. Its tests include a small synthetic metric fixture and a
static assertion that the evaluation code has neither model fitting nor
threshold-selection code.

Structured evidence: `reports/p9/p9_frozen_compphish_metrics.json`.
