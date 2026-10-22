# P7 Controlled URL Model Retraining and Internal Validation

## Scope

P7 trained candidate artifacts only. It did not read
`data/processed/compphish_external_frozen.csv`, modify served artifacts, change
the 33-feature schema, or promote a model.

## Development data and split

The approved fitting corpus is PHIUSIL plus MURL. LegitPhish and URL-Phish are
development diagnostics only and were not used as rows in the P7 split. MURL
had 402 exact URLs with contradictory labels in those diagnostics; they were
quarantined before deterministic class-capped sampling. The cap is 60,000 MURL
rows per class.

Rows are joined into connected components if they share either an offline-PSL
registered domain or the project's normalized template. `GroupShuffleSplit`
then holds out 15% of components with seed 42. This creates 224,121 components;
the largest contains 6,238 rows, so component-level splitting changes the class
mix and is intentionally not rebalanced.

| Partition | Rows | Legitimate | Phishing | PHIUSIL | MURL |
|---|---:|---:|---:|---:|---:|
| Train partition | 291,693 | 165,094 | 126,599 | 192,639 | 99,054 |
| Validation partition | 61,553 | 29,732 | 31,821 | 41,907 | 19,646 |

The fit uses a deterministic stratified 150,000-row sample of the train
partition, seed 42, for tractable baseline training. Assertions confirm zero
overlap between partitions for exact URL, conservative normalized URL,
registered domain, and normalized template.

## Candidates and validation results

Both candidates use the unchanged canonical 33 static URL features and project
labels `0 = legitimate`, `1 = phishing`. Threshold maximizes F1 on the held-out
validation partition only.

| Candidate | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP / legit | FN / phish |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.398084 | 0.8277 | 0.9373 | 0.8791 | 0.9366 | 0.9233 | 6,210 / 29,732 (20.89%) | 1,994 / 31,821 (6.27%) |
| RandomForest | 0.293361 | 0.8451 | 0.9789 | 0.9071 | 0.9720 | 0.9743 | 5,711 / 29,732 (19.21%) | 671 / 31,821 (2.11%) |

Random Forest has the better internal validation metrics. This does not justify
promotion: the validation set is drawn from the same approved development
sources and cannot establish performance on an independent contemporary corpus.
Historical served-model results are contextual only because they use a different
split and training protocol.

## Candidate artifacts

| Artifact | SHA-256 |
|---|---|
| `models/p7_url_logisticregression.joblib` | `6a27464bc1fbac8785a38f21381a6cbc8697e3e9cff8b4c319ffbd414580aa00` |
| `models/p7_url_randomforest.joblib` | `57a053267dc5f0e08c64508593a2c165cddb74a2f6acefa1407a46c121c11957` |

Corresponding metadata files contain class maps, feature schema fingerprint,
threshold, validation metrics, and an explicit candidate-only purpose. The
artifacts were loaded through the existing compatibility guard after saving.

## Validation

Two deterministic preparation runs produced identical SHA-256 values for train
and validation CSVs. `python -m pytest tests/test_p6_url_audit.py
tests/test_p6_compphish_freeze.py tests/test_p7_url_candidates.py -q` passed:
`7 passed`.

## Limitation and next step

The retained CompPhish set remains frozen and unscored. P7 does not establish
external generalization. Review this candidate baseline before approving P8 or
a single final evaluation against the frozen candidate.
