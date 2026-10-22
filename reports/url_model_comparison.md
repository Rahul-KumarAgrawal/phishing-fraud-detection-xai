# URL Model Comparison — original vs revised

Reproduce with `python scripts/compare_url_models.py`.

Positive class is **phishing**. Confusion matrix layout is `[[tn, fp], [fn, tp]]`. FPR is on legitimate URLs, FNR on phishing URLs.

| model | subset | n features | threshold | trained on |
|---|---|---:|---:|---|
| baseline (production) | `all` | 33 | 0.6294 | PHIUSIL |
| baseline (domain-heldout) | `all` | 33 | 0.6545 | PHIUSIL |
| revised | `no_path_no_query` | 24 | 0.3066 | data/processed/urls_aug_train.csv (+val for the final fit): PHIUSIL domain-isolated pool + MURL benign/legitimate/phishing |

## PHIUSIL random test (in-distribution)

| model | P | R | F1 | ROC-AUC | FP | FN | FPR | FNR | confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline (production) | 0.9997 | 0.9948 | 0.9973 | 0.9984 | 4 | 78 | 0.02% | 0.52% | [[20224, 4], [78, 15000]] |
| baseline (domain-heldout) | 0.9998 | 0.9954 | 0.9976 | 0.9991 | 3 | 69 | 0.01% | 0.46% | [[20225, 3], [69, 15009]] |
| revised | 0.9889 | 0.8836 | 0.9333 | 0.9612 | 150 | 1,755 | 0.74% | 11.64% | [[20078, 150], [1755, 13323]] |

## PHIUSIL template-clean test (unseen domains + templates)

| model | P | R | F1 | ROC-AUC | FP | FN | FPR | FNR | confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline (production) | 0.9998 | 0.9962 | 0.9980 | 0.9992 | 4 | 69 | 0.02% | 0.38% | [[20293, 4], [69, 17976]] |
| baseline (domain-heldout) | 0.9997 | 0.9959 | 0.9978 | 0.9986 | 6 | 74 | 0.03% | 0.41% | [[20291, 6], [74, 17971]] |
| revised | 0.9912 | 0.9032 | 0.9452 | 0.9699 | 144 | 1,746 | 0.71% | 9.68% | [[20153, 144], [1746, 16299]] |

## LegitPhish — development benchmark

| model | P | R | F1 | ROC-AUC | FP | FN | FPR | FNR | confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline (production) | 0.9930 | 1.0000 | 0.9965 | 0.9944 | 449 | 0 | 1.20% | 0.00% | [[36931, 449], [0, 63492]] |
| baseline (domain-heldout) | 0.9930 | 1.0000 | 0.9965 | 0.9943 | 448 | 0 | 1.20% | 0.00% | [[36932, 448], [0, 63492]] |
| revised | 0.9952 | 0.9625 | 0.9786 | 0.9874 | 293 | 2,378 | 0.78% | 3.75% | [[37087, 293], [2378, 61114]] |

Byte-identity stratification (the failure case from `reports/url_diagnosis.md`):

| model | FPR on new legitimate URLs | recall on unseen phishing |
|---|---:|---:|
| baseline (production) | **100.0%** (445 URLs) | 100.0% (63492 URLs) |
| baseline (domain-heldout) | **100.0%** (445 URLs) | 100.0% (63492 URLs) |
| revised | **4.0%** (445 URLs) | 96.3% (63492 URLs) |

## URL-Phish — FINAL external (untouched during development)

| model | P | R | F1 | ROC-AUC | FP | FN | FPR | FNR | confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline (production) | 0.1511 | 0.9955 | 0.2624 | 0.5496 | 92,790 | 75 | 94.07% | 0.45% | [[5851, 92790], [75, 16515]] |
| baseline (domain-heldout) | 0.1510 | 0.9954 | 0.2621 | 0.5675 | 92,880 | 77 | 94.16% | 0.46% | [[5761, 92880], [77, 16513]] |
| revised | 0.1412 | 0.6459 | 0.2318 | 0.5329 | 65,165 | 5,875 | 66.06% | 35.41% | [[33476, 65165], [5875, 10715]] |

Stratified (this dataset's HTTPS rate is inverted relative to PHIUSIL, so an `is_https` shortcut cannot pass it):

| model | benign FPR | phishing recall | phishing http recall | phishing https recall |
|---|---:|---:|---:|---:|
| baseline (production) | 94.07% | 99.55% | 100.00% | 99.51% |
| baseline (domain-heldout) | 94.16% | 99.54% | 100.00% | 99.49% |
| revised | 66.06% | 64.59% | 99.72% | 61.29% |

Integrity strata — 595 benign URLs are byte-identical to PHIUSIL training data (they cannot test generalisation), and 10,112 phishing URLs sit on a domain seen in PHIUSIL:

| model | P (excl. memorised) | R (excl. memorised) | F1 (excl. memorised) | P (unseen domain) | R (unseen domain) | F1 (unseen domain) |
|---|---:|---:|---:|---:|---:|---:|
| baseline (production) | 0.1511 | 0.9955 | 0.2624 | 0.0730 | 0.9895 | 0.1360 |
| baseline (domain-heldout) | 0.1510 | 0.9954 | 0.2621 | 0.0729 | 0.9892 | 0.1358 |
| revised | 0.1412 | 0.6459 | 0.2318 | 0.0511 | 0.4822 | 0.0924 |

## Which datasets influenced development

- **PHIUSIL (Dataset B)** — training data for the original model; also the source of the formatting shortcut identified in Step 1.
- **MURL (Dataset D)** — training augmentation only.
- **LegitPhish (Dataset C)** — development benchmark. The diagnosis and the choice of the revised feature subset came from it.
- **URL-Phish (Dataset E)** — untouched final external set. It was downloaded, its composition asserted, and then not inspected until the single evaluation above. No model, feature or threshold decision used it.

No model was trained on LegitPhish or URL-Phish.
