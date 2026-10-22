# P8 — Controlled URL Feature Experiments

## Scope and boundary

P8 used only `data/processed/p7_url_train.csv` and
`data/processed/p7_url_val.csv`: PHIUSIIL plus conflict-quarantined MURL. It
did not open `data/processed/compphish_external_frozen.csv`, its manifest, or
any CompPhish source file. It made no network, DNS, HTTP, or destination
request. The canonical 33 static features were not changed and no served model,
threshold, SMS code, or prior candidate was changed.

P7's split remains the development boundary: connected components formed by a
shared registered domain or normalized template were separated across its train
and validation partitions. Its recorded split checks are zero for exact URL,
conservative normalized URL, registered domain, and normalized template. P8
uses this split without regenerating it. The validation partition has already
been used to select P7 thresholds, so every validation result below is reused
development evidence, never independent performance.

## Baseline reproduction

P8 loaded P7 artifacts through the schema guard and reproduced their recorded
metrics on the recorded P7 validation partition. Their hashes still match P7:

| Candidate | SHA-256 | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP / legitimate | FN / phishing |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P7 Logistic Regression | `6a27464bc1fbac8785a38f21381a6cbc8697e3e9cff8b4c319ffbd414580aa00` | 0.398084 | 0.8277 | 0.9373 | 0.8791 | 0.9366 | 0.9233 | 6,210 / 29,732 (20.89%) | 1,994 / 31,821 (6.27%) |
| P7 Random Forest | `57a053267dc5f0e08c64508593a2c165cddb74a2f6acefa1407a46c121c11957` | 0.293361 | 0.8451 | 0.9789 | 0.9071 | 0.9720 | 0.9743 | 5,711 / 29,732 (19.21%) | 671 / 31,821 (2.11%) |

The P7 Random Forest is the diagnostic baseline in this report.

## Error diagnosis

The baseline's false positives are highly source coupled: 5,685 of 5,711
(99.54%) come from MURL. They are 60.16% of MURL legitimate validation rows
(5,685 / 9,450), versus 0.13% of PHIUSIIL legitimate rows (26 / 20,282).
This is measured association, not proof that `source` itself caused a decision;
source is not a model feature.

The corresponding structural associations are strong:

| Subgroup | Legitimate FP rate | Phishing FN rate | Evidence |
|---|---:|---:|---|
| Non-HTTPS | 60.16% | 3.40% | 9,450 legitimate / 18,302 phishing |
| HTTPS | 0.13% | 0.36% | 20,282 legitimate / 13,519 phishing |
| Non-empty path | 56.93% | 1.83% | 7,899 legitimate / 22,904 phishing |
| Bare host | 5.56% | 2.81% | 21,833 legitimate / 8,917 phishing |
| Query present | 75.16% | 2.61% | 2,460 legitimate / 1,800 phishing |
| Fragment present | 100.00% | 1.39% | 4 legitimate / 144 phishing; too small for a stable claim |

False positives are longer on average than correct records (46.64 versus 38.86
characters), have more digits (4.59 versus 3.00), and fewer hostname and
subdomain characters (17.01 / 3.27 versus 21.77 / 7.59). False negatives have
few suspicious tokens (mean 0.04; median 0). Neither observation assigns
causality to an individual feature. The complete static subgroup data and 25
stored-string examples per error type are in
`reports/p8/p8_error_subgroups.csv` and
`reports/p8/p8_representative_errors.csv`; URLs were not visited.

## Controlled ablations

All ablations retain the canonical extractor and remove only the named columns.
Exploration used a deterministic stratified 60,000-row sample of the P7 training
partition and three GroupKFold folds over connected domain/template components.
Fold metrics use the fixed 0.5 threshold, so the comparison does not tune a
separate threshold on each held-out fold. P7 validation results use max-F1
thresholds and are displayed separately.

| Ablation | Grouped CV F1 ± SD | CV ROC-AUC ± SD | CV FPR | Reused validation F1 | Reused validation FPR | Reused validation recall |
|---|---:|---:|---:|---:|---:|---:|
| All features | 0.8127 ± 0.0143 | 0.9252 ± 0.0084 | 14.53% | 0.8791 | 20.89% | 93.73% |
| Remove transport (`is_https`) | 0.7048 ± 0.0168 | 0.8540 ± 0.0098 | 14.54% | 0.8535 | 29.67% | 95.09% |
| Remove path/query group | 0.7929 ± 0.0145 | 0.8617 ± 0.0087 | 21.78% | 0.8597 | 30.64% | 96.97% |
| Remove length/character group | 0.7636 ± 0.0130 | 0.8304 ± 0.0258 | 26.84% | 0.8453 | 33.76% | 96.31% |
| Remove hostname/subdomain group | 0.7869 ± 0.0254 | 0.9067 ± 0.0147 | 13.65% | 0.8741 | 23.03% | 94.35% |
| Remove suspicious-token count | 0.8106 ± 0.0139 | 0.9225 ± 0.0105 | 14.89% | 0.8753 | 25.78% | 96.57% |
| Remove brand fields | **0.8146 ± 0.0143** | **0.9292 ± 0.0039** | 14.51% | 0.8789 | 20.94% | 93.73% |

Removing transport, path/query, length/character, hostname/subdomain, or the
suspicious-token field did not reduce reused-validation false positives while
preserving the baseline's overall behavior. The path/query result is especially
important: its poorer grouped-CV and reused-validation performance contradicts
the assumption that removing those fields alone fixes the problem.

The predeclared selection rule chose `without_brand`: highest grouped-CV mean
F1, then lower grouped-CV FPR, then more retained canonical features. It does
not claim that the three brand fields are harmful in general; the difference
from all features is small and only development data supports it.

## P8 candidates

Both P8 artifacts use the selected 30-feature subset, retain canonical order,
declare `feature_subset=without_brand`, and load successfully via the schema
guard. Thresholds maximize F1 on the already-used P7 validation data; they were
not chosen to reach a target false-positive rate.

| Candidate | SHA-256 | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix [[TN, FP], [FN, TP]] | FP / legitimate | FN / phishing |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| P8 LR without brand | `78f019026a4e1b1eb4c701e5857551b2a6e870cfdb32580f6586aad4cf475740` | 0.395872 | 0.8273 | 0.9373 | 0.8789 | 0.9367 | 0.9232 | [[23,505, 6,227], [1,994, 29,827]] | 6,227 / 29,732 (20.94%) | 1,994 / 31,821 (6.27%) |
| P8 RF without brand | `b3e839b0b0ead20fa9a6d030886f6ebbd95b1c65ac1f7ab5f7dec3efcb0c3344` | 0.468657 | 0.8781 | 0.9439 | 0.9098 | 0.9731 | 0.9754 | [[25,563, 4,169], [1,785, 30,036]] | 4,169 / 29,732 (14.02%) | 1,785 / 31,821 (5.61%) |

Compared with P7 RF, the P8 RF has 1,542 fewer false positives (a 5.19
percentage-point absolute FPR reduction) but 1,114 more false negatives (a
3.50 percentage-point absolute FNR increase). Its F1 rises by 0.0027 on reused
development validation. That operating trade-off is real in this split, but it
is not evidence of external improvement.

## Recommendation and limits

If the product priority accepts a 5.61% phishing miss rate in exchange for a
14.02% legitimate false-positive rate, use **P8 RF without brand** as the sole
recommended candidate for a future, single external evaluation. If recall near
P7 RF's 97.89% is mandatory, retain P7 RF as the recall-first candidate and do
not substitute the P8 RF on this evidence alone. Do not evaluate both candidates
iteratively against frozen CompPhish: that would consume the independent set.

The sources still expose a pronounced HTTP/HTTPS and path/query source-format
coupling, and repeated use of P7 validation limits confidence. The frozen
CompPhish candidate remains wholly unscored. No candidate is promoted.

## Reproducibility and checks

Run from the repository root:

```powershell
python scripts/run_p8_url_experiments.py
python -m pytest tests/test_p6_url_audit.py tests/test_p6_compphish_freeze.py tests/test_p7_url_candidates.py tests/test_p8_url_experiments.py -q
```

The test run completed with `9 passed`. The structured, full-precision evidence
is in `reports/p8/p8_experiments.json`.
