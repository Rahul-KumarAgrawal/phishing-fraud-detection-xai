# URL Feature Ablation

Reproduce with `python scripts/ablate_url_features.py`.

Training pools: **phiusil** (`urls_template_train.csv`) and **augmented** (`urls_aug_train.csv`, PHIUSIL + MURL). Both use a domain-held-out threshold slice. Models are fitted on a stratified 150,000 sample; RandomForest uses 150 trees here (production uses 200) for runtime.

**URL-Phish is not used here** - it is the untouched final external set.

`newLegitFPR` = false-positive rate on the 445 legitimate URLs that are byte-distinct from PHIUSIL (the failure case in `reports/url_diagnosis.md`). `unseenPhishR` = recall on the 63,492 phishing URLs that are byte-distinct from PHIUSIL (62,482 of them are also on a domain PHIUSIL never had).

| pool | subset | n feat | model | val F1 | PHIUSIL test F1 | dev F1 | **new legit FPR** | unseen phish recall |
|---|---|---:|---|---:|---:|---:|---:|---:|
| phiusil | all | 33 | RandomForest | 0.9976 | 0.9977 | 0.9965 | **100.0%** | 100.0% |
| phiusil | all | 33 | LogisticRegression | 0.9947 | 0.9947 | 0.9958 | **99.1%** | 100.0% |
| phiusil | no_path_features | 30 | RandomForest | 0.9964 | 0.9967 | 0.9964 | **98.0%** | 100.0% |
| phiusil | no_path_features | 30 | LogisticRegression | 0.9895 | 0.9910 | 0.9953 | **92.8%** | 100.0% |
| phiusil | no_path_no_query | 24 | RandomForest | 0.9964 | 0.9966 | 0.9963 | **98.0%** | 100.0% |
| phiusil | no_path_no_query | 24 | LogisticRegression | 0.9896 | 0.9910 | 0.9953 | **92.8%** | 100.0% |
| phiusil | no_path_no_length | 26 | RandomForest | 0.9560 | 0.9251 | 0.9922 | **82.5%** | 99.2% |
| phiusil | no_path_no_length | 26 | LogisticRegression | 0.9196 | 0.8996 | 0.9777 | **71.2%** | 99.1% |
| phiusil | no_path_query_length | 20 | RandomForest | 0.9559 | 0.9246 | 0.9921 | **81.3%** | 99.2% |
| phiusil | no_path_query_length | 20 | LogisticRegression | 0.9191 | 0.8994 | 0.9773 | **71.0%** | 99.1% |
| phiusil | no_shape_features | 23 | RandomForest | 0.9475 | 0.9115 | 0.9928 | **58.9%** | 99.1% |
| phiusil | no_shape_features | 23 | LogisticRegression | 0.9081 | 0.8664 | 0.9868 | **39.6%** | 98.9% |
| phiusil | no_https | 32 | RandomForest | 0.9935 | 0.9934 | 0.9954 | **100.0%** | 100.0% |
| phiusil | no_https | 32 | LogisticRegression | 0.9527 | 0.9654 | 0.9823 | **99.6%** | 99.8% |
| phiusil | no_brand | 30 | RandomForest | 0.9976 | 0.9976 | 0.9965 | **100.0%** | 100.0% |
| phiusil | no_brand | 30 | LogisticRegression | 0.9947 | 0.9948 | 0.9958 | **99.1%** | 100.0% |
| phiusil | consistent_only | 16 | RandomForest | 0.9420 | 0.9069 | 0.9927 | **56.6%** | 99.1% |
| phiusil | consistent_only | 16 | LogisticRegression | 0.8994 | 0.8555 | 0.9882 | **36.2%** | 98.8% |
| phiusil | consistent_no_https | 15 | RandomForest | 0.7456 | 0.7495 | 0.9432 | **65.2%** | 98.8% |
| phiusil | consistent_no_https | 15 | LogisticRegression | 0.6403 | 0.6692 | 0.9197 | **47.0%** | 98.0% |
| augmented | all | 33 | RandomForest | 0.8767 | 0.9973 | 0.9919 | **100.0%** | 99.1% |
| augmented | all | 33 | LogisticRegression | 0.8142 | 0.9763 | 0.9884 | **42.9%** | 98.1% |
| augmented | no_path_features | 30 | RandomForest | 0.8664 | 0.9935 | 0.9670 | **97.5%** | 94.3% |
| augmented | no_path_features | 30 | LogisticRegression | 0.7706 | 0.9259 | 0.9778 | **3.6%** | 96.0% |
| augmented | no_path_no_query | 24 | RandomForest | 0.8621 | 0.9942 | 0.9682 | **97.8%** | 94.5% |
| augmented | no_path_no_query | 24 | LogisticRegression | 0.7707 | 0.9429 | 0.9781 | **4.9%** | 96.3% |
| augmented | no_path_no_length | 26 | RandomForest | 0.8435 | 0.9183 | 0.9692 | **82.2%** | 94.9% |
| augmented | no_path_no_length | 26 | LogisticRegression | 0.7393 | 0.7104 | 0.9763 | **4.7%** | 97.1% |
| augmented | no_path_query_length | 20 | RandomForest | 0.8363 | 0.9166 | 0.9562 | **81.3%** | 92.5% |
| augmented | no_path_query_length | 20 | LogisticRegression | 0.7388 | 0.6993 | 0.9792 | **3.6%** | 96.9% |
| augmented | no_shape_features | 23 | RandomForest | 0.7806 | 0.8986 | 0.9723 | **64.9%** | 96.6% |
| augmented | no_shape_features | 23 | LogisticRegression | 0.7318 | 0.7206 | 0.9795 | **8.1%** | 97.1% |
| augmented | no_https | 32 | RandomForest | 0.8706 | 0.9921 | 0.9910 | **85.4%** | 99.0% |
| augmented | no_https | 32 | LogisticRegression | 0.7362 | 0.8613 | 0.9540 | **57.8%** | 97.6% |
| augmented | no_brand | 30 | RandomForest | 0.8786 | 0.9974 | 0.9923 | **100.0%** | 99.2% |
| augmented | no_brand | 30 | LogisticRegression | 0.8150 | 0.9779 | 0.9889 | **43.8%** | 98.2% |
| augmented | consistent_only | 16 | RandomForest | 0.7774 | 0.8899 | 0.9651 | **62.7%** | 94.5% |
| augmented | consistent_only | 16 | LogisticRegression | 0.7274 | 0.7056 | 0.9800 | **4.9%** | 97.0% |
| augmented | consistent_no_https | 15 | RandomForest | 0.6826 | 0.6809 | 0.9517 | **44.5%** | 95.7% |
| augmented | consistent_no_https | 15 | LogisticRegression | 0.5931 | 0.6516 | 0.9058 | **43.6%** | 97.3% |
