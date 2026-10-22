# Limitations

**Project:** AI-Based Fraud and Phishing Detection with Explainable AI
**Course:** CSET312, Bennett University

## P9 external-evaluation update (2026-10-19)

P9 evaluated the P7 Random Forest once on the frozen, group-excluded CompPhish
subset. It produced 98.46% false-positive rate (4,421/4,490 legitimate URLs),
0.31% false-negative rate (6/1,953 phishing URLs), ROC-AUC 0.3216, and F1
0.4680. The P7 candidate is not promotable. The frozen inputs and P9 results
are preserved unchanged. For the final college showcase, a byte-identical copy
of the same subset was explicitly reclassified as development data, so it is no
longer an independent benchmark for the newly integrated model. P10's
aggregate-only integrity audit remains valid for the historical P9 record. See
`reports/p9/`, `reports/p10/`, and `reports/showcase/`.

---

## Dataset Limitations

### Dataset A – SMS/Smishing (Mendeley)

- 5,971 messages total (4,844 ham / 638 smishing / 489 spam); a small dataset by
  NLP standards, so a single test-set error moves F1 by ~0.001 and ROC-AUC is
  reported at a resolution the sample size cannot really support.
- Downloaded automatically (the archive is fetched from the Mendeley public API
  and its SHA-256 is verified). Network access to Mendeley is still required; if
  Cloudflare blocks both `requests` and `urllib`, the downloader falls back to
  printed manual steps.
- Generic `spam` category (489 rows) is excluded from the binary experiment to
  avoid label ambiguity — treating generic spam as smishing would inflate the
  positive class with messages that contain no phishing mechanics.
- **Known label noise.** The true-class-negative example in
  `reports/error_analysis.md` is a stock-promotion spam message labelled
  `smishing`. The dataset is also documented in the literature as overlapping the
  SMS Spam Collection. Some residual error is therefore irreducible without manual
  re-labelling.
- Labels in the published file are inconsistently cased (`Smishing`/`smishing`,
  `spam`/`Spam`); normalisation is required and is asserted in `prepare_messages.py`.
- No temporal split; the model may overfit to the vocabulary of a specific
  collection period.
- Limited language diversity (primarily English).

### Dataset B – PHIUSIL URL Dataset (UCI)

- 235,795 records (134,850 legitimate / 100,945 phishing); 235,370 after removing
  425 exact duplicate URLs.
- **Raw labels are inverted relative to intuition** — the official card states
  `1 = legitimate`, `0 = phishing`. `prepare_urls.py` maps and asserts this, and
  `tests/test_url_split_leakage.py` guards it, because reading it backwards
  inverts every metric while still producing excellent-looking numbers.
- **The dataset's internal structure is a shortcut, and this is the single most
  important limitation of the project.** Every legitimate URL in PHIUSIL is a bare
  `https://domain` with no path and no query. As a result `is_https` (1.000 for
  legitimate vs 0.483 for phishing) and `slash_count` (2.000 vs 3.033) almost
  perfectly separate the classes without any phishing-specific reasoning. For a
  bare `https://host` the decision collapses onto hostname length and
  digit/hyphen content, which is why all 249 misclassified URLs have
  `slash_count == 2` and `path_length == 0`.
  **The reported >0.99 F1 does not imply real-world phishing-detection
  performance.** It is largely a property of the dataset, not of the model.
- Domain leakage has been **quantified and removed**: the previous "domain-aware"
  evaluation used a random-split model that shared 32,826 identical URLs and 86.2%
  of URL path templates with the test set (see `reports/url_leakage_report.json`).
  The current unseen-domain figures come from a model trained only on the
  domain-isolated pool, and the domain split has zero shared registered domains.
- URLs were collected at a specific point in time; phishing domains change rapidly,
  so no lexical model trained on this snapshot can track attacker drift.
- The dataset's existing webpage-derived feature columns are **not** used; only the
  raw URL string and the label are consumed.
- Domain/suffix extraction relies on `tldextract` and the public suffix list; rare
  or newly delegated suffixes may be parsed as unregistered domains.

### Dataset C – LegitPhish (development benchmark)

- 101,219 URLs (63,678 phishing / 37,540 legitimate); 100,872 after dropping 1
  unlabelled row and 346 duplicate URLs. Never used for training.
- Uses the **same inverted label convention as PHIUSIL** (`0` = phishing,
  `1` = legitimate); `prepare_external_urls.py` asserts this and fails hard if the
  counts ever change.
- **Role: development / diagnostic benchmark.** No model is trained on it, but it
  is the dataset that revealed the false-positive defect and it informed the
  choice of the revised feature subset. Its results are therefore **no longer a
  clean generalisation claim**; Dataset E fills that role.
- **LegitPhish is only half independent.** 98.8% of its *legitimate* URLs are
  byte-identical to PHIUSIL's, so a precision figure computed on that class is
  mostly a memorisation check. The 445 legitimate URLs that are genuinely new to
  the model are the ones the P3 diagnosis uses. Its *phishing* class (63,492
  URLs from URLHaus/PhishTank) has zero PHIUSIL overlap.
- Its phishing URLs are 92.4% `http://`, i.e. it agrees with PHIUSIL that "HTTP
  implies phishing". That is a property of the *era* it was collected in, not of
  phishing in general, and it is why Dataset E disagrees so sharply.

### Dataset D – MURL (training augmentation only)

- 783,705 rows; only `benign`, `legitimate` (→ 0) and `phishing` (→ 1) are used,
  capped at 60,000 per class. `defacement`, `malware`, `spam` and `Exploit` are
  dropped rather than folded into either class. Never used for evaluation.
- **URLs are stored without a scheme.** The extractor therefore reads
  `is_https = 0` for all 120,000 augmented rows. This is a property of the file,
  not of the URLs, and it is the reason MURL cannot fill the "HTTPS + path"
  region that PHIUSIL lacks. Normalising the scheme was tested and rejected —
  see `reports/url_remediation.md` §2.2.
- It is a different collection with a different source than PHIUSIL (0 exact URL
  overlap), but it shares 10,219 registered domains with it, so it is
  augmentation, not independent evidence.

### Dataset E – URL-Phish (development-informed diagnostic dataset)

- 116,600 rows (100,000 benign / 16,600 phishing); 115,231 after de-duplication.
- Uses the **intuitive** convention (`0` = benign, `1` = phishing), the opposite
  of PHIUSIL and LegitPhish; `prepare_urlphish()` asserts the composition and the
  convention.
- **Role: development-informed diagnostic data.** It has informed URL
  development decisions and has measurable overlap with the development corpus,
  so it cannot be presented as an untouched final test.
- Benign URLs are Research Organization Registry institution homepages; phishing
  URLs come from PhishTank. It is small in phishing terms (16,590 after dedup)
  and 595 of its benign URLs are byte-identical to PHIUSIL training data; those
  are reported as a separate integrity stratum rather than folded in.
- **Its scheme distribution is inverted relative to PHIUSIL**: 17.9% of benign
  URLs are HTTPS and 86.2% of phishing URLs are HTTPS. This is what makes it a
  genuinely hard test, and it is the reason both models score near chance on it.

---

## Model Limitations

### Message Classifier

- TF-IDF is a bag-of-words representation; it ignores word order and sentence
  structure, so "we will not charge your account" and "we will charge your
  account" are nearly identical.
- The model does not understand semantic meaning or context.
- New phishing vocabulary not seen during training will not be detected.
- Class imbalance is addressed via `class_weight="balanced"` and threshold tuning;
  the operating threshold (0.6181) was selected to maximise validation F1 and is
  not tuned to a cost model (a false negative is far more expensive than a false
  positive in practice).
- The selected candidate is `LinearSVC`, so the served probability comes from
  `CalibratedClassifierCV` (sigmoid). Calibration is fitted by internal
  cross-validation on a small corpus, so the probabilities are coarse.

### URL Classifier

- Features are purely lexical/static; no DNS, WHOIS, certificate, or page content
  is used, so a phishing page on a clean-looking domain is undetectable.
- **The final showcase model has no independent external evaluation.** It passed
  a grouped internal test after CompPhish was intentionally made development
  data (F1 0.9181, ROC-AUC 0.9799, FPR 9.59%). These figures are not expected
  production accuracy. URL-Phish remains development-informed (prior served
  ROC-AUC 0.5496; revised-candidate ROC-AUC 0.5329), while historical P9 found
  P7 poor on CompPhish (ROC-AUC **0.3216**, F1 **0.4680**, **98.46%** FPR).
  See `reports/showcase/`, `reports/url_remediation.md`, and `reports/p9/`.
- **Path- and query-bearing legitimate URLs remain difficult.** On the final
  grouped internal test, 2,496/9,118 legitimate path-bearing URLs were flagged
  (27.37% FPR) and 291/1,216 legitimate query URLs were flagged (23.93% FPR).
  This is materially lower than the preceding baseline's 100% subgroup FPR,
  but still too high for an operational security control.
- **Root cause of the original false-positive defect (fixed):** the conjunction
  **"HTTPS and a non-empty path" occurred only in the phishing class** in
  training data — 0 legitimate vs 32,780 phishing rows. PHIUSIL's legitimate
  URLs are all bare `https://domain` homepages and the augmentation source (MURL)
  stores no scheme, so the region was empty. Every legitimate URL in the
  development benchmark that had a path was therefore flagged:
  **100.0% of the 445 genuinely-new legitimate URLs**, mean P(phishing) 0.9998,
  against 0.02% on legitimate URLs the model had already seen. The apparent
  >0.99 F1 was memorisation.
- **The revision reduces that to 4.0%** (revised model, 445 URLs) but does not
  make the classifier trustworthy, for two reasons:
  - it loses phishing recall — 96.3% vs 100% on the development benchmark, and
    64.6% vs 99.6% on URL-Phish;
  - the dominant remaining feature is `is_https`, and the corpora disagree about
    it (legitimate HTTPS rate: PHIUSIL 100%, URL-Phish 17.9%; phishing HTTPS
    rate: PHIUSIL ~49%, URL-Phish 86.2%). PHIUSIL and LegitPhish were collected
    when phishing largely used plain HTTP and they reward "HTTP implies
    phishing"; URL-Phish is modern and inverts that. **No model selected using
    the available development data can learn the modern direction**, which is why
    the revised candidate is not promoted (see recommendation in
    `reports/url_remediation.md` §6).
- **The headline offline numbers must not be presented as expected production
  accuracy.** The realistic failure mode is a false-alarm storm on ordinary
  legitimate URLs (66–94% of institution homepages are flagged on URL-Phish),
  which a plain random split cannot reveal.
- A long legitimate path full of digits (e.g. a publication date) still scores
  slightly above the revised model's threshold; recorded as an xfail test.
- Brand-impersonation and lookalike-domain features are now implemented
  (`brand_token_count`, `brand_domain_mismatch`, `brand_lookalike_domain`) and
  unit-tested, but the fixed brand list bounds their coverage and they did not
  measurably change the 445-URL false-positive rate. `apple.stackexchange.com`
  is a known accepted false positive, because `apple` appears as a subdomain of
  a different registered domain.
- A clever attacker can craft a long URL with a low suspicious-token count.
- URL shorteners reduce feature information significantly.
- Random Forest can memorise training examples (mitigated by `min_samples_split`),
  though the domain-isolated evaluation shows memorisation is not driving the
  headline score.

---

## SHAP / Explainability Limitations

- SHAP values indicate correlation between features and predictions, not causation.
- **Message explanations are anchored to the calibrated probability.** The
  explainer runs `KernelExplainer` on the fitted `CalibratedClassifierCV`'s
  `predict_proba`, so `base_value + Σcontributions` equals the displayed
  `P(smishing)` exactly (measured gap `0.0`), and the explained class is
  resolved from `classes_`. The earlier implementation explained fold 1 of 3
  *uncalibrated* `LinearSVC`s, off the displayed score by 1.1557 on a ham sample.
- **Message explanations are a sampling approximation.** `KernelExplainer` draws
  `nsamples=100` coalitions, so the values are additive by construction but not
  exact. The two URL explainers remain exact for their model families.
- **The message SHAP baseline is an empty message** (an all-zero TF-IDF row), i.e.
  "the model's score with no information" — `P(smishing) = 0.0023` for the served
  model. That reference is out-of-distribution. A sample of real reference
  messages is available via `background_texts`, but it makes tokens that are
  absent from the input yet present in the reference appear as large
  contributions, which misleads in a UI.
- **Contribution sign is not token polarity.** Each calibration fold has its own
  coefficient vector, so the ensemble is not one monotone linear readout; a
  high-weight token can carry a negative contribution.
- **Units are stated, not assumed.** URL payloads declare `units`
  (`probability` for the served `TreeExplainer`, `log_odds` for the exact
  `LinearExplainer` on a logistic model), `explained_output`, `base_value` and
  the measured `additivity_gap`. Log-odds contributions map monotonically onto
  the same class probability but are **not** probability contributions, and the
  UI says so.
- **SHAP contributions and feature values are separate fields**
  (`shap_value` / `feature_value`). The previous payload reused a field named
  `value` for the SHAP value, which the UI rendered as if it were the feature's
  value, and rendered every contribution bar at width 0.
- No faithfulness or robustness evaluation of the explanations themselves
  (e.g. deletion/insertion curves) is included; the tests assert alignment,
  additivity and direction, not that the attributions are "good".
- Explanations do not prove that a message or URL is malicious.
- A feature flagged as suspicious (e.g. `login` in a URL) also appears in
  legitimate URLs.
- For URLs, the top SHAP contributors are dominated by `is_https` and
  `slash_count`. That is the explanation honestly surfacing the dataset shortcut —
  and a warning that the model's reasoning is degenerate on bare-URL inputs.

Full P4 detail: `reports/explainability_fixes.md`.

---

## Scope Limitations (Baseline)

- No authentication, user accounts, or persistent storage.
- No real-time URL reputation lookup (by design – URL destination is never visited).
- No email header analysis; only message body text.
- No image/attachment analysis.
- No multi-language support.
- No model monitoring or drift detection.
- No adversarial robustness evaluation.
- API rate limiting and production hardening are out of scope for this baseline.

---

## Reproducibility

- Dataset access may change over time (Mendeley / UCI URLs can break). The SMS
  archive is pinned by SHA-256 so a changed file is detected rather than silently
  trained on.
- Model performance depends on the exact dataset version downloaded.
- All random seeds are fixed (`RANDOM_SEED = 42`).
- Dependency versions are **pinned in `requirements.txt` but not enforced**: the
  development environment had drifted (e.g. scikit-learn 1.8.0 vs the pinned
  1.5.1, pandas 3.0.1 vs 2.2.2, shap 0.51.0 vs 0.45.1). Results are reproducible
  only for a matching environment; a lockfile or version check is recommended.
- Model artefacts record a feature-schema version, a `feature_schema_fingerprint`
  (a hash of the canonical schema, so an edit is caught even without a version
  bump) and their own feature-name list. `load_model()` /
  `load_domain_holdout_model()` / `load_revised_model()` require the metadata
  file, reject missing required keys, unknown/duplicate/reordered names, a
  `n_features` that disagrees with the list, an undeclared feature subset, a
  pipeline-width mismatch, a stale version or a stale fingerprint — so a change
  to `url_features.py` fails loudly instead of silently mis-mapping columns. The
  message loader applies the same policy to its metadata (`threshold`,
  `label_map`, `dataset_provenance`, `FEATURE_SPEC`), which previously could be
  absent and silently fall back to a `0.5` threshold.
- **The guards check compatibility, not correctness.** They cannot tell whether
  an artifact is any good, and a refusal is only as reliable as the metadata a
  future training run writes. Artefacts saved before P5 carry no fingerprint and
  rely on the structural checks alone. No hash of the `.joblib` bytes is stored,
  so a corrupted-but-loadable pipeline is out of scope.

Full P5 detail: `reports/p5_artifact_safeguards.md`.

---

*This baseline is an educational prototype for CSET312. It is not production-ready.*
