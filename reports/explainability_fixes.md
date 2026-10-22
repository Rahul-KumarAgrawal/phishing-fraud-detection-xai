# Explainability Correctness Fixes (P4)

**Scope:** SHAP and prediction-explanation correctness only.

**Not changed:** the served models (`models/url_model.joblib`,
`models/message_model.joblib`), any classification threshold, training data,
features, or hyperparameters. `models/url_model_revised.joblib` remains an
un-promoted candidate. No dataset was downloaded, re-split, or scored in this
pass, and the previously evaluated URL-Phish external set was **not** touched.

**Result:** the explanation payload now describes the same class, the same model
output and the same units that the UI displays, and says so explicitly. The full
test suite is **135 passed, 2 skipped, 1 xfailed** (was 113 passed).

---

## 1. Defects found

| # | Defect | Evidence | Status |
|---|---|---|---|
| 1 | The message explainer unwrapped the `CalibratedClassifierCV` and explained **one fold's uncalibrated `LinearSVC` decision function** while the API displayed the calibrated probability | §2 | Fixed |
| 2 | SHAP contributions were surfaced in a field named `value`, which reads as "the feature's value" | §3 | Fixed |
| 3 | The URL payload never stated *what* the contributions add up to or in which units | §4 | Fixed |
| 4 | Word and character n-grams can share a raw name (e.g. `no`), so a reported `feature` name could map to two different TF-IDF columns | §5 | Fixed |
| 5 | The `KernelExplainer` fallback in the message explainer unpacked a `(n_samples, n_features, n_classes)` return value as if it were `(n_samples, n_features)`, producing a 2-D "contribution" vector | §5 | Fixed |
| 6 | Message feature values were never reported at all, so the UI printed the SHAP value in the feature-value slot | §3 | Fixed |

Defects 1-3 were in scope. Defects 4-6 were found while verifying item 4 of the
scope ("verify labels, probabilities, SHAP directions and UI descriptions are
consistent") and are correctness bugs, not refactors.

---

## 2. Defect 1 - the message explanation described the wrong function

### What the code did

`message_explainer._extract_linear_clf()` walked
`clf.calibrated_classifiers_`, took the **first** inner estimator with a `coef_`
attribute, and ran `shap.LinearExplainer` on it. Its docstring said this was
"supported for LR and CalibratedClassifier wrapping a linear model".

### What that produces (measured, served artifact)

The served message classifier is `CalibratedClassifierCV(LinearSVC, cv=3,
method="sigmoid")` - three fold models, each a `LinearSVC` with its own
coefficient vector, averaged through three sigmoid calibrators.

| | visible score `P(smishing)` | fold-1 decision | fold-2 decision | fold-3 decision | old `base + sum(shap)` | \|old explanation - visible score\| |
|---|---|---|---|---|---|---|
| ham sample | 0.000207 | -1.1555 | -1.1769 | -1.2490 | **-1.1555** | **1.1557** |
| smishing sample | 0.999414 | +0.9332 | +0.7051 | +0.6446 | **+0.9332** | **0.0662** |

`base + sum(shap)` equals the fold-1 decision function exactly (it is additive
to it, as `LinearExplainer` guarantees), so the old explanation was an exact
decomposition of a quantity that is:

* not the displayed score (off by 1.16 on the ham example),
* in log-odds units, not probability units, and
* not even the ensemble mean: fold means differ from fold 1 by 0.038 / 0.172.

Direction and ranking were preserved, which is why this went unnoticed.

### The fix

`message_explainer` now explains `clf.predict_proba` - the fitted classifier,
calibration wrapper included - with a model-agnostic `KernelExplainer`, and
resolves the positive-class column from `clf.classes_` instead of assuming
column 1 (`explain/shap_utils.py`).

`KernelExplainer` is affordable here because SHAP restricts the solve to the
feature columns that actually vary between the input and the background, which
for sparse TF-IDF is a few hundred of 80,000 columns: **≈0.15 s per message**.
`l1_reg=0` is passed explicitly, which disables shap's deprecated, lossy
Lasso-based attribution sparsification.

### Measured result

| | score | `base_value` | `additivity_gap` = \|base + Σshap − score\| | units |
|---|---|---|---|---|
| ham sample | 0.000207 | 0.002321 | 3.5e-19 | probability |
| smishing sample | 0.999414 | 0.002321 | 0.0 | probability |

The gap is exact to floating point, and it is exact **against the displayed
probability**, which is the invariant the previous code violated by 1.16.

---

## 3. Defect 2 / 6 - the misleading `value` field

### Before

```jsonc
// POST /api/analyze/message  (old payload)
"top_positive_features": [
  { "feature": "now", "value": 0.5046, "direction": "increases_risk" }
]
```

* `value` was the **SHAP contribution**, not the feature's value.
* `shap_value` was `null`, so the UI rendered the contribution bar at **width 0**
  and printed the SHAP value in the `val=` chip.
* The feature's actual value (its TF-IDF weight / URL feature count) was not
  reported at all for messages.

### After

```jsonc
// POST /api/analyze/message
"explanation": {
  "explanation_type": "KernelExplainer",
  "units": "probability",
  "explained_output": "P(smishing)",
  "base_value": 0.002321,
  "additivity_gap": 0.0,
  "top_positive_features": [
    { "feature": "word__now", "shap_value": 0.5046, "feature_value": 0.1208,
      "direction": "increases_risk",
      "description": "TF-IDF weight of the word n-gram 'now' in this message" }
  ],
  "top_negative_features": [
    { "feature": "char__.x", "shap_value": -0.0998, "feature_value": 0.0844,
      "direction": "decreases_risk",
      "description": "TF-IDF weight of the char n-gram '.x' in this message" }
  ]
}
```

`value` is removed from the schema entirely. `FeatureContribution` is now
`{feature, shap_value, feature_value, direction, description}` and the response
schema additionally carries `units`, `explained_output`, `base_value` and
`additivity_gap`. The UI labels the two numbers separately (`value=` chip for the
input feature value, a signed contribution for the SHAP value), prints the
baseline and units in the panel header, and warns explicitly when the units are
log-odds.

---

## 4. Defect 3 - the URL payload did not say what it explained

The URL explainer's class handling was already correct (`TreeExplainer` returns
`(n_samples, n_features, n_classes)` for a binary `RandomForestClassifier` and
the code selected `[0, :, 1]`, i.e. the phishing class, in probability units -
verified: `base + Σshap = 1.0000` vs `P(phishing) = 1.0000`). It nevertheless did
not state any of that, so a reader could not tell a probability contribution
from a log-odds contribution.

It now reports:

| model family | explainer | units | `explained_output` | additivity |
|---|---|---|---|---|
| `RandomForestClassifier` (served) | `TreeExplainer` | `probability` | `P(phishing)` | gap 3.1e-15 vs `predict_proba` |
| `LogisticRegression` (revised candidate, not promoted) | `LinearExplainer` | `log_odds` | `log-odds of P(phishing)` | gap 2.2e-16 vs `decision_function` |

For a logistic model SHAP is exact but lives in log-odds; those values map onto
the displayed probability through a sigmoid, so the class is still aligned while
the *magnitudes* are not probability contributions. That is now stated in the
payload (`units: "log_odds"`) and in the UI instead of being implied.

---

## 5. Defects 4 / 5 - silent inconsistencies found while verifying

**Duplicate feature names.** `FeatureUnion(["word", "char"])` puts both analyzers
in one namespace, so the word feature `no` and the character 2-gram `no` were
reported under the same name `no` while being different TF-IDF columns. Display
names are now block-prefixed (`word__no`, `char__no`), which makes every reported
feature uniquely identifiable. `tests/test_explanation_consistency.py` asserts
name uniqueness.

**Broken fallback unpacking.** The old message fallback did
`sv[0]` on a `(1, 80000, 2)` array, yielding a 2-D "contribution" vector of shape
`(80000, 2)`, and hard-coded class index 1. The new
`shap_utils.select_class_shap_values()` is shape- and class-aware, and is
unit-tested against all four shapes SHAP may return (list of per-class arrays,
3-D, 2-D, 1-D).

---

## 6. Invariants now enforced and tested

`src/fraud_phishing/explain/shap_utils.py` provides:

* `positive_class_index(clf)` - resolves the positive class from `classes_`
  (fallback 1, then last column), so a model with reordered classes cannot
  silently explain the wrong class.
* `select_class_shap_values(values, class_index)` - one instance, one class,
  every supported shape.
* `additivity_gap(base, shap, model_output)` - `|base + Σshap − model_output|`,
  reported in the payload so the invariant is verifiable by any consumer.

`tests/test_explanation_consistency.py` (22 tests) pins:

| Invariant | Test |
|---|---|
| class index comes from `classes_`, not a hard-coded 1 | `test_positive_class_index_follows_classes_not_a_hardcoded_1` |
| every supported SHAP shape unpacks correctly | `test_select_class_shap_values_handles_every_supported_shape` |
| tree model is additive to `P(phishing)`, not `P(legitimate)` | `test_url_explainer_explains_tree_model_in_probability_units` |
| linear model is additive to log-odds and agrees with the sigmoid | `test_url_explainer_explains_linear_model_in_log_odds_units` |
| direction flips with the class prediction | `test_url_explainer_direction_follows_positive_class` |
| `score` == `predict_proba(...)[1]`, `label` == `int(score >= threshold)`, for **both** classes | `test_message_score_and_label_mapping[ham/smishing]` |
| `base + Σall shap` == displayed `P(smishing)` for both classes | `test_message_shap_adds_up_to_the_displayed_probability[ham/smishing]` |
| the explanation is **not** the uncalibrated SVM (|Δ| > 0.05) while it is the calibrated probability | `test_message_explanation_does_not_explain_the_uncalibrated_svm` |
| every direction matches the sign of its SHAP value | `test_message_direction_agrees_with_shap_sign`, `test_url_direction_agrees_with_shap_sign` |
| `feature_value` equals the real TF-IDF weight / extractor value, and is not the SHAP value | `test_message_feature_value_is_the_tfidf_weight_not_the_shap_value`, `test_url_feature_value_matches_the_feature_extractor` |
| the API payload has no `value` field and is additive | `test_api_message_explanation_payload_is_unambiguous`, `test_api_url_explanation_payload_is_unambiguous` |

The URL micro-model tests exist because the served URL model currently scores
**every** input as phishing (the P3 pathology, deliberately not fixed in this
pass), so it cannot exercise both classes end-to-end.

---

## 7. Before/after, end to end

Message `"URGENT! Your bank account is suspended. Verify now at
http://hsbc-secure-verify-account.xyz/login"` (real API payload):

| | before | after |
|---|---|---|
| explained object | fold-1 `LinearSVC` decision function | calibrated `predict_proba` |
| units | log-odds (unlabelled) | `probability` (`explained_output: P(smishing)`) |
| contribution field | `value` | `shap_value` + `feature_value` |
| additive to the displayed score | no (off by 0.066) | yes (gap 0.0) |
| contribution bar in the UI | always width 0 | true relative magnitude |
| feature value shown | the SHAP value | the token's TF-IDF weight |

URL `https://www.google.com/maps/place/Paris/@48.8566,2.3522,12z`:

| | before | after |
|---|---|---|
| score / label | 1.0000 / phishing | unchanged (1.0000 / phishing) |
| units | unstated | `probability`, `P(phishing)`, `base_value 0.5001` |
| contribution field | `value` | `shap_value` + `feature_value` |
| additivity checked | no | yes (gap 3.1e-15) |
| top contributors | `slash_count`, `path_length`, `digit_ratio` | identical values, now labelled with the real feature values 6, 37, 0.2203 |

The verdicts in §7 are unchanged by this pass: P4 did not touch a model. In
particular the URL model still flags legitimate URLs, which is the known P3
generalisation failure.

---

## 8. Verification

```text
python -m pytest tests/ -q
135 passed, 2 skipped, 1 xfailed in 27.35s     # was 113 passed, 2 skipped, 1 xfailed

frontend: npm run lint  -> clean
frontend: npm run build -> 20 modules transformed, built in 234ms
```

API verified through `TestClient` against the real artifacts: both models load,
`explanation.top_positive_features` is non-empty and every emitted item has
`shap_value`/`feature_value`/`direction`/`description` with no `value` key.

---

## 9. Remaining explainability limitations

1. **Message SHAP uses a model-agnostic approximation.** `KernelExplainer`
   samples coalitions (`nsamples=100`); the returned values are additive to the
   displayed probability by construction but are not exact. `TreeExplainer` and
   `LinearExplainer` remain exact for the two URL model families.
2. **The message SHAP baseline is an empty message.** The reference point is an
   all-zero TF-IDF row, i.e. "the model's score with no information"
   (`P(smishing) = 0.0023` for the served model). An empty document is
   out-of-distribution; the alternative (a sample of real reference messages)
   makes absent-but-referenced tokens appear as large contributions, which reads
   as nonsense in a UI. The default is therefore the empty message, and
   `background_texts` is available to callers who want a data sample.
3. **Contributions are not token polarity.** For the calibrated model, a token
   with a large TF-IDF weight can carry a negative contribution; each fold has
   its own coefficient vector, so the ensemble is not a single monotone linear
   readout.
4. **Log-odds vs probability for linear URL models.** The exact
   `LinearExplainer` reports log-odds. Labelled as such in the payload and in the
   UI, but the numbers are not comparable with the probability contributions of
   the served `RandomForestClassifier`.
5. **SHAP explains the model, not reality.** The top URL contributors remain
   `is_https`, `slash_count` and `path_length` - the explanation honestly
   surfacing the dataset artifact documented in `reports/url_remediation.md`.
6. **No faithfulness/robustness evaluation** of the explanations themselves
   (e.g. deletion curves) is included; the tests assert alignment invariants,
   not that the attributions are "good".

---

*Generated from the P4 measurement runs. Test counts and payload examples are the
actual output of the commands in §8.*
