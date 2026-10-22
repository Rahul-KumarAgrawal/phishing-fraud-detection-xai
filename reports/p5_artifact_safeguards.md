# P5 – Correctness Cleanup and Artifact Safeguards

**Status:** implemented and validated. Nothing committed, no model promoted, no
threshold, training data, feature definition or hyperparameter changed.
**Date:** 2026-10-19

---

## 1. What the scope asked for, and what was found

| # | P5 item | Outcome |
|---|---|---|
| 1 | Correct the stale feature-count documentation in `url_features.py` | **Already correct** (fixed in P2). Verified against the code and now pinned by a test so it cannot drift again |
| 2 | Remove the redundant URL-shortener check without changing behaviour | **Already removed** (P2). Verified against `HEAD`. Behaviour unchanged |
| 3 | Add artifact / feature-schema compatibility checks so stale model files cannot run silently | **Added.** The guard was incomplete: a *missing* metadata file loaded silently with the default `0.5` threshold, the message model had no guard at all, and a feature inserted into the middle of `FEATURE_NAMES` was undetectable |
| 4 | Review the breaking explanation API change (removed `value` field) | **Done.** No backend, frontend, test or documented example depends on it; the only occurrences are labelled "before" payloads and assertions that the field is absent |
| 5 | Keep metadata and documentation consistent with the code and artifacts | **Done.** Docstrings, `README.md`, `limitations.md`, `error_analysis.md`, the checklist and `url_remediation.md` updated |

Items 1 and 2 were listed as open in the original audit but were closed during
P2. They are reported here as **verified, not re-changed** – no cosmetic edit was
made to code that was already correct.

### Evidence for item 1 (`HEAD` → working tree)

```diff
-Features (28 total):
+Features (33 total):
   Group 1 – Length & shape (5)
-  Group 2 – Character counts (14)
+  Group 2 – Character counts (13)
   Group 3 – Domain structure (3)
   Group 4 – URL components (3)
   Group 5 – Boolean / lexical indicators (6)
+  Group 6 – Brand impersonation (3)
```

Current counts, re-derived from the tuple: `5 + 13 + 3 + 3 + 6 + 3 = 33`, matching
`FEATURE_NAMES` exactly.

### Evidence for item 2 (`HEAD` → working tree)

```diff
-    is_url_shortener = int(
-        regd_lower in URL_SHORTENERS
-        or any(regd_lower == s for s in URL_SHORTENERS)
-    )
+    is_url_shortener = int(regd_lower in URL_SHORTENERS)
```

The removed clause is logically identical to the membership test, so the feature
value is unchanged for every input (see `tests/test_url_features.py`).

---

## 2. Artifact / feature-schema guards (item 3)

### The failure modes

1. **Metadata file absent.** `load_model()` returned `(pipeline, {})`. Every
   caller then fell back to `threshold = 0.5` and the full 33-feature schema —
   so the *revised* model (24-feature subset, threshold 0.3066) would have been
   fed the wrong columns and decided at the wrong cut-off, and the served model
   would have decided at 0.5 instead of 0.6294.
2. **The message model had no guard at all.** Same silent `0.5` fallback, and no
   check that the artifact was built with the current TF-IDF specification.
3. **A feature inserted in the middle of `FEATURE_NAMES`** shifts every later
   column but leaves the *relative* order of the recorded names intact, and the
   recorded count, width and schema version could all still agree. The old
   checks (unknown names + relative order) passed while the model read the wrong
   column for every feature after the insertion point.

### The checks now enforced

| Check | URL model | Message model |
|---|---|---|
| Metadata file must exist and be a JSON object | ✅ `FileNotFoundError` / `RuntimeError` | ✅ |
| Metadata must not be empty | ✅ | ✅ |
| Required keys must be present | ✅ `feature_names`, `n_features`, `feature_schema_version` | ✅ `model_name`, `threshold`, `label_map`, `dataset_provenance` |
| `threshold` must be a probability in (0, 1) | *(unchanged, read by callers)* | ✅ new |
| Feature names must be known | ✅ | n/a (vocabularies live inside the pipeline) |
| Feature names must be unique and in canonical order | ✅ | n/a |
| `n_features` must equal the number of listed names | ✅ new | n/a |
| A non-canonical feature list must declare `feature_subset` | ✅ new | n/a |
| Pipeline input width must equal the listed names | ✅ (now mandatory) | n/a |
| `feature_schema_version` must equal the code constant | ✅ (now mandatory) | n/a |
| `feature_schema_fingerprint` must match, when present | ✅ new | n/a |
| `feature_type` must equal the code's `FEATURE_SPEC` | n/a | ✅ new |
| Pipeline must expose `predict_proba` and the `features` / `clf` steps | n/a | ✅ new |

`feature_schema_fingerprint()` is a 16-hex SHA-256 over the version and the
ordered canonical names (current value `1de7363ea0a84a41`). It is stamped into
the metadata by every `save_*` function, so a future edit to the schema is caught
even if nobody remembers to bump `FEATURE_SCHEMA_VERSION`; the check is skipped
only for artifacts saved before this change, which the structural checks above
still protect.

The revised model remains loadable because it *declares* its deviation
(`feature_subset: "no_path_no_query"`), which is exactly the contract the new
rule enforces.

### Before / after behaviour

| Situation | Before | After |
|---|---|---|
| Served model, matching metadata | loads | loads (unchanged) |
| Revised model, declared 24-feature subset | loads | loads (unchanged) |
| Domain-holdout model | loads | loads (unchanged) |
| Message model | loads | loads (unchanged) |
| Metadata file deleted | loads with `threshold=0.5`, full schema, **no warning** | `FileNotFoundError`: *"its metadata file … is missing, so the feature schema cannot be verified"* |
| Metadata truncated / invalid JSON | **crash later at `json.loads` with no context** | `RuntimeError`: *"… is not valid JSON (…)"* |
| Metadata missing `n_features` | **accepted** | `RuntimeError`: *"metadata is missing required key(s) ['n_features']"* |
| Feature inserted mid-schema | **accepted — columns silently shifted** | `RuntimeError`: *"the feature list is not the current canonical schema, yet the metadata does not declare a feature_subset"* |
| Stale `feature_schema_version` | accepted if absent | `RuntimeError`: *"model says 1, code says 2"* |
| Stale fingerprint | n/a | `RuntimeError`: *"the model was trained under … but the code now builds …"* |
| Message metadata missing `threshold` | loads with `0.5` | `RuntimeError`: *"metadata is missing required key(s) ['threshold']"* |
| Message trained on a different TF-IDF spec | **accepted** | `RuntimeError`: *"Feature-spec mismatch … the artifact was trained with …"* |

**Fail-safe path.** In the API, `ModelRegistry.load_models()` catches these
errors, logs them and marks the model unloaded, so `/api/analyze/*` returns
**503** with the guard's message rather than serving a mis-mapped model.

---

## 3. Breaking explanation API change (item 4)

An exhaustive search for the removed `value` field (`*.py`, `*.jsx`, `*.md`,
`*.json`) found:

* **no** backend read or write of an explanation field named `value`;
* **no** frontend reference (`ExplanationPanel.jsx` renders `feature_value` and
  `shap_value`); the only `.value` uses in the frontend are ordinary DOM inputs;
* **no** documented example presenting `value` as current — the single JSON
  sample containing it is under a heading that says *"Before … (old payload)"*;
* tests **assert its absence** (`"value" not in item`) in
  `tests/test_explanations.py` and `tests/test_explanation_consistency.py`.

The change is therefore fully propagated; nothing was left dangling.

---

## 4. Metadata / documentation consistency (item 5)

| File | Change |
|---|---|
| `src/fraud_phishing/models/url_model.py` | Module docstring states the load-time verification; guards described above |
| `src/fraud_phishing/models/message_model.py` | Module docstring states the refusal behaviour; `FEATURE_SPEC`, `REQUIRED_METADATA_KEYS`, `_assert_artifact` |
| `scripts/train_message_model.py` | Writes `FEATURE_SPEC` from the single shared constant instead of duplicating the literal string |
| `reports/url_remediation.md` | The "16 of 30 features" diagnosis figure is clarified as the schema that existed when the diagnosis ran (the 3 brand features were added afterwards) |
| `README.md`, `reports/limitations.md`, `reports/error_analysis.md`, `reports/implementation_checklist.md` | Updated for the new guards, the new test count and the new stage |

**No artifact was modified.** `models/*.joblib` and `models/*_metadata.json` are
byte-identical before and after P5 (SHA-256 verified); the legacy metadata files
simply lack the optional fingerprint key, which the loader tolerates by design.

---

## 5. Validation

```
python -m pytest tests/ -q
183 passed, 2 skipped, 1 xfailed in 18.56s      (was 135 passed before P5)

npm run lint   → clean
npm run build  → 20 modules transformed, built
```

48 new tests:

* `tests/test_model_artifact_guards.py` (47 tests) — the four shipped artifacts
  still load; every guard above has a negative test asserting the diagnostic; the
  save/load round trip stamps the schema even when the caller forgets it; the
  loaders are exercised against synthetic artifacts in `tmp_path` (missing
  metadata, missing model, invalid JSON, stale schema).
* `tests/test_url_features.py` (+1 test) — the docstring total and per-group
  counts are cross-checked against `FEATURE_NAMES` and against the group markers
  in the tuple itself.

---

## 6. Compatibility risks

* **A future retrain must keep the metadata writer.** The guards are only as good
  as the metadata: a model saved by hand with `joblib.dump` and no metadata will
  now be *refused* (that is the intent) rather than run at `0.5`.
* **New estimator families.** `_assert_schema` now requires the pipeline to
  report `n_features_in_`. Any future model type that does not (e.g. a custom
  text transformer) must relax that check deliberately.
* **`feature_subset` is a declaration, not a proof.** A model is trusted to
  describe its own subset; the check prevents *silent* drift, not a deliberately
  wrong declaration.
* **Legacy artifacts carry no fingerprint.** Their protection rests on the
  structural checks. Re-saving them (a retrain) closes that gap.
* **Dependency drift is unchanged** (`requirements.txt` pins are still not
  enforced) — tracked as an open item, out of P5 scope.

---

## 7. Remaining limitations

* The guards verify that an artifact *matches the code's schema*; they say nothing
  about whether the model is any good. The served URL model still scores almost
  every URL as phishing (the P3 pathology) and is still not promoted/revised.
* The message model's TF-IDF vocabularies are stored inside the pipeline, so a
  stale message artifact cannot mis-map columns — only the feature *spec* and
  threshold are guarded.
* No artifact *signing* or integrity hash of the `.joblib` bytes is stored; a
  corrupted-but-loadable pipeline is out of scope.
* `feature_schema_fingerprint` hashes the canonical schema, not the fitted
  pipeline; it is a compatibility check, not a provenance record.
