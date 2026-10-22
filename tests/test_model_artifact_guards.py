"""
test_model_artifact_guards.py
=============================
Regression tests for the P5 model-artifact / feature-schema compatibility guards.

Two silent failure modes motivated them:

1. A URL artifact whose metadata was missing (or stale) still loaded. The caller
   then fell back to the default ``0.5`` threshold and the full 33-feature
   schema, so a model trained on a 24-feature subset could be fed the wrong
   columns and decide at the wrong cut-off without a single warning.
2. Editing ``url_features.FEATURE_NAMES`` - for example inserting a new feature
   in the middle - shifts every later column while leaving the *relative* order
   of the recorded names intact, so the old checks (unknown names + order) all
   passed and the model kept returning confident, wrong answers.

The guards are exercised against the real artifacts when they are present
(skipped on a clean checkout, like the rest of the model-dependent suite) and
against synthetic artifacts built in ``tmp_path``.
"""

import json
import pathlib
import sys

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fraud_phishing.features.url_features import FEATURE_NAMES  # noqa: E402
from fraud_phishing.models import message_model, url_model  # noqa: E402

ARTIFACT_PATH = pathlib.Path("url_model.joblib")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fitted_url_pipeline(n_features: int = len(FEATURE_NAMES)) -> Pipeline:
    """A tiny fitted pipeline whose input width is ``n_features``."""
    rng = np.random.RandomState(0)
    X = rng.rand(60, n_features)
    y = (X[:, 0] > 0.5).astype(int)
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]
    )
    pipe.fit(X, y)
    return pipe


def _url_metadata(**overrides) -> dict:
    """Metadata that matches the current extractor exactly, plus overrides."""
    names = list(FEATURE_NAMES)
    metadata = {
        "feature_names": names,
        "n_features": len(names),
        "feature_schema_version": url_model.FEATURE_SCHEMA_VERSION,
        "feature_schema_fingerprint": url_model.feature_schema_fingerprint(),
    }
    metadata.update(overrides)
    return metadata


def _fitted_message_pipeline() -> Pipeline:
    texts = [
        "hello there friend",
        "urgent verify your account now",
        "see you soon",
        "click here to win a prize",
    ]
    pipe = Pipeline(
        [("features", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=500))]
    )
    pipe.fit(texts, [0, 1, 0, 1])
    return pipe


def _message_metadata(**overrides) -> dict:
    metadata = {
        "model_name": "LinearSVC",
        "threshold": 0.618,
        "label_map": {"0": "ham", "1": "smishing"},
        "dataset_provenance": "real",
        "feature_type": message_model.FEATURE_SPEC,
    }
    metadata.update(overrides)
    return metadata


# ── The artifacts that ship with the repo still load ─────────────────────────

@pytest.mark.skipif(not url_model.URL_MODEL_PATH.exists(), reason="no served URL model")
def test_served_url_artifact_loads_with_declared_schema():
    pipeline, metadata = url_model.load_model()
    names = list(metadata["feature_names"])
    assert names
    assert all(name in FEATURE_NAMES for name in names)
    assert [FEATURE_NAMES.index(name) for name in names] == sorted(
        FEATURE_NAMES.index(name) for name in names
    )
    if names != list(FEATURE_NAMES):
        assert metadata["feature_subset"], "subset model must declare its subset"
    assert metadata["n_features"] == len(names)
    assert metadata["feature_schema_version"] == url_model.FEATURE_SCHEMA_VERSION
    assert 0.0 < float(metadata["threshold"]) < 1.0
    assert int(pipeline.n_features_in_) == len(names)


@pytest.mark.skipif(
    not url_model.URL_REVISED_MODEL_PATH.exists(), reason="no revised URL model"
)
def test_revised_url_artifact_loads_with_declared_subset():
    pipeline, metadata = url_model.load_revised_model()
    names = list(metadata["feature_names"])
    assert names, "revised metadata has no feature_names"
    assert names != list(FEATURE_NAMES), "revised model is expected to use a subset"
    assert metadata["feature_subset"], "a subset model must declare feature_subset"
    assert metadata["n_features"] == len(names)
    assert int(pipeline.n_features_in_) == len(names)


@pytest.mark.skipif(
    not url_model.URL_DOMAIN_MODEL_PATH.exists(), reason="no domain-holdout model"
)
def test_domain_holdout_artifact_loads():
    pipeline, metadata = url_model.load_domain_holdout_model()
    assert metadata["n_features"] == len(metadata["feature_names"])
    assert int(pipeline.n_features_in_) == len(metadata["feature_names"])


@pytest.mark.skipif(
    not message_model.MESSAGE_MODEL_PATH.exists(), reason="no message model"
)
def test_message_artifact_loads_with_canonical_feature_spec():
    pipeline, metadata = message_model.load_model()
    assert metadata["feature_type"] == message_model.FEATURE_SPEC
    assert 0.0 < float(metadata["threshold"]) < 1.0
    assert hasattr(pipeline, "predict_proba")


@pytest.mark.skipif(not url_model.URL_MODEL_PATH.exists(), reason="no served URL model")
def test_artifacts_carry_a_fingerprint_matching_the_code():
    """The fingerprint is optional for pre-P5 artifacts but must never disagree."""
    _, metadata = url_model.load_model()
    fingerprint = metadata.get("feature_schema_fingerprint")
    if fingerprint is not None:
        assert fingerprint == url_model.feature_schema_fingerprint()


# ── Fingerprint properties ───────────────────────────────────────────────────

def test_schema_fingerprint_is_stable_and_order_sensitive():
    fingerprint = url_model.feature_schema_fingerprint()
    assert fingerprint == url_model.feature_schema_fingerprint()
    assert len(fingerprint) == 16
    assert fingerprint != url_model.feature_schema_fingerprint(
        tuple(reversed(FEATURE_NAMES))
    )


def test_schema_fingerprint_changes_when_a_feature_is_added():
    extended = tuple(FEATURE_NAMES) + ("brand_new_feature",)
    assert url_model.feature_schema_fingerprint() != url_model.feature_schema_fingerprint(
        extended
    )


# ── URL guard: compatible metadata passes ────────────────────────────────────

def test_matching_metadata_passes():
    url_model._assert_schema(
        _fitted_url_pipeline(), _url_metadata(), ARTIFACT_PATH
    )


def test_legacy_metadata_without_fingerprint_is_still_accepted():
    """Artifacts saved before P5 have no fingerprint; the structural checks apply."""
    metadata = _url_metadata()
    metadata.pop("feature_schema_fingerprint")
    url_model._assert_schema(_fitted_url_pipeline(), metadata, ARTIFACT_PATH)


def test_declared_feature_subset_plus_matching_pipeline_passes():
    names = [n for n in FEATURE_NAMES if n not in {"path_length", "has_query"}]
    url_model._assert_schema(
        _fitted_url_pipeline(len(names)),
        _url_metadata(
            feature_names=names, n_features=len(names), feature_subset="tiny"
        ),
        ARTIFACT_PATH,
    )


# ── URL guard: incompatible metadata fails with a clear diagnostic ───────────

def test_missing_metadata_is_rejected():
    with pytest.raises(RuntimeError, match="metadata is missing or empty"):
        url_model._assert_schema(_fitted_url_pipeline(), {}, ARTIFACT_PATH)


@pytest.mark.parametrize("key", ["feature_names", "n_features", "feature_schema_version"])
def test_missing_required_metadata_key_is_rejected(key):
    metadata = _url_metadata()
    metadata.pop(key)
    with pytest.raises(RuntimeError, match=key):
        url_model._assert_schema(_fitted_url_pipeline(), metadata, ARTIFACT_PATH)


def test_unknown_feature_name_is_rejected():
    names = ["not_a_real_feature"] + list(FEATURE_NAMES)[1:]
    with pytest.raises(RuntimeError, match="unknown feature"):
        url_model._assert_schema(
            _fitted_url_pipeline(), _url_metadata(feature_names=names), ARTIFACT_PATH
        )


def test_reordered_features_are_rejected():
    names = list(FEATURE_NAMES)
    names[10], names[11] = names[11], names[10]
    with pytest.raises(RuntimeError, match="feature order"):
        url_model._assert_schema(
            _fitted_url_pipeline(), _url_metadata(feature_names=names), ARTIFACT_PATH
        )


def test_duplicate_feature_names_are_rejected():
    names = list(FEATURE_NAMES)
    names[1] = names[0]
    with pytest.raises(RuntimeError, match="duplicate feature names"):
        url_model._assert_schema(
            _fitted_url_pipeline(), _url_metadata(feature_names=names), ARTIFACT_PATH
        )


def test_undeclared_feature_subset_is_rejected():
    names = list(FEATURE_NAMES)[:-3]
    with pytest.raises(RuntimeError, match="feature_subset"):
        url_model._assert_schema(
            _fitted_url_pipeline(len(names)),
            _url_metadata(feature_names=names, n_features=len(names)),
            ARTIFACT_PATH,
        )


def test_feature_inserted_into_the_canonical_schema_is_rejected(monkeypatch):
    """
    The specific drift the old checks missed: a feature added in the middle of
    FEATURE_NAMES leaves the old names in canonical relative order, so only the
    "not the canonical list" rule catches it.
    """
    monkeypatch.setattr(
        url_model, "FEATURE_NAMES", tuple(FEATURE_NAMES) + ("brand_new_feature",)
    )
    with pytest.raises(RuntimeError, match="feature_subset"):
        url_model._assert_schema(
            _fitted_url_pipeline(), _url_metadata(), ARTIFACT_PATH
        )


def test_n_features_disagreeing_with_the_name_list_is_rejected():
    with pytest.raises(RuntimeError, match="n_features=7"):
        url_model._assert_schema(
            _fitted_url_pipeline(), _url_metadata(n_features=7), ARTIFACT_PATH
        )


def test_non_integer_n_features_is_rejected():
    with pytest.raises(RuntimeError, match="not an integer"):
        url_model._assert_schema(
            _fitted_url_pipeline(), _url_metadata(n_features="thirty-three"), ARTIFACT_PATH
        )


def test_pipeline_width_not_matching_metadata_is_rejected():
    with pytest.raises(RuntimeError, match="takes 5 features"):
        url_model._assert_schema(
            _fitted_url_pipeline(5), _url_metadata(), ARTIFACT_PATH
        )


def test_pipeline_without_input_width_is_rejected():
    with pytest.raises(RuntimeError, match="n_features_in_"):
        url_model._assert_schema(
            Pipeline([("scaler", StandardScaler())]), _url_metadata(), ARTIFACT_PATH
        )


def test_stale_schema_version_is_rejected():
    with pytest.raises(RuntimeError, match="version mismatch"):
        url_model._assert_schema(
            _fitted_url_pipeline(),
            _url_metadata(feature_schema_version=url_model.FEATURE_SCHEMA_VERSION - 1),
            ARTIFACT_PATH,
        )


def test_stale_fingerprint_is_rejected():
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        url_model._assert_schema(
            _fitted_url_pipeline(),
            _url_metadata(feature_schema_fingerprint="0123456789abcdef"),
            ARTIFACT_PATH,
        )


# ── URL loader: fails safely on disk, round-trips on save ────────────────────

def _point_url_paths_at(monkeypatch, directory: pathlib.Path) -> None:
    monkeypatch.setattr(url_model, "URL_MODEL_PATH", directory / "url_model.joblib")
    monkeypatch.setattr(
        url_model, "URL_METADATA_PATH", directory / "url_model_metadata.json"
    )


def test_load_model_round_trips_a_saved_artifact(tmp_path, monkeypatch):
    _point_url_paths_at(monkeypatch, tmp_path)
    url_model.save_model(
        _fitted_url_pipeline(), {"threshold": 0.4, "feature_names": list(FEATURE_NAMES)}
    )
    pipeline, metadata = url_model.load_model()
    assert metadata["threshold"] == 0.4
    assert metadata["n_features"] == len(FEATURE_NAMES)
    assert metadata["feature_schema_fingerprint"] == url_model.feature_schema_fingerprint()


def test_save_model_stamps_the_schema_even_if_the_caller_forgets(tmp_path, monkeypatch):
    _point_url_paths_at(monkeypatch, tmp_path)
    url_model.save_model(_fitted_url_pipeline(), {"threshold": 0.4})
    metadata = json.loads(
        (tmp_path / "url_model_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["feature_names"] == list(FEATURE_NAMES)
    assert metadata["feature_schema_version"] == url_model.FEATURE_SCHEMA_VERSION
    assert metadata["feature_schema_fingerprint"]


def test_load_model_refuses_a_missing_metadata_file(tmp_path, monkeypatch):
    _point_url_paths_at(monkeypatch, tmp_path)
    joblib.dump(_fitted_url_pipeline(), url_model.URL_MODEL_PATH)
    with pytest.raises(FileNotFoundError, match="metadata file"):
        url_model.load_model()


def test_load_model_refuses_missing_model_file(tmp_path, monkeypatch):
    _point_url_paths_at(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="Model not found"):
        url_model.load_model()


def test_load_model_refuses_invalid_json_metadata(tmp_path, monkeypatch):
    _point_url_paths_at(monkeypatch, tmp_path)
    joblib.dump(_fitted_url_pipeline(), url_model.URL_MODEL_PATH)
    url_model.URL_METADATA_PATH.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        url_model.load_model()


def test_load_model_refuses_a_stale_artifact(tmp_path, monkeypatch):
    """The pre-P5 failure mode: a stale artifact must not load silently."""
    _point_url_paths_at(monkeypatch, tmp_path)
    joblib.dump(_fitted_url_pipeline(5), url_model.URL_MODEL_PATH)
    url_model.URL_METADATA_PATH.write_text(
        json.dumps(_url_metadata()), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Feature-schema mismatch"):
        url_model.load_model()


# ── Message guard ────────────────────────────────────────────────────────────

def test_message_matching_metadata_passes():
    message_model._assert_artifact(
        _fitted_message_pipeline(),
        _message_metadata(),
        message_model.MESSAGE_MODEL_PATH,
    )


def test_message_missing_metadata_is_rejected():
    with pytest.raises(RuntimeError, match="metadata is missing or empty"):
        message_model._assert_artifact(
            _fitted_message_pipeline(), {}, message_model.MESSAGE_MODEL_PATH
        )


@pytest.mark.parametrize(
    "key", ["model_name", "threshold", "label_map", "dataset_provenance"]
)
def test_message_missing_required_metadata_key_is_rejected(key):
    metadata = _message_metadata()
    metadata.pop(key)
    with pytest.raises(RuntimeError, match=key):
        message_model._assert_artifact(
            _fitted_message_pipeline(), metadata, message_model.MESSAGE_MODEL_PATH
        )


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.2, "high", None, True])
def test_message_invalid_threshold_is_rejected(threshold):
    with pytest.raises(RuntimeError, match="threshold"):
        message_model._assert_artifact(
            _fitted_message_pipeline(),
            _message_metadata(threshold=threshold),
            message_model.MESSAGE_MODEL_PATH,
        )


def test_message_feature_spec_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="Feature-spec mismatch"):
        message_model._assert_artifact(
            _fitted_message_pipeline(),
            _message_metadata(feature_type="TF-IDF word only"),
            message_model.MESSAGE_MODEL_PATH,
        )


def test_message_pipeline_without_expected_steps_is_rejected():
    with pytest.raises(RuntimeError, match="step"):
        message_model._assert_artifact(
            TfidfVectorizer(), _message_metadata(), message_model.MESSAGE_MODEL_PATH
        )


def test_message_load_refuses_a_missing_metadata_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        message_model, "MESSAGE_MODEL_PATH", tmp_path / "message_model.joblib"
    )
    monkeypatch.setattr(
        message_model, "MESSAGE_METADATA_PATH", tmp_path / "message_model_metadata.json"
    )
    joblib.dump(_fitted_message_pipeline(), message_model.MESSAGE_MODEL_PATH)
    with pytest.raises(FileNotFoundError, match="metadata file"):
        message_model.load_model()


def test_message_load_round_trips_a_saved_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(
        message_model, "MESSAGE_MODEL_PATH", tmp_path / "message_model.joblib"
    )
    monkeypatch.setattr(
        message_model, "MESSAGE_METADATA_PATH", tmp_path / "message_model_metadata.json"
    )
    metadata = _message_metadata()
    metadata.pop("feature_type")  # save_model fills the canonical spec in
    message_model.save_model(_fitted_message_pipeline(), metadata)
    pipeline, loaded = message_model.load_model()
    assert loaded["feature_type"] == message_model.FEATURE_SPEC
    assert loaded["threshold"] == 0.618
    assert hasattr(pipeline, "predict_proba")
