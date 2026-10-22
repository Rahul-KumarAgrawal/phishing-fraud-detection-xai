"""
test_dataset_provenance.py
==========================
Guards the real/synthetic distinction for the SMS dataset.

The project previously trained on a 65-message synthetic placeholder and reported
metrics from it while the documentation described the real Mendeley dataset: a
model that looked, logged and loaded like a real one. The provenance detector and
the trainer guard exist so that cannot recur silently.
"""

import pathlib

import pandas as pd
import pytest

from fraud_phishing.data import prepare_messages as pm


def _df(labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"LABEL": labels, "TEXT": [f"message {i}" for i in range(len(labels))]})


def _no_marker(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pm, "SYNTHETIC_MARKER", tmp_path / "SMS_dataset.SYNTHETIC")


# ── Provenance detection ─────────────────────────────────────────────────────

def test_marker_file_marks_data_synthetic(tmp_path, monkeypatch):
    marker = tmp_path / "SMS_dataset.SYNTHETIC"
    monkeypatch.setattr(pm, "SYNTHETIC_MARKER", marker)
    marker.write_text("synthetic", encoding="utf-8")

    provenance, detail = pm.detect_dataset_provenance(_df(["ham"] * 3), "LABEL")
    assert provenance == "synthetic"
    assert "marker" in detail


def test_synthetic_signature_detected_without_marker(tmp_path, monkeypatch):
    _no_marker(tmp_path, monkeypatch)
    labels = (
        ["ham"] * pm.SYNTHETIC_LABEL_COUNTS["ham"]
        + ["smishing"] * pm.SYNTHETIC_LABEL_COUNTS["smishing"]
    )

    provenance, detail = pm.detect_dataset_provenance(_df(labels), "LABEL")
    assert provenance == "synthetic"
    assert "create_synthetic_sms" in detail


def test_documented_mendeley_composition_is_real(tmp_path, monkeypatch):
    _no_marker(tmp_path, monkeypatch)
    counts = pm.MENDELEY_EXPECTED_LABEL_COUNTS
    labels = (
        ["ham"] * counts["ham"]
        + ["smishing"] * counts["smishing"]
        + ["spam"] * counts["spam"]
    )

    provenance, detail = pm.detect_dataset_provenance(_df(labels), "LABEL")
    assert provenance == "real"
    assert "Mendeley" in detail


def test_unrecognised_composition_is_unknown_not_real(tmp_path, monkeypatch):
    _no_marker(tmp_path, monkeypatch)
    provenance, _ = pm.detect_dataset_provenance(_df(["ham"] * 50 + ["smishing"] * 50), "LABEL")
    assert provenance == "unknown"


def test_provenance_detection_normalises_label_casing(tmp_path, monkeypatch):
    """
    The published file mixes 'Smishing'/'smishing' and 'spam'/'Spam'. Detection
    must not depend on the casing, or real data would be misclassified.
    """
    _no_marker(tmp_path, monkeypatch)
    counts = pm.MENDELEY_EXPECTED_LABEL_COUNTS
    labels = (
        ["ham"] * counts["ham"]
        + ["Smishing"] * 600
        + ["smishing"] * (counts["smishing"] - 600)
        + ["spam"] * 400
        + ["Spam"] * (counts["spam"] - 400)
    )

    provenance, _ = pm.detect_dataset_provenance(_df(labels), "LABEL")
    assert provenance == "real"


# ── Label value guard ────────────────────────────────────────────────────────

def test_known_labels_are_accepted():
    pm.validate_label_values(pd.Series(["ham", "smishing", "spam"]))


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="Unexpected label values"):
        pm.validate_label_values(pd.Series(["ham", "phishing"]))
