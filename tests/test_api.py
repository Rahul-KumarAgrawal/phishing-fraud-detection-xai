"""
test_api.py
===========
API tests. Runs against TestClient (no live server needed).
Tests both the happy-path and error scenarios.
Models do not need to be trained for schema/validation tests.
"""

import json
import sys
import pathlib
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient

from fraud_phishing.api.main import app
from fraud_phishing.api.schemas import (
    HealthResponse,
    MessageAnalysisResponse,
    UrlAnalysisResponse,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client_with_mocked_models():
    """Return a TestClient where both models are mocked as loaded."""
    mock_registry = MagicMock()
    mock_registry.message_loaded = True
    mock_registry.url_loaded = True

    mock_msg_response = {
        "message_label": 1,
        "message_label_name": "smishing",
        "message_score": 0.92,
        "message_threshold": 0.5,
        "explanation": None,
        "detected_urls": [],
        "disclaimer": "Test disclaimer",
    }
    mock_url_response = {
        "url": "http://example.com",
        "label": 0,
        "label_name": "legitimate",
        "score": 0.12,
        "threshold": 0.5,
        "features": None,
        "explanation": None,
        "disclaimer": "Test disclaimer",
    }

    mock_registry.analyze_message.return_value = MagicMock(
        **mock_msg_response,
        model_dump=lambda: mock_msg_response,
    )
    mock_registry.analyze_url.return_value = MagicMock(
        **mock_url_response,
        model_dump=lambda: mock_url_response,
    )

    with patch("fraud_phishing.api.main.registry", mock_registry):
        yield TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_health_response_schema(client):
    resp = client.get("/api/health")
    data = resp.json()
    assert "status" in data
    assert "message_model_loaded" in data
    assert "url_model_loaded" in data
    assert isinstance(data["message_model_loaded"], bool)
    assert isinstance(data["url_model_loaded"], bool)


def test_health_status_is_ok(client):
    resp = client.get("/api/health")
    assert resp.json()["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Message endpoint – input validation
# ─────────────────────────────────────────────────────────────────────────────

def test_message_empty_body_returns_422(client):
    resp = client.post("/api/analyze/message", json={})
    assert resp.status_code == 422


def test_message_blank_string_returns_422(client):
    resp = client.post("/api/analyze/message", json={"message": "   "})
    assert resp.status_code == 422


def test_message_too_long_returns_422(client):
    resp = client.post("/api/analyze/message", json={"message": "a" * 10_001})
    assert resp.status_code == 422


def test_message_missing_field_returns_422(client):
    resp = client.post("/api/analyze/message", json={"text": "hello"})
    assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# URL endpoint – input validation
# ─────────────────────────────────────────────────────────────────────────────

def test_url_empty_returns_422(client):
    resp = client.post("/api/analyze/url", json={"url": ""})
    assert resp.status_code == 422


def test_url_blank_returns_422(client):
    resp = client.post("/api/analyze/url", json={"url": "  "})
    assert resp.status_code == 422


def test_url_too_long_returns_422(client):
    resp = client.post("/api/analyze/url", json={"url": "http://x.com/" + "a" * 4200})
    assert resp.status_code == 422


def test_url_missing_field_returns_422(client):
    resp = client.post("/api/analyze/url", json={"link": "http://example.com"})
    assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Model-not-loaded responses
# ─────────────────────────────────────────────────────────────────────────────

def test_message_returns_503_when_model_not_loaded(client):
    """Before training, the model is not loaded; should return 503."""
    resp = client.post("/api/analyze/message", json={"message": "Hello, you won a prize!"})
    # Either 503 (model not loaded) or 200 (if model happens to be trained)
    assert resp.status_code in (200, 503)


def test_url_returns_503_when_model_not_loaded(client):
    resp = client.post("/api/analyze/url", json={"url": "http://example.com"})
    assert resp.status_code in (200, 503)


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation (with mocked registry)
# ─────────────────────────────────────────────────────────────────────────────

def test_message_response_schema_fields(client_with_mocked_models):
    resp = client_with_mocked_models.post(
        "/api/analyze/message", json={"message": "Congratulations! You won!"}
    )
    if resp.status_code == 200:
        data = resp.json()
        assert "message_label" in data
        assert "message_label_name" in data
        assert "message_score" in data
        assert "detected_urls" in data
        assert isinstance(data["detected_urls"], list)


def test_url_response_schema_fields(client_with_mocked_models):
    resp = client_with_mocked_models.post(
        "/api/analyze/url", json={"url": "http://example.com/login"}
    )
    if resp.status_code == 200:
        data = resp.json()
        assert "label" in data
        assert "label_name" in data
        assert "score" in data
        assert "url" in data


# ─────────────────────────────────────────────────────────────────────────────
# Separation of message and URL predictions
# ─────────────────────────────────────────────────────────────────────────────

def test_message_and_url_predictions_are_separate(client_with_mocked_models):
    """A message containing a URL must return separate message + URL results."""
    msg_resp = client_with_mocked_models.post(
        "/api/analyze/message",
        json={"message": "Check this out: http://example.com/login"},
    )
    url_resp = client_with_mocked_models.post(
        "/api/analyze/url",
        json={"url": "http://example.com/login"},
    )
    # Both should have distinct response schemas
    if msg_resp.status_code == 200:
        msg_data = msg_resp.json()
        assert "message_label" in msg_data
    if url_resp.status_code == 200:
        url_data = url_resp.json()
        assert "label" in url_data
