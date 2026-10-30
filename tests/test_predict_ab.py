"""Tests for the shadow-model A/B comparison endpoint.

These exercise the endpoint contract without loading real YOLO weights: the
detector objects are replaced with fakes, so the test runs in CI without a GPU
or a multi-megabyte checkpoint.
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Import the app the same way tests/test_api.py does. Importing it as
# "app.main" instead would create a second module object, and the Prometheus
# collectors would then be registered twice, raising DuplicateTimeseries.
from src.app.main import app as fastapi_app


class _FakeBox:
    """Mimics a YOLO box: xyxy is a tensor-like object with .tolist()."""

    def __init__(self, cls, conf, xyxy):
        self.cls = cls
        self.conf = conf
        self.xyxy = np.asarray(xyxy, dtype=float)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeDetector:
    """Minimal stand-in for ultralytics.YOLO."""

    def __init__(self, names, boxes):
        self.names = names
        self._boxes = boxes

    def __call__(self, image):
        return [_FakeResult(self._boxes)]


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client(monkeypatch):
    import src.app.main as main

    # API_KEY is set per-test, not at import time. Setting it at module level
    # would leak into every other test module in the same pytest process and
    # silently turn auth on for tests that expect it to be off.
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setattr(main, "model", _FakeDetector({0: "Corrosion"}, [_FakeBox(0, 0.9, [[0, 0, 10, 10]])]))
    monkeypatch.setattr(main, "shadow_model", _FakeDetector({0: "Corrosion"}, [_FakeBox(0, 0.8, [[0, 0, 10, 10]])]))
    monkeypatch.setattr(main, "MODEL_PATH", "champion.pt")
    monkeypatch.setattr(main, "SHADOW_MODEL_PATH", "challenger.pt")
    return TestClient(fastapi_app)


def test_predict_ab_reports_agreement(client):
    resp = client.post(
        "/predict-ab",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["agreement"] is True
    assert body["champion"]["model_path"] == "champion.pt"
    assert body["challenger"]["model_path"] == "challenger.pt"
    assert body["champion_only_classes"] == []
    assert body["challenger_only_classes"] == []


def test_predict_ab_detects_disagreement(client, monkeypatch):
    import src.app.main as main

    monkeypatch.setattr(main, "shadow_model", _FakeDetector({0: "Rust"}, [_FakeBox(0, 0.7, [[0, 0, 5, 5]])]))
    resp = client.post(
        "/predict-ab",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["agreement"] is False
    assert body["champion_only_classes"] == ["Corrosion"]
    assert body["challenger_only_classes"] == ["Rust"]


def test_predict_ab_without_shadow_model_returns_503(client, monkeypatch):
    import src.app.main as main

    monkeypatch.setattr(main, "shadow_model", None)
    resp = client.post(
        "/predict-ab",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 503
    assert "SHADOW_MODEL_PATH" in resp.json()["detail"]
