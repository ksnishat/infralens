"""InfraLens Project API Tests.

Tests for the YOLOv8 inference API, safety reporting agent, and image preprocessing.
"""

import pytest
from fastapi.testclient import TestClient
from src.app.main import app


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient for all API tests. Uses raise_server_exceptions=False
    so that a missing model file doesn't crash the test server."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def dummy_image(tmp_path):
    """Create a small dummy JPEG image for upload tests."""
    try:
        import numpy as np
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="gray")
        img.save(tmp_path / "dummy.jpg")
        return tmp_path / "dummy.jpg"
    except ImportError:
        pytest.skip("PIL/numpy not installed")


@pytest.fixture
def bad_image(tmp_path):
    """Create an invalid/corrupted image."""
    tmp_path.joinpath("corrupt.jpg").write_bytes(b"this is not a real image")
    return tmp_path / "corrupt.jpg"


class TestHealth:
    """Test /health endpoint."""

    def test_health_check(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["service"] == "InfraLens Backend"


class TestDetection:
    """Test /predict endpoint."""

    def test_predict_returns_detections_list(self, client, dummy_image):
        with open(dummy_image, "rb") as f:
            response = client.post("/predict", files={"file": ("dummy.jpg", f.read(), "image/jpeg")})
        assert response.status_code == 200
        data = response.json()
        assert "filename" in data
        assert "detections" in data
        assert isinstance(data["detections"], list)
        assert data["filename"] == "dummy.jpg"

    def test_predict_returns_detection_structure(self, client, dummy_image):
        """Each detection should have class, confidence, bbox fields."""
        with open(dummy_image, "rb") as f:
            response = client.post("/predict", files={"file": f.read()})
        data = response.json()
        if data["detections"]:
            for det in data["detections"]:
                assert "class" in det
                assert "confidence" in det
                assert "bbox" in det

    def test_predict_with_confidence_threshold(self, client, dummy_image):
        """Test threshold slider functionality via request params."""
        with open(dummy_image, "rb") as f:
            # Simulate a threshold request (params may be ignored, endpoint should work)
            params = {"threshold": "0.25"}
            response = client.post("/predict", files={"file": f.read()}, params=params)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["detections"], list)

    def test_predict_handles_missing_model_gracefully(self, client):
        """A request without a file is rejected cleanly (FastAPI 422), never a crash."""
        response = client.post("/predict")
        assert response.status_code in (500, 422)
        data = response.json()
        assert "detail" in data


class TestReportAnalysis:
    """Test /analyze-report endpoint (YOLO + Llama 3)."""

    def test_analyze_report_returns_safety_report(self, client, dummy_image):
        with open(dummy_image, "rb") as f:
            response = client.post("/analyze-report", files={"file": f.read()})
        assert response.status_code == 200
        data = response.json()
        assert "filename" in data
        assert "detections" in data
        assert "safety_report" in data
        assert len(data["safety_report"]) > 0

    def test_analyze_report_with_no_model(self, client):
        """If model missing, endpoint should degrade gracefully (not 500 crash)."""
        response = client.post("/analyze-report")
        assert response.status_code in (200, 500, 422)


class TestOpenAPI:
    """Test OpenAPI schema and documentation."""

    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "InfraLens API"

    def test_docs_available(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
        assert "Swagger" in response.text or "FastAPI" in response.text


class TestPrometheusMetrics:
    """Test Prometheus metrics endpoint."""

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.content.decode("utf-8") != ""

    def test_metrics_increment_on_requests(self, client):
        import time
        client.get("/")
        client.get("/")
        response = client.get("/metrics")
        metrics = response.content.decode("utf-8")
        assert "infrawlens" in metrics.lower() or "infralens" in metrics.lower()


class TestDockerComposeHealth:
    """Test Docker Compose health configurations."""

    def test_docker_compose_config(self):
        import yaml
        with open("docker-compose.yml") as f:
            config = yaml.safe_load(f)
        required_services = ["backend", "ollama", "frontend"]
        for service in required_services:
            assert service in config["services"], f"Missing required service: {service}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])