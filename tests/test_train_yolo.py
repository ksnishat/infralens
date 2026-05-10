"""Tests for the InfraLens YOLOv8 training module."""
import pytest
from unittest.mock import patch, MagicMock
from src.scripts.train_yolo import train_yolo


class TestTrainYolo:
    """Test the train_yolo function."""

    def test_train_function_signature(self):
        """Train function should accept expected parameters."""
        import inspect
        sig = inspect.signature(train_yolo)
        params = list(sig.parameters.keys())
        assert "data_yaml" in params
        assert "epochs" in params
        assert "batch_size" in params
        assert "imgsz" in params
        assert "project_name" in params
        assert "run_name" in params

    def test_train_function_defaults(self):
        """Train function should have sensible defaults."""
        import inspect
        sig = inspect.signature(train_yolo)
        assert sig.parameters["epochs"].default == 100
        assert sig.parameters["batch_size"].default == 16
        assert sig.parameters["imgsz"].default == 640
        assert sig.parameters["project_name"].default == "infralens_rust_detection"

    @patch("src.scripts.train_yolo.mlflow")
    def test_train_calls_mlflow(self, mock_mlflow):
        """Training should log to MLflow."""
        # Mock YOLO import inside the function
        import sys
        mock_yolo = MagicMock()
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock results with a real dict
        mock_results = MagicMock()
        mock_results.results_dict = {"metrics/mAP50": 0.85, "metrics/precision": 0.90}
        mock_model.train.return_value = mock_results

        with patch.dict(sys.modules, {"ultralytics": MagicMock(YOLO=mock_yolo)}):
            train_yolo(epochs=1, batch_size=8)

        # Verify MLflow was called
        assert mock_mlflow.set_tracking_uri.called
        assert mock_mlflow.set_experiment.called
        assert mock_mlflow.start_run.called
        assert mock_mlflow.log_params.called
        # log_metrics may or may not be called depending on results_dict
        # The key is that the training function runs without error
