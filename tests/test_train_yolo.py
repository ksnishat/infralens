"""Tests for the InfraLens YOLOv8 training module."""
from unittest.mock import MagicMock, patch

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
    def test_train_calls_mlflow(self, mock_mlflow, tmp_path, monkeypatch):
        """Training should log to MLflow."""
        # Mock YOLO import inside the function
        import sys
        mock_yolo = MagicMock()
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # A real checkpoint on disk: the function now resolves the trainer's
        # save_dir and raises if no best.pt exists, because silently skipping
        # the log/publish step is exactly the bug this guards against.
        run_name = "mlflow_test"
        save_dir = tmp_path / "runs" / "detect" / "runs" / "train" / run_name
        (save_dir / "weights").mkdir(parents=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"fake")

        mock_results = MagicMock()
        mock_results.results_dict = {"metrics/mAP50": 0.85, "metrics/precision": 0.90}
        mock_results.save_dir = str(save_dir)
        mock_model.train.return_value = mock_results

        monkeypatch.setenv("MODEL_PATH", str(tmp_path / "serving" / "best.pt"))

        with patch.dict(sys.modules, {"ultralytics": MagicMock(YOLO=mock_yolo)}):
            train_yolo(epochs=1, batch_size=8, run_name=run_name)

        # Verify MLflow was called
        assert mock_mlflow.set_tracking_uri.called
        assert mock_mlflow.set_experiment.called
        assert mock_mlflow.start_run.called
        assert mock_mlflow.log_params.called
        # The checkpoint must actually be logged, not silently skipped.
        # log_artifact is used rather than mlflow.pytorch.log_model because the
        # in-memory YOLO object holds an unpicklable DataLoader iterator.
        assert mock_mlflow.log_artifact.called


class TestServingPublish:
    """Training must publish weights to the path the API actually loads.

    Without this the DAG trained a model, logged it to MLflow, and reported
    success — while the API kept loading the old weights from MODEL_PATH. Every
    component was green and the retrain had no effect.
    """

    @patch("src.scripts.train_yolo.mlflow")
    def test_publishes_weights_to_model_path(self, mock_mlflow, tmp_path, monkeypatch):
        import sys
        from pathlib import Path

        # A fake training run that produces a best.pt on disk.
        run_name = "publish_test"
        weights = Path("runs/train") / run_name / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_bytes(b"fake-weights")

        serving = tmp_path / "serving" / "rust_v8s_best.pt"
        monkeypatch.setenv("MODEL_PATH", str(serving))

        mock_yolo = MagicMock()
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_results = MagicMock()
        mock_results.results_dict = {"metrics/mAP50": 0.4}
        mock_model.train.return_value = mock_results

        try:
            with patch.dict(sys.modules, {"ultralytics": MagicMock(YOLO=mock_yolo)}):
                train_yolo(epochs=1, batch_size=8, run_name=run_name)

            assert serving.exists(), (
                "training must copy best.pt to MODEL_PATH, otherwise the reload "
                "task re-reads the old weights and the retrain has no effect"
            )
            assert serving.read_bytes() == b"fake-weights"
        finally:
            import shutil

            shutil.rmtree(Path("runs/train") / run_name, ignore_errors=True)


class TestBestWeightsResolution:
    """The checkpoint path must come from the trainer, not from a guess.

    Ultralytics resolves `project` relative to its own settings directory and
    prepends `runs/detect/`, so passing project="runs/train" writes to
    runs/detect/runs/train/<name>/weights/best.pt. Hardcoding the input path made
    the existence check fail silently: the model was never logged to MLflow and
    never published to the serving path, while the task still reported success.
    """

    def test_prefers_the_trainers_save_dir(self, tmp_path):
        from src.scripts.train_yolo import _resolve_best_weights

        save_dir = tmp_path / "runs" / "detect" / "runs" / "train" / "exp"
        (save_dir / "weights").mkdir(parents=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"x")

        results = MagicMock()
        results.save_dir = str(save_dir)

        assert _resolve_best_weights(results, "exp") == save_dir / "weights" / "best.pt"

    def test_returns_none_when_nothing_exists(self, tmp_path, monkeypatch):
        from src.scripts.train_yolo import _resolve_best_weights

        monkeypatch.chdir(tmp_path)
        results = MagicMock()
        results.save_dir = str(tmp_path / "nowhere")

        assert _resolve_best_weights(results, "missing_run") is None

    def test_falls_back_to_a_glob_when_save_dir_is_absent(self, tmp_path, monkeypatch):
        from src.scripts.train_yolo import _resolve_best_weights

        monkeypatch.chdir(tmp_path)
        nested = tmp_path / "runs" / "detect" / "runs" / "train" / "exp" / "weights"
        nested.mkdir(parents=True)
        (nested / "best.pt").write_bytes(b"x")

        results = MagicMock(spec=[])  # no save_dir attribute at all

        resolved = _resolve_best_weights(results, "exp")
        assert resolved is not None
        assert resolved.resolve() == (nested / "best.pt").resolve()
