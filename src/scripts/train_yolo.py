"""
InfraLens YOLOv8 Training Script with MLflow Tracking.

Trains a YOLOv8 model for rust/corrosion detection with:
- MLflow experiment tracking (metrics, params, artifacts)
- Model checkpointing (best + last)
- Validation metrics (mAP, precision, recall)
- Early stopping support
"""
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sanitize_metric_name(name: str) -> str:
    """Make a YOLO metric key acceptable to MLflow.

    YOLO reports keys such as ``metrics/precision(B)``. MLflow rejects
    parentheses, so they are replaced with underscores. The mapping is
    deterministic, so the same metric always lands under the same name.
    """
    return name.replace("(", "_").replace(")", "")


def _resolve_best_weights(results, run_name: str) -> Optional[Path]:
    """Find the checkpoint Ultralytics actually wrote.

    Ultralytics resolves ``project`` relative to its own settings directory and
    prepends ``runs/detect/``, so passing ``project="runs/train"`` writes to
    ``runs/detect/runs/train/<name>/weights/best.pt``. Hardcoding the input path
    meant the caller's existence check silently failed and the model was never
    logged or published, while the task still reported success.

    The trainer's own ``save_dir`` is authoritative, so it is tried first. The
    glob is a fallback for Ultralytics versions that do not expose it.
    """
    candidates = []

    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        candidates.append(Path(str(save_dir)) / "weights" / "best.pt")

    candidates.append(Path("runs/train") / run_name / "weights" / "best.pt")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Last resort: search for the run by name. Bounded to the runs tree so this
    # cannot walk the whole filesystem.
    for match in Path("runs").glob(f"**/{run_name}/weights/best.pt"):
        return match

    return None


def _default_data_yaml() -> str:
    """Locate the YOLO dataset YAML.

    Prefers the container mount, then falls back to the in-repo Roboflow export
    so ``python src/scripts/train_yolo.py`` works on a host checkout without
    arguments. The export's YAML references ``../train/images``, so it must be
    read from its own directory rather than copied elsewhere.
    """
    candidates = [
        os.getenv("INFRALENS_DATA_YAML", ""),
        "/opt/airflow/dataset/data.yaml",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "Rust Detection.v1i.yolov8",
            "data.yaml",
        ),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return "data/gold/yolo_dataset.yaml"


def train_yolo(
    data_yaml: Optional[str] = None,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    project_name: str = "infralens_rust_detection",
    run_name: Optional[str] = None,
):
    """Train YOLOv8 model with MLflow tracking.

    Args:
        data_yaml: Path to YOLO dataset YAML file. Defaults to the container
            mount, then the in-repo Roboflow export.
        epochs: Number of training epochs
        batch_size: Training batch size
        imgsz: Image size for training
        project_name: MLflow experiment name
        run_name: Optional run name
    """
    data_yaml = data_yaml or _default_data_yaml()
    from ultralytics import YOLO

    # 1. Setup MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(project_name)

    if run_name is None:
        run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 2. Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "data_yaml": data_yaml,
        })

        # 3. Initialize model
        model = YOLO("yolov8n.pt")  # Start from pretrained COCO weights
        print(f"Training YOLOv8 for {epochs} epochs...")

        # 4. Train
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project="runs/train",
            name=run_name,
            exist_ok=True,
            device=0 if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
        )

        # 5. Log metrics
        # YOLOv8 results contain validation metrics
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    # YOLO's keys contain parentheses, e.g. "metrics/precision(B)".
                    # MLflow rejects those: names may only contain alphanumerics,
                    # underscores, dashes, periods, spaces, colons and slashes.
                    mlflow.log_metric(_sanitize_metric_name(key), value)

        # 6. Log model
        #
        # Ultralytics resolves `project` relative to its own settings directory
        # and prepends `runs/detect/`, so the checkpoint lands at
        # runs/detect/runs/train/<name>/weights/best.pt — not the path passed in.
        # Hardcoding the input path meant this block silently never ran: the
        # model was never logged to MLflow and never published to the serving
        # path, while the task still reported success. Ask the trainer where it
        # actually wrote instead of guessing.
        best_model_path = _resolve_best_weights(results, run_name)
        if best_model_path is None:
            raise FileNotFoundError(
                f"Training finished but no best.pt was found for run {run_name!r}. "
                f"Checked the trainer's save_dir and runs/**/{run_name}/weights/."
            )

        if best_model_path.exists():
            # Log the checkpoint as an artifact rather than via
            # mlflow.pytorch.log_model. The in-memory YOLO object holds a
            # _SingleProcessDataLoaderIter, which cannot be pickled, so
            # log_model fails with "('{} cannot be pickled',
            # '_SingleProcessDataLoaderIter')" and takes the whole task down.
            # The file on disk is the artifact that matters, and it is what the
            # serving path needs.
            #
            # Best-effort: MLflow's artifact root is a host path (run_stack.sh
            # passes --default-artifact-root "$DIR/mlruns"), which a container
            # cannot write. Losing the uploaded copy must not fail an otherwise
            # successful training run — the checkpoint is published to the
            # serving path below regardless.
            try:
                mlflow.log_artifact(str(best_model_path), artifact_path="model_weights")
            except Exception as e:  # noqa: BLE001 - artifacts are optional
                print(f"WARNING: could not log weights to MLflow: {e}")

            # 7. Publish to the serving path.
            #
            # Without this the DAG trained a model, logged it to MLflow, and
            # reported success — while the API kept loading the old weights from
            # MODEL_PATH. Every component was green and the retrain had no
            # effect. The copy is what closes the loop; the reload task that
            # follows only re-reads whatever is at MODEL_PATH.
            serving_path = os.getenv("MODEL_PATH", "src/ai_models/rust_v8s_best.pt")
            try:
                serving = Path(serving_path)
                serving.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_model_path, serving)
                print(f"Published new weights to {serving}")
            except Exception as e:  # noqa: BLE001 - must not fail a good run
                print(f"WARNING: could not publish weights to {serving_path}: {e}")

        # 8. Log training plots. These live next to the weights, so derive the
        # directory from the resolved checkpoint rather than guessing again.
        # Best-effort for the same reason as the weights above.
        run_dir = best_model_path.parent.parent
        for plot_name in ["results.png", "confusion_matrix.png"]:
            plot_path = run_dir / plot_name
            if plot_path.exists():
                try:
                    mlflow.log_artifact(str(plot_path))
                except Exception as e:  # noqa: BLE001 - artifacts are optional
                    print(f"WARNING: could not log {plot_name} to MLflow: {e}")

        print(f"Training complete! Run: {run_name}")
        print(f"Best model: {best_model_path}")

        return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the InfraLens YOLOv8 rust detector."
    )
    parser.add_argument(
        "--data-yaml",
        default=None,
        help="YOLO dataset YAML. Defaults to the container mount, then the "
        "in-repo Roboflow export.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project-name", default="infralens_rust_detection")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        project_name=args.project_name,
        run_name=args.run_name,
    )
