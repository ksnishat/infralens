"""
InfraLens YOLOv8 Training Script with MLflow Tracking.

Trains a YOLOv8 model for rust/corrosion detection with:
- MLflow experiment tracking (metrics, params, artifacts)
- Model checkpointing (best + last)
- Validation metrics (mAP, precision, recall)
- Early stopping support
"""
import os
import sys
from datetime import datetime
from pathlib import Path

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
    data_yaml: str | None = None,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    project_name: str = "infralens_rust_detection",
    run_name: str | None = None,
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
        best_model_path = Path(f"runs/train/{run_name}/weights/best.pt")
        if best_model_path.exists():
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                conda_env={
                    "channels": ["conda-forge"],
                    "dependencies": [
                        "python=3.10",
                        "pip",
                        {"pip": ["ultralytics>=8.0.0", "torch>=2.0.0"]},
                    ],
                },
            )
            mlflow.log_artifact(str(best_model_path), artifact_path="model_weights")

        # 7. Log training plots
        for plot_name in ["results.png", "confusion_matrix.png"]:
            plot_path = Path(f"runs/train/{run_name}/{plot_name}")
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path))

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
