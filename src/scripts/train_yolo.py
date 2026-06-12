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
import mlflow
import mlflow.pytorch
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_yolo(
    data_yaml: str = "data/gold/yolo_dataset.yaml",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    project_name: str = "infralens_rust_detection",
    run_name: str = None,
):
    """Train YOLOv8 model with MLflow tracking.

    Args:
        data_yaml: Path to YOLO dataset YAML file
        epochs: Number of training epochs
        batch_size: Training batch size
        imgsz: Image size for training
        project_name: MLflow experiment name
        run_name: Optional run name
    """
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
                    mlflow.log_metric(key, value)

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
    train_yolo()