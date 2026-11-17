#!/usr/bin/env python3
"""
InfraLens Per-Class Evaluation Script

Loads the trained YOLOv8 model and computes per-class AP@0.5 and AP@0.5:0.95
on the validation set. Logs results to MLflow and prints a summary table.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import mlflow.pytorch
from ultralytics import YOLO

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_best_weights() -> Path:
    """Find the best.pt checkpoint."""
    candidates = [
        Path("runs/detect/runs_rust/train/weights/best.pt"),
        Path("runs/train/runs_rust/weights/best.pt"),
        Path("src/ai_models/rust_v8s_best.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No best.pt found in expected locations")


def _load_data_yaml() -> dict:
    """Load the dataset YAML to get class names."""
    candidates = [
        "/opt/airflow/dataset/data.yaml",
        Path(__file__).parent.parent.parent / "Rust Detection.v1i.yolov8" / "data.yaml",
    ]
    for c in candidates:
        if Path(c).exists():
            with open(c) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("data.yaml not found")


def evaluate_model(
    model_path: Path,
    data_yaml: str,
    conf: float = 0.001,
    iou: float = 0.6,
    imgsz: int = 640,
) -> dict:
    """Run YOLO validation and return per-class metrics."""
    model = YOLO(str(model_path))
    
    # Run validation
    results = model.val(
        data=str(data_yaml),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=True,
    )
    
    # Extract per-class metrics
    if hasattr(results, 'box') and results.box is not None:
        ap50 = results.box.ap50  # shape (nc,)
        ap = results.box.ap      # shape (nc, 10) for AP@0.5:0.95
        ap_class = results.box.ap_class_index  # class indices
        
        # Get class names from data.yaml (list, not dict)
        data = _load_data_yaml()
        names = data.get('names', [])
        class_names = {i: name for i, name in enumerate(names)}
        
        results_dict = {
            'mAP50': float(ap50.mean()) if len(ap50) > 0 else 0.0,
            'mAP50-95': float(ap.mean()) if len(ap) > 0 else 0.0,
            'per_class': {}
        }
        
        for i, cls_idx in enumerate(ap_class):
            cls_name = class_names.get(int(cls_idx), f"class_{cls_idx}")
            results_dict['per_class'][cls_name] = {
                'AP50': float(ap50[i]),
                'AP50-95': float(ap[i].mean()),
            }
        
        return results_dict
    
    return {'mAP50': 0.0, 'mAP50-95': 0.0, 'per_class': {}}


def log_to_mlflow(metrics: dict, model_path: Path):
    """Log metrics to MLflow."""
    mlflow.set_experiment("infralens-evaluation")
    with mlflow.start_run(run_name="per-class-evaluation"):
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_metric("mAP50", metrics['mAP50'])
        mlflow.log_metric("mAP50-95", metrics['mAP50-95'])
        
        for cls_name, metrics_cls in metrics['per_class'].items():
            mlflow.log_metric(f"AP50_{cls_name}", metrics_cls['AP50'])
            mlflow.log_metric(f"AP50-95_{cls_name}", metrics_cls['AP50-95'])


def print_results_table(metrics: dict):
    """Print a formatted table of per-class results."""
    print("\n" + "=" * 70)
    print(f"{'Class':<20} {'AP@0.5':>10} {'AP@0.5:0.95':>12}")
    print("-" * 70)
    for cls_name, m in metrics['per_class'].items():
        print(f"{cls_name:<20} {m['AP50']:>10.4f} {m['AP50-95']:>12.4f}")
    print("-" * 70)
    print(f"{'mAP@0.5':<20} {metrics['mAP50']:>10.4f} {metrics['mAP50-95']:>12.4f}")
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate InfraLens YOLOv8 model per class")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()
    
    # Resolve model path
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = _resolve_best_weights()
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    
    # Resolve data.yaml
    if args.data:
        data_yaml = Path(args.data)
    else:
        data_yaml = Path("/opt/airflow/dataset/data.yaml")
        if not data_yaml.exists():
            data_yaml = Path("Rust Detection.v1i.yolov8/data.yaml")
    
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found at {data_yaml}")
        sys.exit(1)
    
    print(f"Evaluating model: {model_path}")
    print(f"Using data: {data_yaml}")
    
    # Run evaluation
    metrics = evaluate_model(
        model_path=model_path,
        data_yaml=str(data_yaml),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    
    # Print results
    print_results_table(metrics)
    
    # Log to MLflow
    if not args.no_mlflow:
        try:
            log_to_mlflow(metrics, model_path)
            print("Logged to MLflow")
        except Exception as e:
            print(f"Warning: MLflow logging failed: {e}")
    
    # Save results to JSON for README update
    output_path = Path("evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
