# FILE: dags/retraining_pipeline.py
"""Scheduled retraining for the InfraLens rust detector.

Flow: retrain YOLOv8 -> reload the serving model -> report the drift baseline.

Every project import is inside the task callables. A module-level import of
ultralytics pulls in torch, and if that import fails the DAG file fails to parse
and the DAG silently disappears from the Airflow UI. Importing lazily means a
dependency problem surfaces as a failed task with a traceback instead.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add src to path so Airflow can find the project modules. This must point
# outside /opt/airflow/dags: Airflow scans that directory for DAG files, so
# mounting src inside it makes Airflow try to parse every project module as a
# DAG and report import errors.
sys.path.append(os.getenv("INFRALENS_SRC", "/opt/airflow/src"))


def retrain_model(**context):
    """Retrain the detector and return a JSON-serializable summary for XCom."""
    from scripts.train_yolo import train_yolo

    model_dir = os.getenv("INFRALENS_MODEL_DIR", "/opt/airflow/models")
    result = train_yolo(
        data_yaml=os.getenv("INFRALENS_DATA_YAML", "/opt/airflow/dataset/data.yaml"),
        epochs=int(os.getenv("INFRALENS_EPOCHS", "10")),
        project_name=os.getenv("INFRALENS_EXPERIMENT", "infralens_rust_detection"),
        run_name=f"scheduled_{context['ds']}",
    )
    return {"status": "trained", "model_dir": model_dir, "result": str(result)[:200]}


def reload_serving_model(**context):
    """Tell the serving API to pick up the freshly retrained weights.

    Best-effort by design: if the API is not running (for example when only the
    DAG is being exercised) the task logs that and succeeds. The point is to
    make the retrain -> serve handoff visible in the DAG graph and to fail
    loudly when the API is reachable but rejects the new weights.
    """
    import json
    import urllib.error
    import urllib.request

    url = os.getenv(
        "INFRALENS_API_RELOAD_URL", "http://host.docker.internal:8001/admin/reload-model"
    )

    try:
        request = urllib.request.Request(url, method="POST", data=b"")
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
        return {"status": "reloaded", "response": payload}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:200]
        if exc.code in (403, 404):
            # Reload is disabled or not implemented, not broken.
            return {"status": "skipped", "reason": f"HTTP {exc.code}: {detail}"}
        raise RuntimeError(f"API rejected the reload ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        return {"status": "skipped", "reason": f"API unreachable: {exc.reason}"}


def report_drift(**context):
    """Read the serving API's drift state so the run records it."""
    import json
    import urllib.error
    import urllib.request

    url = os.getenv("INFRALENS_DRIFT_URL", "http://host.docker.internal:8001/drift")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode())
        return {"status": "reported", "drift": payload}
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        return {"status": "skipped", "reason": str(exc)}


default_args = {
    "owner": "nishat",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="infralens_continuous_learning",
    default_args=default_args,
    description="Weekly YOLOv8 retraining + model reload + drift report",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    tags=["infralens", "mlops"],
) as dag:
    train_task = PythonOperator(
        task_id="retrain_yolo_model",
        python_callable=retrain_model,
    )

    reload_task = PythonOperator(
        task_id="reload_serving_model",
        python_callable=reload_serving_model,
    )

    drift_task = PythonOperator(
        task_id="report_drift",
        python_callable=report_drift,
    )

    train_task >> reload_task >> drift_task
