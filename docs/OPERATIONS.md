# InfraLens — Operations Reference

Setup, configuration and troubleshooting. The [README](../README.md) covers what the project is and how it scores.

## Running Tests
```bash
uv run --frozen --extra dev python -m pytest tests/ -v   # 117 passed, 8 deselected
```

The 8 skipped tests are integration tests that need the live docker stack. They
are excluded from CI (which has no model checkpoint or dataset) and run against
real services instead of an in-process `TestClient`:

```bash
./run_stack.sh
LIVE_STACK=1 pytest tests/test_integration_live.py -v   # 8 passed
```

They assert the API detects objects in a real image, that Prometheus is actually
scraping the API (not merely healthy), and that MLflow, Grafana and MinIO are
reachable.

## Tech Stack
- **AI**: YOLOv8s detection, Ollama + a local LLM for report generation
- **Backend**: FastAPI + Uvicorn, SQLAlchemy + PostgreSQL, Alembic
- **Frontend**: Streamlit
- **Monitoring**: Prometheus + Grafana, structured JSON logging
- **DevOps**: Docker Compose, Kubernetes, Helm, Terraform, DVC
- **GPU**: NVIDIA device plugin for Kubernetes

## Docker Compose Notes
The `ollama` service previously declared `network_mode: "bridge"` alongside an
implicit default network, which makes `docker compose up` fail with a
"mutually exclusive network_mode and networks" error. This has been removed.

## Kubernetes Deployment
```bash
helm install infralens ./helm-chart
# or
kubectl apply -f k8s/
```

## Monitoring & Live Demo
```bash
./run_stack.sh                      # API + MLflow + MinIO + Prometheus + Grafana
python3 provision_dashboards.py infralens   # datasource + dashboard
./verify_stack.sh                    # 7 checks, exit 0 when all pass
```

| Service | URL |
|---------|-----|
| FastAPI (Swagger) | http://localhost:8001/docs |
| Prometheus | http://localhost:9091 |
| Grafana | http://localhost:3002 (`admin` / `admin`) |
| Streamlit UI | http://localhost:8502 |

> **Running the stack:** `./run_stack.sh` starts this project's API, MLflow, MinIO, Prometheus and Grafana. Dashboard: `streamlit run src/app/ui.py` (port 8502). Airflow and the feature store run through `docker compose`. The four-project orchestrator is kept in the author's workspace and is not part of this repository.

### Exposed metrics
| Metric | Type | Meaning |
|--------|------|---------|
| `infralens_requests_total` | counter | Requests, labelled by `endpoint` and `status` |
| `infralens_detections_total` | counter | Detections returned, labelled by `class_name` |
| `infralens_inference_latency_seconds` | histogram | YOLO inference latency |
| `infralens_model_loaded` | gauge | 1 when the model is loaded, 0 otherwise |
| `infralens_detection_drift_psi` | gauge | PSI of the live detection distribution vs the validation baseline |
| `infralens_drift_status` | gauge | 0 stable, 1 moderate, 2 significant, -1 unknown |

### Drift detection
`GET /drift` reports the Population Stability Index between the live detection
distribution and the validation baseline. Drift for a detector is measured on
*what the model reports* (mostly-Corrosion vs mostly-Rust), not on raw pixels,
because a shift in the class mix is the actionable signal and it needs no labels
to compute. One observation is recorded per image — the highest-confidence class,
or `none` when nothing is found — so a single busy image cannot dominate the
window.

Verified discrimination: a balanced stream gives PSI 0.0006 (stable) while an
all-Corrosion stream gives 6.91 (significant).

### Scheduled retraining
`dags/retraining_pipeline.py` runs weekly: retrain YOLOv8 → reload the serving
model → report drift.

```bash
docker compose up -d airflow-postgres airflow-webserver airflow-scheduler
docker exec infralens_airflow_scheduler airflow dags unpause infralens_continuous_learning
docker exec infralens_airflow_scheduler airflow dags trigger infralens_continuous_learning
```

Airflow UI: http://localhost:8082 (`admin` / `admin`).

The reload step calls `POST /admin/reload-model`, which is disabled unless
`ENABLE_MODEL_RELOAD=true`. Without it a retrained model would sit on disk while
the running process kept serving the old weights.

## Troubleshooting
| Issue | Solution |
|-------|----------|
| **YOLO model not found** | Train with the command above and copy `best.pt` to `src/ai_models/rust_v8s_best.pt`, or point `MODEL_PATH` at an existing checkpoint |
| **`ModuleNotFoundError: mlflow / cv2 / PIL`** | These are declared in `pyproject.toml`; re-run `uv sync --extra dev` |
| **`docker compose up` → "mutually exclusive network_mode"** | Fixed in this commit; pull the latest `docker-compose.yml` |
| **Ollama connection refused** | Verify Ollama is running on port 11434 |
| **`/metrics` returns 404** | Fixed in this commit — the endpoint is registered on the app |
| **Grafana shows "No data"** | The provisioning or dashboards directory is not mounted, or Grafana is not on the `mlops-stack` network so it cannot resolve `prom_<project>` | Run `python3 provision_dashboards.py` — it reports which of the two it is |
| **Agent report empty** | Check the Ollama model is loaded: `ollama list` |

## Project Structure
```plaintext
infralens/
├── dags/
├── docs/
├── helm-chart/
├── infrastructure/
├── k8s/
├── src/
│   ├── app/
│   ├── core/
│   ├── scripts/
│   ├── utils/
│   ├── auth.py
│   ├── config.py
│   ├── ratelimit.py
│   ├── retry.py
│   ├── uploads.py
│   └── versioning.py
├── terraform/
├── tests/
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.gpu
├── Dockerfile.serve
├── dvc.yaml
├── load_test.py
├── Makefile
├── params.yaml
├── provision_dashboards.py
├── pyproject.toml
├── README.md
├── requirements-gpu.txt
├── requirements-serve.txt
├── run_stack.sh
├── uv.lock
└── verify_stack.sh
```
