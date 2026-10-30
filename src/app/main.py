import io
import os
import sys
import time

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# src/ must be on the path before the project modules below are imported. This
# runs at import time, so it cannot be moved into a function.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth import auth_enabled, require_api_key  # noqa: E402
from ratelimit import PREDICT_LIMIT, REPORT_LIMIT, limiter  # noqa: E402
from ratelimit import install as install_rate_limit
from uploads import enforce_content_length, read_upload  # noqa: E402
from versioning import install as install_versioning

# Prometheus instrumentation
try:
    from fastapi import Response
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    REQUEST_COUNT = Counter(
        "infralens_requests_total",
        "Total InfraLens API requests",
        ["endpoint", "status"],
    )
    DETECTION_COUNT = Counter(
        "infralens_detections_total",
        "Total rust detections returned",
        ["class_name"],
    )
    INFERENCE_LATENCY = Histogram(
        "infralens_inference_latency_seconds",
        "YOLO inference latency in seconds",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    MODEL_LOADED = Gauge(
        "infralens_model_loaded",
        "1 if the rust detection model is loaded, 0 otherwise",
    )
    DRIFT_PSI = Gauge(
        "infralens_detection_drift_psi",
        "Population Stability Index of the live detection distribution vs the "
        "validation baseline",
    )
    DRIFT_STATUS = Gauge(
        "infralens_drift_status",
        "Drift interpretation: 0=stable, 1=moderate, 2=significant, -1=unknown",
    )
    AB_AGREEMENT = Gauge(
        "infralens_ab_agreement_ratio",
        "Fraction of images where the shadow model agreed with the champion "
        "on the set of detected classes",
    )
    AB_COMPARISONS = Counter(
        "infralens_ab_comparisons_total",
        "Total shadow-model A/B comparisons run",
    )
    AB_LATENCY_DELTA = Histogram(
        "infralens_ab_latency_delta_seconds",
        "Shadow minus champion inference latency in seconds",
        buckets=(-1.0, -0.5, -0.1, -0.05, 0.0, 0.05, 0.1, 0.5, 1.0),
    )
except ImportError:  # pragma: no cover
    REQUEST_COUNT = DETECTION_COUNT = INFERENCE_LATENCY = MODEL_LOADED = None
    DRIFT_PSI = DRIFT_STATUS = None
    AB_AGREEMENT = AB_COMPARISONS = AB_LATENCY_DELTA = None
    generate_latest = CONTENT_TYPE_LATEST = Response = None

# Import our new AI Agent (The Brain)
# Ensure you created src/core/agent.py as discussed!
try:
    from src.core.agent import generate_safety_report
except ImportError:
    print("WARNING: Could not import AI Agent. Make sure src/core/agent.py exists.")
    generate_safety_report = None

app = FastAPI(
    title="InfraLens API",
    description="Industrial Rust Detection & Safety Reporting System",
    version="2.0.0"
)
install_rate_limit(app)

# 1. CORS Configuration (Allow Streamlit to talk to FastAPI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the AI Model (The Eyes)
# We load this once at startup to save RAM
MODEL_PATH = os.getenv("MODEL_PATH", "src/ai_models/rust_v8s_best.pt")

# Optional challenger weights for shadow A/B comparison. When set, /predict-ab
# runs both models on the same image and reports agreement, so a candidate
# model can be evaluated on live traffic before it is promoted.
SHADOW_MODEL_PATH = os.getenv("SHADOW_MODEL_PATH", "")
shadow_model = None


def load_model() -> bool:
    """Load (or reload) the YOLO detector from disk.

    Returns True on success. Kept as a function so the retraining DAG can
    hot-swap the weights over HTTP instead of requiring a container restart.
    """
    global model

    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ Custom Rust Model Loaded Successfully ({MODEL_PATH})")
        if MODEL_LOADED is not None:
            MODEL_LOADED.set(1)
        return True
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        model = None
        if MODEL_LOADED is not None:
            MODEL_LOADED.set(0)
        return False


load_model()


def load_shadow_model() -> bool:
    """Load the optional challenger model used for shadow A/B comparison.

    Returns False when SHADOW_MODEL_PATH is unset, which is the default: A/B
    comparison is opt-in so a normal deployment does not pay for a second model
    in memory.
    """
    global shadow_model

    if not SHADOW_MODEL_PATH:
        shadow_model = None
        return False

    try:
        shadow_model = YOLO(SHADOW_MODEL_PATH)
        print(f"✅ Shadow model loaded for A/B comparison ({SHADOW_MODEL_PATH})")
        return True
    except Exception as e:
        print(f"❌ Error loading shadow model: {e}")
        shadow_model = None
        return False


load_shadow_model()


@app.post("/admin/reload-model", dependencies=[Depends(require_api_key)])
async def reload_model():
    """Re-read the detector weights from disk without restarting the API.

    The retraining DAG writes new weights, then calls this so the fresh model is
    served immediately. Disabled unless ENABLE_MODEL_RELOAD=true: swapping model
    weights is a privileged operation and should not be reachable by default.
    """
    if os.getenv("ENABLE_MODEL_RELOAD", "false").lower() not in {"1", "true", "yes"}:
        raise HTTPException(
            status_code=403,
            detail="Model reload is disabled. Set ENABLE_MODEL_RELOAD=true to enable it.",
        )

    if not load_model():
        raise HTTPException(status_code=500, detail="Reload failed: model could not be loaded")

    return {"status": "reloaded", "model_path": MODEL_PATH}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if generate_latest is None:
        raise HTTPException(status_code=501, detail="prometheus_client not installed")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def health_check():
    """Heartbeat endpoint to check if the API is running."""
    return {"status": "online", "service": "InfraLens Backend", "auth_enabled": auth_enabled()}

@app.post("/predict", dependencies=[Depends(require_api_key), Depends(enforce_content_length)])
@limiter.limit(PREDICT_LIMIT)
async def predict_rust(request: Request, file: UploadFile = File(...)):
    """
    Standard Detection Endpoint.
    Returns: JSON with bounding boxes and confidence scores.
    """
    if not model:
        if REQUEST_COUNT is not None:
            REQUEST_COUNT.labels(endpoint="predict", status="500").inc()
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Read Image
    contents = await read_upload(file)
    image = Image.open(io.BytesIO(contents))

    # 2. Run Inference
    t0 = time.perf_counter()
    results = model(image)
    elapsed = time.perf_counter() - t0

    # 3. Process Results
    detections = []
    for result in results:
        for box in result.boxes:
            cls_name = model.names[int(box.cls)]
            detections.append({
                "class": cls_name,
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
            })

    # 4. Record metrics
    if REQUEST_COUNT is not None:
        REQUEST_COUNT.labels(endpoint="predict", status="200").inc()
        INFERENCE_LATENCY.observe(elapsed)
        for det in detections:
            DETECTION_COUNT.labels(class_name=det["class"]).inc()

    # 5. Drift: track the detection distribution against the validation baseline
    drift = observe_detection_drift(detections)

    return {"filename": file.filename, "detections": detections, "drift": drift}


def _run_detector(detector, image):
    """Run one detector and return (detections, elapsed_seconds)."""
    t0 = time.perf_counter()
    results = detector(image)
    elapsed = time.perf_counter() - t0

    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": detector.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0],
            })
    return detections, elapsed


@app.post("/predict-ab", dependencies=[Depends(require_api_key), Depends(enforce_content_length)])
@limiter.limit(PREDICT_LIMIT)
async def predict_ab(request: Request, file: UploadFile = File(...)):
    """Shadow A/B comparison of the champion and challenger detectors.

    Both models run on the same image. The response reports each model's
    detections, their latencies, and whether they agreed on the set of detected
    classes. This is how a candidate model is evaluated on live traffic before
    promotion: the champion still serves /predict, so a bad challenger cannot
    affect users.

    Requires SHADOW_MODEL_PATH to be set; returns 503 otherwise.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Champion model not loaded")
    if not shadow_model:
        raise HTTPException(
            status_code=503,
            detail="No shadow model configured. Set SHADOW_MODEL_PATH to enable A/B comparison.",
        )

    contents = await read_upload(file)
    image = Image.open(io.BytesIO(contents))

    champion_detections, champion_elapsed = _run_detector(model, image)
    shadow_detections, shadow_elapsed = _run_detector(shadow_model, image)

    champion_classes = {d["class"] for d in champion_detections}
    shadow_classes = {d["class"] for d in shadow_detections}
    agreement = champion_classes == shadow_classes

    if AB_COMPARISONS is not None:
        AB_COMPARISONS.inc()
        AB_AGREEMENT.set(1.0 if agreement else 0.0)
        AB_LATENCY_DELTA.observe(shadow_elapsed - champion_elapsed)

    return {
        "filename": file.filename,
        "champion": {
            "model_path": MODEL_PATH,
            "detections": champion_detections,
            "latency_ms": round(champion_elapsed * 1000, 2),
        },
        "challenger": {
            "model_path": SHADOW_MODEL_PATH,
            "detections": shadow_detections,
            "latency_ms": round(shadow_elapsed * 1000, 2),
        },
        "agreement": agreement,
        "champion_only_classes": sorted(champion_classes - shadow_classes),
        "challenger_only_classes": sorted(shadow_classes - champion_classes),
        "latency_delta_ms": round((shadow_elapsed - champion_elapsed) * 1000, 2),
    }


# --- Detection-distribution drift -------------------------------------------

# Baseline detection distribution measured on the validation split. Overridable
# so a retrained model can ship its own baseline without a code change.
_DEFAULT_BASELINE = {"Corrosion": 0.5, "Rust": 0.5}
_drift_baseline: dict = dict(_DEFAULT_BASELINE)
_detection_window: list = []
_DRIFT_WINDOW = 100


def set_drift_baseline(distribution: dict) -> None:
    """Set the reference detection distribution used for PSI."""
    global _drift_baseline

    total = sum(distribution.values())
    if total <= 0:
        return
    _drift_baseline = {k: v / total for k, v in distribution.items()}


def get_drift_baseline() -> dict:
    """Return the current reference distribution."""
    return dict(_drift_baseline)


def _drift_samples(n: int = 1000):
    """Expand the baseline and live class proportions into weighted samples.

    PSI is defined on a distribution, so each class proportion is expanded into
    a weighted sample of class indices. This keeps the standard PSI formula
    while comparing categorical distributions.

    Returns:
        (baseline_sample, live_sample, labels)
    """
    labels = sorted(set(_detection_window) | set(_drift_baseline))
    if not labels:
        return np.array([]), np.array([]), labels

    baseline_props = np.array([_drift_baseline.get(c, 0.0) for c in labels], dtype=float)
    live_counts = np.array([_detection_window.count(c) for c in labels], dtype=float)
    total = live_counts.sum()
    live_props = live_counts / total if total > 0 else np.zeros_like(live_counts)

    baseline_sample = np.repeat(np.arange(len(labels)), np.round(baseline_props * n).astype(int))
    live_sample = np.repeat(np.arange(len(labels)), np.round(live_props * n).astype(int))
    return baseline_sample, live_sample, labels


def observe_detection_drift(detections: list, min_samples: int = 30) -> dict:
    """Record one image's detections and recompute PSI.

    Drift for a detector is measured on the *class distribution of detections*,
    not on raw pixels: a shift from mostly-Corrosion to mostly-Rust is the
    actionable signal, and it needs no labels to compute.

    Never raises: drift reporting must not break a prediction.
    """
    global _detection_window

    try:
        from src.utils.drift import classify_psi, compute_psi

        # One observation per image: the dominant class, or "none" when the
        # model finds nothing. Counting every box would let one busy image
        # dominate the window.
        if detections:
            dominant = max(detections, key=lambda d: d["confidence"])["class"]
        else:
            dominant = "none"

        _detection_window.append(dominant)
        if len(_detection_window) > _DRIFT_WINDOW:
            del _detection_window[0]

        if len(_detection_window) < min_samples:
            if DRIFT_STATUS is not None:
                DRIFT_STATUS.set(-1)
            return {"enabled": True, "status": "insufficient_data",
                    "samples": len(_detection_window)}

        baseline_sample, live_sample, labels = _drift_samples()
        if baseline_sample.size == 0 or live_sample.size == 0:
            return {"enabled": True, "status": "insufficient_data"}

        psi = compute_psi(baseline_sample, live_sample, bins=max(2, len(labels)))
        status = classify_psi(psi)

        if DRIFT_PSI is not None:
            DRIFT_PSI.set(psi)
            DRIFT_STATUS.set({"stable": 0, "moderate": 1, "significant": 2}.get(status, -1))

        return {"enabled": True, "status": status, "psi": round(psi, 6),
                "samples": len(_detection_window)}
    except Exception as exc:  # noqa: BLE001 - drift must never break /predict
        return {"enabled": True, "error": str(exc)}


@app.get("/drift", dependencies=[Depends(require_api_key)])
def drift_report():
    """Detection-distribution drift against the validation baseline."""
    from src.utils.drift import classify_psi, compute_psi

    if len(_detection_window) < 30:
        return {
            "enabled": True,
            "status": "insufficient_data",
            "samples": len(_detection_window),
            "baseline": get_drift_baseline(),
        }

    baseline_sample, live_sample, labels = _drift_samples()
    live_counts = {c: _detection_window.count(c) for c in labels}
    total = sum(live_counts.values())
    psi = compute_psi(baseline_sample, live_sample, bins=max(2, len(labels)))

    return {
        "enabled": True,
        "status": classify_psi(psi),
        "psi": round(psi, 6),
        "samples": len(_detection_window),
        "observed": {c: live_counts[c] / total for c in labels},
        "baseline": get_drift_baseline(),
    }

@app.post("/analyze-report", dependencies=[Depends(require_api_key), Depends(enforce_content_length)])
@limiter.limit(REPORT_LIMIT)
async def analyze_and_report(request: Request, file: UploadFile = File(...)):
    """
    🚀 SMART ENDPOINT: Vision + Language
    1. Detects rust (YOLO).
    2. Generates a technical safety report (Llama 3).
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Read Image
    contents = await read_upload(file)
    image = Image.open(io.BytesIO(contents))

    # 2. Run YOLO (The Eyes)
    results = model(image)

    # 3. Format Data for the LLM
    # We strip out the heavy pixel data and just send the facts (stats) to Llama 3
    detection_summary = []
    rust_count = 0
    highest_conf = 0.0

    for result in results:
        for box in result.boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            detection_summary.append(f"- Found {cls_name} with {conf:.2f} confidence.")

            if "rust" in cls_name.lower():
                rust_count += 1
                if conf > highest_conf:
                    highest_conf = conf

    # Create a clean text summary for the AI
    if not detection_summary:
        llm_input = "No visible damage or rust detected in this image."
    else:
        llm_input = "\n".join(detection_summary)
        llm_input += f"\n\nTotal Rust Spots: {rust_count}"
        llm_input += f"\nMax Confidence: {highest_conf:.2f}"

    # 4. Run Llama 3 (The Brain) via our Agent
    if generate_safety_report:
        try:
            print("🧠 Sending data to Llama 3...")
            safety_report = generate_safety_report(llm_input)
        except Exception as e:
            safety_report = f"Error generating report: {str(e)}"
    else:
        safety_report = "AI Agent module not found. Check server logs."

    return {
        "filename": file.filename,
        "detections": detection_summary, # Raw data for UI
        "safety_report": safety_report   # The "Consultant" output
    }


# ---------------------------------------------------------------------------
# API versioning
# ---------------------------------------------------------------------------
# Called at the *end* of the module on purpose. `install_versioning` clones the
# routes that already exist, so running it next to `install_rate_limit(app)`
# (before any @app.post/@app.get has executed) would find nothing to clone and
# silently alias zero endpoints. Keeping it last means every route defined above
# gets a /v1 alias, and the unversioned paths keep working unchanged.
install_versioning(app)
