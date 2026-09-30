from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from PIL import Image
from ultralytics import YOLO

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
try:
    model = YOLO("src/ai_models/rust_v8s_best.pt") 
    print("✅ Custom Rust Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

@app.get("/")
async def health_check():
    """Heartbeat endpoint to check if the API is running."""
    return {"status": "online", "service": "InfraLens Backend"}

@app.post("/predict")
async def predict_rust(file: UploadFile = File(...)):
    """
    Standard Detection Endpoint.
    Returns: JSON with bounding boxes and confidence scores.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Read Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # 2. Run Inference
    results = model(image)
    
    # 3. Process Results
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
            })

    return {"filename": file.filename, "detections": detections}

@app.post("/analyze-report")
async def analyze_and_report(file: UploadFile = File(...)):
    """
    🚀 SMART ENDPOINT: Vision + Language
    1. Detects rust (YOLO).
    2. Generates a technical safety report (Llama 3).
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Read Image
    contents = await file.read()
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