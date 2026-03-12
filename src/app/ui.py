import streamlit as st
import requests
from PIL import Image
import io
import json

# --- CONFIGURATION ---
# We use the internal Docker DNS name "backend" to talk to the API
API_URL = "http://backend:8000"

st.set_page_config(page_title="InfraLens AI", page_icon="🏗️", layout="wide")

# --- HEADER ---
st.title("🏗️ InfraLens: Industrial Inspection AI")
st.markdown("""
**System Status:** 🟢 Online | **AI Agent:** 🧠 Llama 3 Connected
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Controls")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    st.info("Upload an image to begin inspection.")

# --- MAIN APP ---
uploaded_file = st.file_uploader("Upload Inspection Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 2. Action Buttons
    col1, col2 = st.columns(2)
    
    with col1:
        detect_btn = st.button("🔍 Detect Only (Fast)")
    
    with col2:
        analyze_btn = st.button("🛡️ Full Safety Report (Slow)")

    # 3. Handle Requests
    if detect_btn:
        with st.spinner("Running Computer Vision..."):
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                files = {"file": uploaded_file.getvalue()}
                
                response = requests.post(f"{API_URL}/predict", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Found {len(data['detections'])} objects!")
                    st.json(data["detections"])
                else:
                    st.error("Error connecting to backend.")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    if analyze_btn:
        with st.spinner("Consulting AI Agent (Llama 3)... This may take 30s..."):
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                files = {"file": uploaded_file.getvalue()}
                
                # Call the SMART endpoint
                response = requests.post(f"{API_URL}/analyze-report", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # --- DISPLAY REPORT ---
                    st.divider()
                    st.subheader("📝 Technical Safety Report")
                    
                    # Create a professional card for the report
                    with st.container(border=True):
                        st.markdown(data["safety_report"])
                    
                    # Show raw detections below
                    with st.expander("View Raw Vision Data"):
                        st.json(data["detections"])
                        
                else:
                    st.error(f"Server Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection failed: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("InfraLens v2.0 | Powered by YOLOv8 + Llama 3")