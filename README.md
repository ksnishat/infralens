# InfraLens: Industrial Rust Detection System

![Python](https://img.shields.io/badge/Python-3.10-00599C?style=flat&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Microservices-2496ED?style=flat&logo=docker&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8-blue)
![Llama 3](https://img.shields.io/badge/GenAI-Llama3-orange)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

**InfraLens** is an automated computer vision system for industrial safety. It detects corrosion on infrastructure using a custom-trained **YOLOv8** model and employs a **Generative AI Agent (Llama 3)** to draft technical safety reports in German/English compliant with ISO standards.

---

## 🚀 Key Features
* **🔍 Custom Vision:** Fine-tuned YOLOv8 model (`rust_v8s_best.pt`) optimized for industrial surface defects.
* **🧠 AI Safety Consultant:** Local **Llama 3** agent (via Ollama) that analyzes detection data and writes maintenance recommendations.
* **🏗️ Microservices Architecture:** Fully containerized setup with separate Backend (FastAPI), Frontend (Streamlit), and AI Engine (Ollama).
* **⚡ Production Ready:** Includes **Kubernetes** manifests (`k8s/`) for scalable deployment.

---

# InfraLens: Industrial Rust Detection System

![Python](https://img.shields.io/badge/Python-3.10-00599C?style=flat&logo=python&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-Microservices-2496ED?style=flat&logo=docker&logoColor=white) ![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8-blue) ![Llama 3](https://img.shields.io/badge/GenAI-Llama3-orange) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

InfraLens is an automated computer-vision system for industrial safety that detects corrosion on infrastructure using a custom-trained YOLOv8 model and generates ISO-style technical reports via a local Llama 3 agent.

## Key Features

- **Custom Vision:** Fine-tuned YOLOv8 model (`src/ai_models/rust_v8s_best.pt`) for surface-defect detection.
- **AI Safety Consultant:** Local Llama 3 agent (via Ollama) for automated report generation and recommendations.
- **Microservices Architecture:** Containerized Backend (FastAPI), Frontend (Streamlit), and AI Engine.
- **Deployment:** Docker Compose for local development and Kubernetes manifests in `k8s/` for production.

## Tech Stack

- AI: YOLOv8, Ollama + Llama 3
- Orchestration: LangChain (optional)
- Backend: FastAPI + Uvicorn
- Frontend: Streamlit
- DevOps: Docker Compose, Kubernetes

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/ksnishat/infralens.git
cd infralens
```

2. Start services with Docker Compose (local dev):

```bash
docker compose up -d
```

3. Open the apps:

- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs

Notes:

- Place model weights under `src/ai_models/` if not present (`rust_v8s_best.pt`, etc.).
- Configure Ollama/LLM locally according to your environment before starting the agent service.

## Architecture (high level)

- Input: Image uploaded via Streamlit UI.
- Detection: Backend runs YOLOv8 inference and returns detection boxes/scores.
- Agent: Detection metrics passed to Llama 3 agent to evaluate severity and produce a technical report.
- Output: Report displayed in the UI and downloadable as PDF/Markdown.

## Contact

Developed by Khaled. For collaboration or questions open an issue or contact the maintainer.
