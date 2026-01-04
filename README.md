# An AI-Powered 3-D Facial Recognition Framework for Crime Intelligence Using Reconstructive and Predictive Techniques

This repository contains a full-stack prototype for crime-intelligence workflows combining:

- **3D facial reconstruction** (geometry + landmarks)
- **Face recognition** (embeddings + watchlist/search)
- **Crime analytics & forecasting** (time series + hotspot/summary views)
- **Evidence handling** (upload, retrieval, verification endpoints)
- **Bias + DPIA support** (reporting and assessment endpoints)
- **Reporting** (PDF report generation)

> Large files (e.g., `police_face_db.sqlite`, model weights) are handled via **Git LFS**.

---

## Table of contents
- [Project layout](#project-layout)
- [Tech stack](#tech-stack)
- [Quick start (local)](#quick-start-local)
- [Quick start (Docker)](#quick-start-docker)
- [Configuration (.env)](#configuration-env)
- [API overview](#api-overview)
- [Data & model artifacts](#data--model-artifacts)
- [Training & evaluation](#training--evaluation)
- [Notes on Git + empty folders](#notes-on-git--empty-folders)
- [License](#license)

---

## Project layout

> This is intentionally in a code block so GitHub renders it correctly.

```text
.
├─ backend/                     FastAPI application (routes, services, DB)
│  ├─ api/
│  │  ├─ main.py                FastAPI app entrypoint
│  │  ├─ routes/                API endpoints (auth, reconstruct, recognize, crime, dpia, bias, evidence, report)
│  │  └─ models/                Pydantic schemas
│  ├─ core/                     Config, security, utilities
│  ├─ database/                 SQLAlchemy + Alembic migrations
│  ├─ models/                   ML/vision modules (landmarks/reconstruction/recognition)
│  ├─ orchestration/            Pipeline runners + report generation
│  └─ services/                 Supporting services (matching, helpers, etc.)
│
├─ frontend/                    React UI (dashboard, uploads, visualisation)
│  ├─ src/
│  └─ public/
│
├─ scripts/                     Automation + diagnostics + figure generation
├─ evaluation/                  Evaluation scripts for ML components
├─ training/                    Training scripts for landmarks/recognition/reconstruction
├─ synthetic_data_generator/    Utilities for synthetic dataset generation
├─ docs/                        Figures, screenshots, documentation artefacts
├─ reports/                     Generated exports/reports
├─ kubernetes/                  K8s manifests (deployment-oriented)
├─ Data/                        Local data layout (datasets, watchlists, processed outputs)
├─ logs/                        Training logs + model outputs
│
├─ docker-compose.yml           Local orchestration
├─ Dockerfile                   Backend container build
├─ alembic.ini                  DB migration config
├─ requirements.txt             Backend dependencies
└─ requirements-dev.txt         Dev/testing dependencies

Tech stack
Backend
•	FastAPI + Uvicorn
•	SQLAlchemy + Alembic (+ Postgres driver)
•	Torch ecosystem + vision libraries (OpenCV, scikit-image, Pillow)
•	InsightFace / ONNXRuntime / Ultralytics YOLO (as used by the project code & deps)
•	Prophet for forecasting
•	FPDF2 for PDF report generation
•	Optional integrations in codebase: Neo4j, crypto utilities, etc.
Frontend
•	React (react-scripts)
•	UI components + charts/visualisation (as used in /frontend/src)
