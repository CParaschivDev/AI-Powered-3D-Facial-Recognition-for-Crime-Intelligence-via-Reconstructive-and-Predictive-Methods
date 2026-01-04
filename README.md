# AI‑Powered 3D Facial Recognition for Crime Intelligence  
**Reconstructive + Predictive Techniques (Prototype)**

This repository contains a full‑stack prototype that brings together **3D face reconstruction**, **face recognition**, and **crime analytics/forecasting** behind a FastAPI backend, with a React frontend for investigation workflows (upload → analyze → report).

Large artifacts (for example `police_face_db.sqlite`) are stored with **Git LFS**.

---

## What you get

- **3D reconstruction pipeline**: upload an image and get a reconstruction result.
- **Recognition pipeline**: identify/verify faces against watchlists/embeddings.
- **Evidence module**: upload, retrieve, verify (including watermark verification endpoints).
- **Crime analytics**: summary + hotspots + time series endpoints.
- **Bias & DPIA support**: reporting endpoints + assessment workflow.
- **Reporting**: generate a PDF report from analysis outputs.

> Full, always-up-to-date API contract is available in Swagger once the backend is running: `http://localhost:8000/docs`.

---

## Project layout

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
```

---

## Tech stack

### Backend
- **FastAPI** + **Uvicorn**
- **SQLAlchemy 2.x** + **Alembic**
- **Torch** ecosystem + CV tooling (OpenCV, scikit-image, Pillow)
- **InsightFace**, **ONNXRuntime**, **Ultralytics YOLO**
- **Prophet** (time-series forecasting)
- **FPDF2** (PDF generation)
- Optional integrations in the codebase: **Neo4j**, cryptography helpers, etc.

### Frontend
- React (`react-scripts`)
- Routing + visualisation components (see `frontend/src/`)

---

## Quick start (local)

### Prereqs
- Python 3.10+ recommended
- Node.js 18+ recommended
- Git + Git LFS

### 1) Clone and fetch large files (Git LFS)
```bash
git lfs install
git clone https://github.com/CParaschivDev/AI-Powered-3D-Facial-Recognition-for-Crime-Intelligence-via-Reconstructive-and-Predictive-Methods
cd AI-Powered-3D-Facial-Recognition-for-Crime-Intelligence-via-Reconstructive-and-Predictive-Methods
git lfs pull
```

### 2) Backend (FastAPI)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open:
- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

### 3) Frontend (React)
```bash
cd frontend
npm install
npm start
```

Frontend dev server (default): `http://localhost:3000`

---

## Quick start (Docker)

Build and run the stack:

```bash
docker compose up --build
```

This compose file includes Traefik labels for host-based routing. By default:
- Frontend is routed on `http://app.localhost`
- Backend is routed on `http://api.localhost` (and/or TLS depending on your local Traefik config)

If you prefer simple ports instead of host routing, you can add `ports:` to the `backend` and `frontend` services, e.g. `8000:8000` and `3000:3000`.

---

## Configuration (.env)

The backend uses environment variables (loaded via `pydantic-settings`) and typically reads from a root `.env`.

Common keys used by the project:
- **DB**: `DATABASE_URL`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- **Auth/Security**: `SECRET_KEY`, `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`
- **Paths/Data**: `DATA_ROOT`, dataset/watchlist directory names (e.g., `AFLW2K3D_DIR`, `UK_POLICE_DIR`, `WATCHLIST_DIRS`)
- Optional: SMTP notification settings

If this repository is public, make sure you understand what you’re publishing before committing `.env`.

---

## API overview (actual routes)

All routes are mounted under the base prefix:

- **Base prefix:** `/api/v1`

### Authentication
- `POST /api/v1/auth/token`

### 3D reconstruction
- `POST /api/v1/reconstruct`

### Recognition
- `POST /api/v1/recognize`

### Reporting
- `POST /api/v1/report`

### Evidence (multimodal)
- `POST /api/v1/evidence`
- `GET  /api/v1/evidence`
- `GET  /api/v1/evidence/{evidence_id}`
- `DELETE /api/v1/evidence/{evidence_id}`
- `POST /api/v1/evidence/verify-watermark`

### Evidence verification
- `POST /api/v1/evidence/verify`

### Bias monitoring
- `GET  /api/v1/reports`
- `GET  /api/v1/reports/{report_id}`
- `POST /api/v1/reports/{report_id}/audit`
- `GET  /api/v1/summary`
- `GET  /api/v1/trend`

### DPIA compliance
- `POST /api/v1/dpia/assessments`
- `GET  /api/v1/dpia/assessments`
- `GET  /api/v1/dpia/assessments/{assessment_id}`
- `GET  /api/v1/dpia/status`
- `POST /api/v1/dpia/assessments/{assessment_id}/approve`
- `POST /api/v1/dpia/assessments/{assessment_id}/reject`
- `POST /api/v1/dpia/run-automated-check`

### Analytics
- `GET  /api/v1/analytics/predictions`
- `POST /api/v1/analytics/run-predictions`
- `POST /api/v1/analytics/crime/context`
- `GET  /api/v1/analytics/crime/context/status`

### Crime analytics
- `GET  /api/v1/crime/summary`
- `GET  /api/v1/crime/hotspots/latest`
- `GET  /api/v1/crime/lsoa/series`
- `GET  /api/v1/crime/forces`
- `GET  /api/v1/crime/lsoas`
- `GET  /api/v1/crime/forces/monthly`
- `GET  /api/v1/crime/forces/monthly/all`
- `GET  /api/v1/crime/debug/sample`
- `GET  /api/v1/crime/debug/diagnostics`
- `GET  /api/v1/crime/debug/distincts`
- `POST /api/v1/crime/debug/clear-cache`

---

## Dependencies (from `requirements.txt`)

```text
# Core API and Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
websockets==12.0
python-dotenv==1.0.0

# Database and ORM
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9

# Data Processing and Analysis
polars==0.20.31
pandas==2.1.4
pyarrow==14.0.1
numpy==1.26.2

# Machine Learning and AI
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
scikit-learn==1.3.2
transformers==4.36.2
ultralytics==8.3.234
insightface==0.7.3
onnxruntime==1.16.3

# Computer Vision
opencv-python-headless==4.8.1.78
Pillow==10.1.0
scikit-image==0.22.0

# Authentication and Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
cryptography==41.0.7
tenseal==0.3.16

# API and HTTP
aiohttp==3.9.1
httpx==0.26.0
python-multipart==0.0.6

# Time Series and Forecasting
prophet==1.1.5

# Visualization
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0

# PDF and Document Generation
fpdf2==2.7.8

# Graph Database
neo4j==5.17.0

# Audio Processing
librosa==0.10.1
torch-audiomentations==0.11.1
pydub==0.25.1

# Utility Libraries
typing-extensions>=4.8.0
email-validator==2.1.0
jinja2==3.1.2
python-json-logger==2.0.7
watchdog==3.0.0
tenacity==8.2.3
geoip2==4.7.0
pymupdf==1.23.8
flask-cors==4.0.0
duckdb==0.9.2

# Development and Testing
pytest==7.4.3
pytest-asyncio==0.21.1
```

---

## Database & migrations (Alembic)

If you are using Postgres (recommended for full functionality), set `DATABASE_URL` in `.env` and run:

```bash
alembic upgrade head
```

If you’re experimenting locally, the project also includes SQLite artifacts (tracked with LFS), but Postgres + Alembic is the intended workflow for migrations.

---

## Training & evaluation

- Training scripts: `training/`
- Evaluation scripts: `evaluation/`
- Automation/figure tooling: `scripts/`
- Documentation figures and generated assets: `docs/`

Run tests (dev dependencies required):

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## Git notes (empty folders + large files)

### Empty folders
Git does not store directories by themselves—only files. If you need an empty folder to appear on GitHub, add a placeholder file (e.g., `.gitkeep`).

Windows CMD example:
```bat
type nul > "Data\actor_faces\.gitkeep"
```

### Large files
If you add new large artifacts, track them with Git LFS:
```bash
git lfs track "*.sqlite"
git lfs track "*.pth"
git add .gitattributes
```

---

## License
MIT License — see `LICENSE`.
