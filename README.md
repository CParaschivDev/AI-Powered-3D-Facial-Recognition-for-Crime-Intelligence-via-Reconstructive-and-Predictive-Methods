# An AI-Powered 3-D Facial Recognition Framework for Crime Intelligence Using Reconstructive and Predictive Techniques

An end-to-end prototype that combines **3D face reconstruction**, **face recognition**, and **crime analytics/forecasting** behind a FastAPI backend, with a React + Three.js frontend for interactive visualisation and reporting.

> Note: This repository contains large assets tracked with **Git LFS** (e.g., `police_face_db.sqlite`). Make sure Git LFS is installed and enabled when cloning.

---

## What’s in this project

At a high level, the system is split into:

- **Backend (FastAPI)**  
  API for authentication, 3D reconstruction, recognition, evidence handling, bias/DPIA reporting, crime analytics, and PDF report generation.

- **Frontend (React + @react-three/fiber)**  
  UI for uploading images/evidence, viewing results, dashboards, and visual outputs.

- **Data + artifacts**  
  Local folder structure for datasets, watchlists, embeddings DBs, logs, generated figures, and reports.

- **Ops & deployment**  
  Docker Compose setup (Traefik + backend + frontend + Postgres) and Kubernetes manifests.

---

## Repository structure


├─ backend/ # FastAPI app, services, models, database logic
│ ├─ api/ # routes, schemas, main app
│ ├─ core/ # config, security, utilities
│ ├─ database/ # SQLAlchemy models + Alembic migrations
│ ├─ models/ # landmarks, reconstruction, recognition
│ ├─ orchestration/ # pipeline runners/report generator hooks
│ └─ services/ # matching, notifications, etc.

├─ frontend/ # React app (Three.js / charts / UI)
│ ├─ src/
│ └─ public/

├─ scripts/ # automation + figure generation + diagnostics
├─ evaluation/ # evaluation scripts for key ML components
├─ training/ # training scripts for landmark/recognition/reconstruction
├─ synthetic_data_generator/ # synthetic dataset generation utilities
├─ docs/ # screenshots, figures, evidence documentation
├─ reports/ # generated reports/exports
├─ kubernetes/ # k8s manifests for production-ish deployments

├─ docker-compose.yml # local orchestration
├─ Dockerfile # backend container build
├─ alembic.ini # migration configuration
├─ requirements.txt # backend dependencies
└─ requirements-dev.txt # dev/test tooling

markdown
Copy code

---

## Key backend endpoints (high level)

Once the backend is running, interactive docs are available at:

- `http://localhost:8000/docs` (Swagger UI)

Routes implemented in `backend/api/routes/` include:

- **Auth**: `POST /api/v1/auth/token`
- **3D Reconstruction**: `POST /api/v1/reconstruct`
- **Recognition**: `POST /api/v1/recognize`
- **Report generation**: `POST /api/v1/report`
- **Evidence**: `POST /api/v1/evidence`, `GET /api/v1/evidence`, `GET /api/v1/evidence/{id}`, `DELETE /api/v1/evidence/{id}`
- **Bias reporting**: `GET /api/v1/reports`, `GET /api/v1/summary`, `GET /api/v1/trend`
- **DPIA**: `POST /api/v1/dpia/assessments`, status + approve/reject endpoints
- **Crime analytics**: summary + hotspot + time series endpoints under `/api/v1/crime/*`
- **Analytics**: prediction runners and context endpoints under `/api/v1/analytics/*`

(Exact behaviour is documented in the FastAPI `/docs` UI.)

---

## Quick start (local dev)

### 1) Clone with Git LFS enabled
This repo uses Git LFS for large files. After cloning:

```bash
git lfs install
git lfs pull
2) Backend setup (Python)
Recommended: Python 3.10+ (Torch + ecosystem tend to be happiest here).

bash
Copy code
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
Run the API:

bash
Copy code
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
Open:

http://localhost:8000/ (health)

http://localhost:8000/docs (API docs)

3) Frontend setup (Node)
From frontend/:

bash
Copy code
cd frontend
npm install
npm start
Frontend dev server typically runs on http://localhost:3000.

Running with Docker Compose (recommended for a full stack demo)
This repo includes:

Traefik reverse proxy

backend (FastAPI)

frontend (React)

db (Postgres)

Start everything:

bash
Copy code
docker compose up --build
Notes on the current docker-compose.yml
The backend depends_on references redis, but the compose file does not define a redis service.
If you intend to use Celery/Redis (CELERY_BROKER_URL, CELERY_RESULT_BACKEND exist in .env), add a Redis service or remove that dependency.

Compose references a secret: ./secrets/he_context.bin. That file is not present in the ZIP; create it (or remove secrets wiring) if you want the encryption workflow enabled.

GPU support is configured for NVIDIA in deploy.resources. If you don’t have GPU or don’t want GPU mode, remove that block.

Environment configuration
The backend uses pydantic-settings and loads .env from the project root.

Your .env currently includes keys like:

DATABASE_URL, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB

SECRET_KEY, ENCRYPTION_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

DATA_ROOT, WATCHLIST_DIRS, AFLW2K3D_DIR, UK_POLICE_DIR, etc.

CELERY_BROKER_URL, CELERY_RESULT_BACKEND

Optional SMTP settings for notifications

A practical approach is to create a safe template file (recommended):

Create env.example with placeholders

Keep your real .env private (especially if the repo is public)

Data, watchlists, and model artifacts
Data root
Backend settings include DATA_ROOT and directory names such as:

actor_faces, actress_faces

AFLW2000

UK DATA CRIME 2022 - 2025

Place your datasets under a local folder and point DATA_ROOT to it in .env.
This prevents hardcoding personal machine paths.

Model weights
The model loader looks for weights under ./logs/... paths, e.g.:

./logs/landmarks/landmark_model_best.pth

./logs/reconstruction/reconstruction_model_best.pth

./logs/recognition/recognition_model_best.pth

If these files don’t exist, you can:

train them via the scripts in training/, or

drop pre-trained weights into the expected locations.

Training and evaluation
Training
Scripts live in training/ (landmarks, recognition, reconstruction):

bash
Copy code
python training/landmark_train.py
python training/reconstruction_train.py
python training/recognition_train.py
Evaluation
Evaluation scripts live in evaluation/:

bash
Copy code
python evaluation/evaluate_landmarks.py
python evaluation/evaluate_reconstruction.py
python evaluation/evaluate_recognition.py
Reports and documentation assets
Generated reports/exports: reports/

Supporting screenshots/figures: docs/

There is a dedicated figures folder with scripts and HTML/SVG assets:

docs/figures/
Includes diagram generation scripts (pipeline, schema diagrams, governance stack, etc.)

Common issues and fixes
“Large files detected” / 100MB limit
Large binaries must be tracked with Git LFS (already configured for police_face_db.sqlite).
If you add new large assets, track them explicitly:

bash
Copy code
git lfs track "*.sqlite"
git lfs track "*.pth"
git add .gitattributes
Empty folders not showing on GitHub
Git tracks files, not directories. If you need empty folders preserved, add a placeholder:

bash
Copy code
type nul > "Data\actor_faces\.gitkeep"
Security and privacy
This project includes features that touch sensitive domains (biometrics, evidence handling, crime context). If you use real-world data:

avoid committing personal data or secrets into the repo

prefer private repositories for any sensitive datasets

sanitize databases used for demos

document data handling assumptions clearly in your report/docs

License
MIT License. See LICENSE.
