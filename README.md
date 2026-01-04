# An AI-Powered 3-D Facial Recognition Framework for Crime Intelligence Using Reconstructive and Predictive Techniques

# AI-Powered 3D Facial Recognition for Crime Intelligence (Reconstructive + Predictive)

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
  Docker Compose setup (Traefik + backend + frontend + Postgres + Ollama) and Kubernetes manifests.

---

## Repository structure
