import logging
import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
 
from backend.api.routes import reconstruct, recognize, auth, report, multimodal, analytics, evidence, bias, dpia
from backend.api.routes import crime
from backend.core.config import settings
from backend.database.db_utils import init_db

# Initialize FastAPI with project settings
app = FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")

# Serve saliency maps and other evidence files
import os
evidence_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "evidence")
os.makedirs(evidence_dir, exist_ok=True)
app.mount("/evidence", StaticFiles(directory=evidence_dir), name="evidence")
 
logger = logging.getLogger(__name__) 

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Running database migrations...")
        init_db()
        logger.info("✅ Startup completed successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Root endpoint - API health check"""
    return {
        "status": "online",
        "service": "AI-Powered 3D Facial Recognition Framework",
        "version": "1.0.0",
        "docs": "/docs",
        "api": settings.API_V1_STR
    }

app.include_router(auth.router, prefix=settings.API_V1_STR, tags=["Authentication"])
app.include_router(reconstruct.router, prefix=settings.API_V1_STR, tags=["3D Reconstruction"])
app.include_router(recognize.router, prefix=settings.API_V1_STR, tags=["Recognition"])
app.include_router(report.router, prefix=settings.API_V1_STR, tags=["Reporting"])
app.include_router(multimodal.router, prefix=settings.API_V1_STR, tags=["Multimodal Evidence"])
app.include_router(evidence.router, prefix=settings.API_V1_STR, tags=["Evidence"])
app.include_router(bias.router, prefix=settings.API_V1_STR, tags=["Bias Monitoring"])
app.include_router(dpia.router, prefix=settings.API_V1_STR, tags=["DPIA Compliance"])
app.include_router(analytics.router, prefix=settings.API_V1_STR, tags=["Analytics"])
app.include_router(crime.router, prefix=settings.API_V1_STR, tags=["Crime Analytics"])

