#!/usr/bin/env python3
"""
Minimal test to isolate server startup issues
"""
import logging
import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from backend.api.main import app
    logger.info("✅ App imported successfully")

    # Try to test the app without running the server
    from fastapi.testclient import TestClient
    client = TestClient(app)
    logger.info("✅ TestClient created successfully")

    # Try a simple request
    response = client.get("/api/v1/models")
    logger.info(f"✅ Test request successful: {response.status_code}")

except Exception as e:
    logger.error(f"❌ Error: {str(e)}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)