import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
import numpy as np

from backend.api.models import schemas
from backend.utils.watermark import verify_watermark

router = APIRouter()
logger = logging.getLogger(__name__)

def parse_obj_file(file_content: bytes) -> np.ndarray:
    """A simple parser to extract vertices from an OBJ file."""
    vertices = []
    try:
        # Guard against extremely large uploads
        MAX_OBJ_VERTICES = 500000  # safety limit for vertices
        MAX_OBJ_BYTES = 10 * 1024 * 1024  # 10 MB
        if len(file_content) > MAX_OBJ_BYTES:
            raise ValueError("Uploaded OBJ file too large")
        lines = file_content.decode('utf-8', errors='ignore').splitlines()
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(vertices) > MAX_OBJ_VERTICES:
                    raise ValueError("OBJ file contains too many vertices")
        return np.array(vertices, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to parse OBJ file: {e}")
        raise HTTPException(status_code=400, detail=str(e) or "Could not parse the provided OBJ file.")

@router.post(
    "/evidence/verify",
    response_model=schemas.WatermarkVerificationResponse,
    summary="Verify watermark in a 3D model file"
)
async def verify_model_watermark(
    case_id: str = Form(...),
    original_file_hash: str = Form(...),
    file: UploadFile = File(...),
):
    # Use centralized safe reader to bound upload sizes
    from backend.core.file_utils import safe_read_upload_bytes
    try:
        file_content = await safe_read_upload_bytes(file)
    except ValueError:
        raise HTTPException(status_code=413, detail="Uploaded file is too large")

    # parse obj/ply etc
    if file.filename.lower().endswith('.obj'):
        vertices = parse_obj_file(file_content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an .obj file.")

    if vertices.size == 0:
        raise HTTPException(status_code=400, detail="No vertices found in the provided file.")

    is_valid, message = verify_watermark(vertices, case_id, original_file_hash)
    return schemas.WatermarkVerificationResponse(is_valid=is_valid, message=message)




