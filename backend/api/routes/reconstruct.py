import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, status

from backend.api.models import schemas
from backend.core.audit import write_event

router = APIRouter()
logger = logging.getLogger(__name__)

from sqlalchemy.orm import Session
from backend.database.dependencies import get_db
from backend.database.models import User as DbUser

# Safety limit for uploaded images
MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MB

@router.post(
    "/reconstruct", 
    response_model=schemas.ReconstructionResult,
    status_code=status.HTTP_200_OK,
    summary="Reconstruct 3D face from image (OFFICER, INVESTIGATOR, or ADMIN)",
    description="Uploads an image for 3D face reconstruction. Returns the reconstruction result directly."
)
async def upload_image_for_reconstruction(
    file: UploadFile = File(...),
    case_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Uploads an image for 3D face reconstruction and returns the result synchronously.
    """
    logger.info(f"Reconstruction request received for file: {file.filename}")
 
    try:
        from backend.core.file_utils import safe_read_upload_bytes

        try:
            image_bytes = await safe_read_upload_bytes(file)
        except ValueError:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read image file: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file.")
 
    # Perform reconstruction synchronously
    try:
        from backend.services.reconstruction_service import reconstruct_face_sync
        
        result = reconstruct_face_sync(image_bytes, case_id, 1)  # Default user_id
        return result
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")




