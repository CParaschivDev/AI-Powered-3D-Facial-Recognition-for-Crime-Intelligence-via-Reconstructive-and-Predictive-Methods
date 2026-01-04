import logging
from fastapi import APIRouter, UploadFile, File, Depends, Form, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session
from typing import Optional

from backend.api.models import schemas
from backend.database.dependencies import get_db
from backend.database.db_utils import store_evidence, get_evidence, get_all_evidence, delete_evidence

router = APIRouter()
logger = logging.getLogger(__name__)

# Safety limit for evidence uploads
MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/evidence",
    response_model=schemas.EvidenceUploadResult,
)
async def upload_evidence(
    file: UploadFile = File(...),
    evidence_type: str = Form(...),  # "audio" or "text"
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Uploads multimodal evidence, such as audio from emergency calls or text from police reports.

    The evidence is stored, and a background task is triggered to extract features
    and correlate it with other evidence.
    """
    logger.info(
        f"Evidence upload received: {file.filename}, type: {evidence_type}"
    )

    from backend.core.file_utils import safe_read_upload_bytes
    try:
        content = await safe_read_upload_bytes(file)
    except ValueError:
        raise HTTPException(status_code=413, detail="Uploaded file is too large")

    evidence_id = store_evidence(
        db,
        file_content=content,
        file_name=file.filename,
        media_type=file.content_type,
        evidence_type=evidence_type,
        description=description,
    )

    # In a real system, you would trigger a background task here for processing.
    # e.g., process_multimodal_evidence.delay(evidence_id)

    return schemas.EvidenceUploadResult(
        evidence_id=str(evidence_id),
        message=f"{evidence_type.capitalize()} evidence uploaded and stored securely.",
    )


@router.get("/evidence")
async def list_evidence(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    Lists all stored evidence records.
    
    Returns metadata about each evidence item without the actual file content.
    """
    evidence_list = get_all_evidence(db, skip=skip, limit=limit)
    
    # Convert to response format
    evidence_items = []
    for evidence in evidence_list:
        evidence_items.append({
            "id": evidence.id,
            "file_name": evidence.file_name,
            "media_type": evidence.media_type,
            "evidence_type": evidence.evidence_type,
            "description": evidence.description or "No description",
            "created_at": evidence.created_at.isoformat() if evidence.created_at else None,
            "file_size": len(evidence.content) if evidence.content else 0
        })
    
    return {
        "total": len(evidence_items),
        "skip": skip,
        "limit": limit,
        "evidence": evidence_items
    }


@router.post("/evidence/verify-watermark")
async def verify_evidence_watermark(
    file: UploadFile = File(...),
    case_id: str = Form(...),
    file_hash: str = Form(...),
):
    """
    Verifies the cryptographic watermark in a 3D model evidence file.
    
    This endpoint checks if the uploaded 3D model contains a valid watermark
    that matches the provided case ID and file hash, ensuring evidence integrity.
    """
    logger.info(f"Watermark verification request for file: {file.filename}")
    
    try:
        # Read the uploaded file content
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Parse OBJ file to extract vertices
        import numpy as np
        vertices = []
        for line in content_str.split('\n'):
            if line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
        
        if not vertices:
            raise HTTPException(status_code=400, detail="No vertices found in OBJ file")
        
        vertices_array = np.array(vertices, dtype=np.float32)
        
        # Import and use watermark verification
        from backend.utils.watermark import verify_watermark
        is_valid, message = verify_watermark(vertices_array, case_id, file_hash)
        
        return {
            "is_valid": is_valid,
            "message": message,
            "case_id": case_id,
            "file_hash": file_hash
        }
        
    except Exception as e:
        logger.error(f"Watermark verification failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Verification failed: {str(e)}")


@router.get("/evidence/{evidence_id}")
async def download_evidence_file(
    evidence_id: int,
    db: Session = Depends(get_db),
):
    """
    Downloads a specific evidence file by ID.
    
    Returns the decrypted file content with appropriate headers for download.
    """
    logger.info(f"Evidence download request for ID: {evidence_id}")
    
    # Get evidence from database
    evidence, decrypted_content = get_evidence(db, evidence_id)
    
    if not evidence:
        raise HTTPException(status_code=404, detail=f"Evidence with ID {evidence_id} not found")
    
    if decrypted_content is None:
        logger.error(f"Failed to decrypt evidence {evidence_id}")
        raise HTTPException(status_code=500, detail="Failed to decrypt evidence file")
    
    # Return file with appropriate headers
    return Response(
        content=decrypted_content,
        media_type=evidence.media_type,
        headers={
            "Content-Disposition": f"attachment; filename={evidence.file_name}",
            "Content-Length": str(len(decrypted_content)),
        }
    )


@router.delete("/evidence/{evidence_id}")
async def delete_evidence_endpoint(
    evidence_id: int,
    db: Session = Depends(get_db)
):
    """
    Deletes a specific evidence file by ID.
    """
    logger.info(f"Evidence deletion request for ID: {evidence_id}")
    
    # Delete evidence from database
    success = delete_evidence(db, evidence_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Evidence with ID {evidence_id} not found")
    
    return {"message": f"Evidence {evidence_id} deleted successfully"}




