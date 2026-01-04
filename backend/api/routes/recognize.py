import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from sqlalchemy.orm import Session
import numpy as np
from datetime import datetime
import uuid
import hashlib
import os
import cv2

from backend.api.models import schemas
from backend.models.utils.model_loader import get_recognition_model, get_reconstruction_model, get_landmark_model
from backend.database.db_utils import store_snapshot
from backend.utils.augmentation import assess_image_quality
from backend.services.matcher import best_match
from backend.core.notifications import send_match_notification_email
from backend.core.audit import write_event
# from backend.api.routes.streams import manager # For WebSocket broadcast
from backend.core.dependencies import get_current_user_optional
from backend.core.utils import read_image_from_bytes
from backend.database.dependencies import get_db
from backend.database.models import User as DbUser
from backend.models.reconstruction.reconstruct import overlay_on_rgb112
from backend.core.paths import REPORTS_PATH

router = APIRouter()
logger = logging.getLogger(__name__)

HIGH_CONFIDENCE_THRESHOLD = 0.85
# Safety limit for uploaded images
MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MB

@router.post(
    "/recognize", 
    response_model=schemas.SimpleRecognitionResult,
    summary="Recognize identity from image with YOLO object detection (DEMO - No Auth Required)",
    description="Recognizes identities from an image using YOLO object detection followed by facial recognition. Authentication disabled for demo purposes."
)
async def recognize_identity(
    file: UploadFile = File(...),
    case_id: str = Form(...),
    db: Session = Depends(get_db),
    current_user: DbUser = Depends(get_current_user_optional),
    location: str = Form("Unknown Location"),
    overlay: bool = False, # New query parameter
    yolo_conf_threshold: float = Form(0.1, description="YOLO confidence threshold for object detection")
):
    """
    Recognizes identities from an uploaded image using integrated YOLO + facial recognition.

    This endpoint performs the following steps:
    1. Receives an image file.
    2. Uses YOLO to detect persons and objects in the image.
    3. For each detected person, performs facial recognition.
    4. Queries the database to find the top N closest matches based on cosine similarity.
    5. Returns a ranked list of potential identities with YOLO detection info.
    6. If a high-confidence match is found, sends an email alert.
    """
    logger.info(f"Recognition request received for file: {file.filename} from {location}")
    
    try:
        from backend.core.file_utils import safe_read_upload_bytes
        from backend.services.crime_scene_processor import process_frame

        try:
            image_bytes = await safe_read_upload_bytes(file)
        except ValueError:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")
        file_hash = hashlib.sha256(image_bytes).hexdigest()
        image = read_image_from_bytes(image_bytes)
        quality_score = assess_image_quality(image)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read or process image file: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Use crime scene processor for integrated YOLO + recognition
    logger.info(f"Processing image with YOLO (threshold: {yolo_conf_threshold}) + facial recognition")
    try:
        result = process_frame(
            image, 
            frame_id=file_hash,
            camera_id='upload',
            location_id=location,
            yolo_conf_threshold=yolo_conf_threshold
        )
        logger.info(f"process_frame returned: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'not dict'}")
    except Exception as e:
        logger.error(f"Failed to process frame with YOLO + recognition: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    overlay_path = None
    if overlay and result.get('persons'):
        try:
            reconstruction_model = get_reconstruction_model()
            landmark_model = get_landmark_model()
            # Use the first detected person for overlay
            person = result['persons'][0]
            if 'face_crop' in person:
                face_crop = person['face_crop']
                if face_crop.shape[0] != 112 or face_crop.shape[1] != 112:
                    img112_rgb = cv2.resize(face_crop, (112, 112))
                else:
                    img112_rgb = face_crop

                overlay_img = overlay_on_rgb112(img112_rgb, reconstruction_model, landmark_model)
                
                overlays_dir = os.path.join(REPORTS_PATH, "overlays")
                os.makedirs(overlays_dir, exist_ok=True)
                
                overlay_filename = f"{uuid.uuid4()}.png"
                overlay_full_path = os.path.join(overlays_dir, overlay_filename)
                cv2.imwrite(overlay_full_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
                overlay_path = f"reports/overlays/{overlay_filename}"
                logger.info(f"Overlay image saved to {overlay_path}")
        except Exception as e:
            logger.warning(f"Could not generate 3D overlay: {e}")
            # Don't break the route if the model isn't there; return an optional overlay path.
            overlay_path = None

    # Format the result from crime_scene_processor
    try:
        # Extract recognition results from YOLO + recognition processing
        persons = result.get('persons', [])
        weapons = result.get('weapons', [])
        
        logger.info(f"YOLO + Recognition completed: {len(persons)} persons detected with recognition, {len(weapons)} weapons detected")
        
        # Find the best recognition match across all detected persons
        best_match_result = None
        best_score = -1
        
        for person in persons:
            if 'recognition' in person and person['recognition'].get('pred_class') is not None:
                score = person['recognition']['pred_conf']
                if score > best_score:
                    best_score = score
                    best_match_result = person
        
        if best_match_result:
            best_id = best_match_result['recognition']['pred_class']
            verdict = "WANTED" if best_score >= HIGH_CONFIDENCE_THRESHOLD else "NOT WANTED"
            
            # Store snapshot for audit trail
            import uuid
            timestamp = datetime.utcnow()
            snapshot_id = str(uuid.uuid4())
            store_snapshot(
                db=db,
                snapshot_id=snapshot_id,
                identity_id=str(best_id),
                location=location,
                image_data=image_bytes
            )
            
            # Send email alert for high-confidence wanted matches
            if verdict == "WANTED" and best_score >= HIGH_CONFIDENCE_THRESHOLD:
                try:
                    send_match_notification_email(
                        suspect_id=best_id,
                        confidence=best_score,
                        location=location,
                        case_id=case_id,
                        image_hash=file_hash
                    )
                except Exception as email_e:
                    logger.warning(f"Failed to send email notification: {email_e}")
            
            result_data = schemas.SimpleRecognitionResult(
                best_id=str(best_id), 
                cosine_score=best_score, 
                threshold=HIGH_CONFIDENCE_THRESHOLD, 
                verdict=verdict, 
                overlay_path=overlay_path,
                yolo_results={
                    'persons': persons,
                    'weapons': weapons,
                    'total_detections': len(persons) + len(weapons)
                }
            )
        else:
            # No persons detected with recognition results
            result_data = schemas.SimpleRecognitionResult(
                best_id="",
                cosine_score=0.0,
                threshold=HIGH_CONFIDENCE_THRESHOLD,
                verdict="NO FACES DETECTED",
                overlay_path=overlay_path,
                yolo_results={
                    'persons': persons,
                    'weapons': weapons,
                    'total_detections': len(persons) + len(weapons)
                }
            )
        
        # Only write audit event if user is authenticated
        if current_user:
            write_event(
                db=db,
                case_id=case_id,
                actor=current_user,
                action="RECOGNITION_SUCCESS",
                payload=result_data.dict(),
                file_hash=file_hash,
            )

        return result_data
    except Exception as e:
        logger.exception("An error occurred during the recognition process.")
        raise HTTPException(status_code=500, detail="Internal server error during recognition.")




