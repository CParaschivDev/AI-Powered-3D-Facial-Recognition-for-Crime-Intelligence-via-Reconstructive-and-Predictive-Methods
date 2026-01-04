import logging
import hashlib
import time
from sqlalchemy.orm import Session
from backend.database.dependencies import SessionLocal
from backend.database.models import User
from backend.core.audit import write_event
from backend.api.models import schemas
from backend.models.utils.model_loader import get_landmark_model, get_reconstruction_model, get_recognition_model
from backend.core.utils import read_image_from_bytes
from backend.utils.watermark import embed_watermark
from backend.utils.forensic_scoring import compute_image_quality_score, compute_prediction_entropy, annotate_wanted_status
from backend.services.matcher import best_match

logger = logging.getLogger(__name__)

def annotate_matches_with_wanted(db, matches):
    """Annotate `matches` with wanted status by querying `WantedPerson`."""
    try:
        from backend.database.models import WantedPerson
        wanted_ids = {wp.person_id for wp in db.query(WantedPerson).all()}
        return annotate_wanted_status(matches, wanted_ids)
    except (ImportError, AttributeError) as e:
        logger.info("WantedPerson model not available; wanted-status lookup disabled.")
        for m in matches:
            m['wanted'] = False
        return matches

def reconstruct_face_sync(image_bytes: bytes, case_id: str, user_id: int):
    """
    Synchronous version of 3D face reconstruction.
    """
    logger.info(f"Starting synchronous reconstruction for case {case_id}")

    db = SessionLocal()
    try:
        actor = db.query(User).filter(User.id == user_id).one_or_none()
        if not actor:
            raise ValueError(f"Actor with ID {user_id} not found.")

        file_hash = hashlib.sha256(image_bytes).hexdigest()
        image = read_image_from_bytes(image_bytes)

        # 0. Compute image quality score
        quality_score = compute_image_quality_score(image)
        logger.info(f"Image quality score: {quality_score:.3f}")

        # 1. Load models
        landmark_model = get_landmark_model()
        reconstruction_model = get_reconstruction_model()
        recognition_model = get_recognition_model()

        # 2. Predict landmarks
        landmarks = landmark_model.predict(image)

        # 3. Reconstruct 3D mesh
        vertices, faces = reconstruction_model.reconstruct(image, landmarks)

        # 3.5 Generate recognition embedding and matches
        matches = []
        prediction_entropy = None
        try:
            embedding = recognition_model.extract_fused_embedding(image)
            if embedding is not None:
                pair = best_match(embedding)
                if pair:
                    best_id, similarity = pair
                    matches = [{
                        "person_id": int(best_id) if isinstance(best_id, str) and best_id.isdigit() else 0,
                        "name": f"Person_{best_id}",
                        "similarity": float(similarity),
                        "wanted": False  # Placeholder - integrate with DB
                    }]

                    # Robust wanted-status lookup
                    try:
                        matches = annotate_matches_with_wanted(db, matches)
                    except Exception as db_err:
                        logger.warning(f"Could not fetch wanted status from DB: {db_err}")

                    # Compute entropy from similarity scores
                    prediction_entropy = compute_prediction_entropy([similarity])
                    logger.info(f"Recognition: best_id={best_id}, similarity={similarity:.3f}, entropy={prediction_entropy:.3f}, wanted={matches[0]['wanted']}")
        except Exception as e:
            logger.warning(f"Recognition failed, continuing without matches: {e}")

        # 4. Embed watermark
        try:
            watermarked_vertices = embed_watermark(vertices, case_id, file_hash)
            logger.info(f"Watermark embedded successfully for case {case_id}")
        except ValueError as e:
            logger.warning(f"Could not embed watermark for case {case_id}: {e}")
            # Proceed with non-watermarked vertices if embedding fails
            watermarked_vertices = vertices

        logger.info(f"Reconstruction successful for case {case_id}")

        result_data = {
            "vertices": watermarked_vertices.tolist(),
            "faces": faces.tolist(),
            "message": "Reconstruction completed successfully.",
            "image_quality_score": quality_score,
            "prediction_entropy": prediction_entropy,
            "matches": matches
        }

        result_schema = schemas.ReconstructionResult(**result_data)

        write_event(
            db=db,
            case_id=case_id,
            actor=actor,
            action="RECONSTRUCTION_SUCCESS_SYNC",
            payload=result_schema.dict(),
            file_hash=file_hash,
        )

        return result_schema

    except Exception as e:
        logger.exception(f"An error occurred during reconstruction for case {case_id}")
        raise
    finally:
        db.close()