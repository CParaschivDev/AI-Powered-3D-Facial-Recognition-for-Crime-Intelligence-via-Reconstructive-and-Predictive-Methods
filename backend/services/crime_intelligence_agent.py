# crime_intelligence_agent.py
"""
Orchestration agent for AI-powered 3D facial recognition and crime-intelligence system.
This agent coordinates YOLO object detection, facial recognition, and crime intelligence storage.
"""
import cv2
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from backend.services.crime_scene_processor import process_frame

class CrimeIntelligenceAgent:
    """
    Orchestration agent for processing CCTV frames and producing crime-intelligence records.
    """

    def __init__(self):
        # For now, store events in memory (can be replaced with database later)
        self.events = []

    def detect_objects(self, frame) -> List[Dict[str, Any]]:
        """
        Detect objects in a frame using YOLO.

        Args:
            frame: np.ndarray (BGR image from cv2)

        Returns:
            List of detections with label, confidence, and bbox
        """
        from backend.services.yolo_detector import YoloWeaponDetector

        detector = YoloWeaponDetector()
        return detector.detect(frame)

    def recognise_face(self, face_image) -> Dict[str, Any]:
        """
        Recognize a face from an image crop.

        Args:
            face_image: np.ndarray (BGR face crop)

        Returns:
            Dict with identity_prediction, confidence, and embedding
        """
        from backend.services.crime_scene_processor import recognise_face_from_person_crop
        return recognise_face_from_person_crop(face_image)

    def store_event(self, record: Dict[str, Any]) -> bool:
        """
        Store a processed crime intelligence record.

        Args:
            record: JSON-like record from process_frame

        Returns:
            Success status
        """
        try:
            # For now, store in memory
            self.events.append(record)
            return True
        except Exception as e:
            print(f"Error storing event: {e}")
            return False

    def process_cctv_frame(self,
                          frame_bgr,
                          frame_id: Optional[int] = None,
                          camera_id: Optional[str] = None,
                          timestamp: Optional[str] = None,
                          location_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main orchestration method: Process a CCTV frame end-to-end.

        Args:
            frame_bgr: np.ndarray (BGR image)
            frame_id: Optional frame identifier
            camera_id: Camera identifier
            timestamp: ISO timestamp string
            location_id: Location identifier

        Returns:
            Processed crime intelligence record
        """
        # Step 1: Process frame with YOLO + recognition
        record = process_frame(
            frame_bgr=frame_bgr,
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=timestamp,
            location_id=location_id
        )

        # Step 2: Store the record
        success = self.store_event(record)
        record["storage_success"] = success

        return record

    def query_events(self,
                    camera_id: Optional[str] = None,
                    location_id: Optional[str] = None,
                    start_time: Optional[str] = None,
                    end_time: Optional[str] = None,
                    has_weapons: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Query stored crime intelligence events.

        Args:
            camera_id: Filter by camera
            location_id: Filter by location
            start_time: Filter events after this time
            end_time: Filter events before this time
            has_weapons: Filter events with/without weapons

        Returns:
            List of matching events
        """
        filtered_events = self.events.copy()

        if camera_id:
            filtered_events = [e for e in filtered_events if e.get("camera_id") == camera_id]
        if location_id:
            filtered_events = [e for e in filtered_events if e.get("location_id") == location_id]
        if start_time:
            filtered_events = [e for e in filtered_events if e.get("timestamp", "") >= start_time]
        if end_time:
            filtered_events = [e for e in filtered_events if e.get("timestamp", "") <= end_time]
        if has_weapons is not None:
            if has_weapons:
                filtered_events = [e for e in filtered_events if e.get("weapons", [])]
            else:
                filtered_events = [e for e in filtered_events if not e.get("weapons", [])]

        return filtered_events

    def get_known_offenders(self) -> List[Dict[str, Any]]:
        """
        Get list of known offenders from the watchlist.

        Returns:
            List of offender profiles with embeddings
        """
        # Placeholder - would integrate with actual watchlist service
        return []

    def match_against_watchlist(self, embedding: List[float], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Match a face embedding against the known offenders watchlist.

        Args:
            embedding: Face embedding vector
            threshold: Similarity threshold

        Returns:
            List of matching offenders with similarity scores
        """
        # Placeholder - would integrate with actual matching service
        return []