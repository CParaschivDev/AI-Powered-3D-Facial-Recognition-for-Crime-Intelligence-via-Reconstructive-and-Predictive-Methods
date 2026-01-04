from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Any, Dict
from datetime import datetime


class Token(BaseModel):
    """
    Schema for the authentication token.
    """
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """
    Schema for the data encoded in the token.
    """
    username: Optional[str] = None
    scopes: List[str] = []


class User(BaseModel):
    username: str
    role: Literal["officer", "investigator", "admin"]

    model_config = ConfigDict(from_attributes=True)


class WatermarkVerificationResponse(BaseModel):
    """
    Schema for the response of a watermark verification.
    """
    is_valid: bool
    message: str


class TaskLaunchResponse(BaseModel):
    """
    Schema for the response when a background task is launched.
    """
    task_id: str


class ProgressMessage(BaseModel):
    """
    Schema for a progress update message sent via WebSocket.
    """
    task_id: str
    status: str # e.g., 'PROGRESS', 'SUCCESS', 'FAILURE'
    progress: int # 0-100
    details: str


# --- Analytics Schemas ---

class PredictionPoint(BaseModel):
    """
    Schema for a single prediction data point in a time series.
    """
    ts: datetime
    crime_type: str
    yhat: float
    yhat_lower: float
    yhat_upper: float

    model_config = ConfigDict(from_attributes=True)


class PredictionResponse(BaseModel):
    """
    Schema for returning a list of time series predictions.
    """
    predictions: List[PredictionPoint]


class ReconstructionResult(BaseModel):
    """
    Schema for the 3D reconstruction result.
    GEN-4 enhanced with forensic analysis.
    """
    vertices: List[List[float]]
    faces: List[List[int]]
    message: str
    image_quality_score: Optional[float] = Field(None, description="Image quality score (0-1)")
    prediction_entropy: Optional[float] = Field(None, description="Recognition confidence entropy (0-1)")
    matches: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Recognition matches with wanted status")


class ProgressMessage(BaseModel):
    """
    Schema for the result of an image upload, containing the image ID.
    """
    image_id: str
    message: str

class RecognitionMatch(BaseModel):
    """
    Schema for a single recognition match.
    """
    identity_id: str
    confidence: float
    saliency_url: Optional[str] = None

class RecognitionResult(BaseModel):
    """
    Schema for the full recognition result.
    """
    matches: List[RecognitionMatch]
    location: Optional[str] = None
    timestamp: Optional[str] = None
    snapshot_id: Optional[str] = None
    quality_score: Optional[float] = Field(None, description="Quality score of the input image (0-1).")
    confidence_scores: Optional[List[float]] = Field(None, description="Softmax-normalized confidence scores for top matches.")
    entropy: Optional[float] = Field(None, description="Entropy of the confidence scores.")
    overlay_path: Optional[str] = Field(None, description="Path to the 3D face overlay image if requested.")


class EvidenceUploadResult(BaseModel):
    """
    Schema for the result of multimodal evidence upload.
    """
    evidence_id: str
    message: str


class ReportRequest(BaseModel):
    """
    Schema for requesting a generated report.
    """
    case_id: str
    case_context: Optional[str] = "No additional context provided."
    # Gen5 additions
    include_risk_assessment: bool = True
    include_predicted_hotspots: bool = True
    include_ethical_concerns: bool = True


class StreamInput(BaseModel):
    """
    Schema for registering a new video stream.
    """
    name: str
    rtsp_url: str
    location: str
    is_active: bool = True


class StreamOutput(StreamInput):
    """
    Schema for returning stream information.
    """
    id: int

    model_config = ConfigDict(from_attributes=True)


class MatchAlert(BaseModel):
    """
    Schema for a live match alert sent via WebSocket.
    """
    identity_id: str
    confidence: float
    location: str
    timestamp: str
    snapshot_url: str


# --- Gen5 Schemas ---

class PredictiveHotspot(BaseModel):
    """
    Schema for a single predictive crime hotspot.
    """
    latitude: float
    longitude: float
    intensity: float
    predicted_at: str

class PredictiveRequest(BaseModel):
    """
    Schema for requesting a prediction.
    """
    area_boundary: Dict[str, float] # e.g., {"lat_min": ..., "lon_max": ...}
    time_horizon_hours: int = 24

class BiasAuditDecision(BaseModel):
    """
    Schema for an officer's decision on a flagged prediction.
    """
    prediction_log_id: int
    decision: Literal["approve", "reject"]
    notes: Optional[str] = None


class Case(BaseModel):
    """
    Schema for a full investigative case.
    """
    id: str
    case_number: str
    title: str
    description: Optional[str]
    status: str
    created_at: str
    updated_at: str

    model_config = ConfigDict(from_attributes=True)


class AgentSuggestion(BaseModel):
    """
    Schema for a suggestion from an AI agent.
    """
    agent_name: str # e.g., "Case Analyst Agent", "Risk Assessment Agent"
    suggestion_type: str # e.g., "LINK_EVIDENCE", "NEXT_STEP", "THREAT_LEVEL_UPDATE"
    suggestion_details: Dict[str, Any]
    confidence: float
    explanation: str # XAI explanation
    requires_approval: bool = True


class EvidenceVerificationRequest(BaseModel):
    """
    Schema for requesting verification of an evidence hash.
    """
    evidence_hash: str


class AuditLogEntry(BaseModel):
    """
    Schema for a single entry in the immutable audit trail.
    """
    id: int
    case_id: str
    actor_id: int
    action: str
    payload_json: str
    file_hash: Optional[str]
    prev_hash: Optional[str]
    hash: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- Model Registry Schemas ---

class ModelVersion(BaseModel):
    id: int
    name: str
    version: int
    path: str
    sha256: str
    created_at: datetime
    active: bool
    training_output_path: Optional[str] = None  # Path to training outputs (logs, metrics, etc.)

    model_config = ConfigDict(from_attributes=True)

class ModelActivationRequest(BaseModel):
    name: str
    version: int
    
class SimpleRecognitionResult(BaseModel):
    """
    Simplified schema for recognition result with best match only.
    GEN-4 enhanced with forensic scoring.
    """
    best_id: str
    cosine_score: float
    threshold: float
    verdict: str
    overlay_path: Optional[str] = None
    quality_score: Optional[float] = Field(None, description="Image quality score (0-1, higher is better)")
    entropy: Optional[float] = Field(None, description="Prediction entropy (0-1, lower is more confident)")
    wanted: Optional[bool] = Field(None, description="Whether the matched person is wanted")
    yolo_results: Optional[dict] = Field(None, description="YOLO object detection results")
    
class GenericOk(BaseModel):
    """
    Generic success response with optional details.
    """
    ok: bool
    details: Any = None


class EvidenceUploadResult(BaseModel):
    """
    Schema for the result of uploading evidence.
    """
    evidence_id: str
    message: str


class ReportRequest(BaseModel):
    """
    Schema for requesting an AI-generated investigation report.
    """
    case_id: str
    case_context: str
    include_risk_assessment: bool = False
    include_predicted_hotspots: bool = False
    include_ethical_concerns: bool = False
    matches: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_ids: List[int] = Field(default_factory=list)
    vertices: List[List[float]] = Field(default_factory=list)
    faces: List[List[int]] = Field(default_factory=list)
