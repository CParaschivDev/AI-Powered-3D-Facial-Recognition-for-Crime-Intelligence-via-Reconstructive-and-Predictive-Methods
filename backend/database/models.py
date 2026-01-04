from sqlalchemy import (Boolean, Column, ForeignKey, Integer, String,
                        DateTime, LargeBinary, Text, Float, UniqueConstraint)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .db_utils import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)  # "officer", "investigator", "admin"
    disabled = Column(Boolean, default=False)


class IdentityEmbedding(Base):
    __tablename__ = "identity_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    identity_id = Column(String, unique=True, index=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # Stores encrypted embedding
    encrypted_dek = Column(LargeBinary, nullable=True)  # Encrypted data encryption key
    is_encrypted = Column(Boolean, default=False, nullable=False)  # Flag to indicate if envelope encryption is used


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, nullable=False, index=True)
    actor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False)
    payload_json = Column(String, nullable=False)
    file_hash = Column(String, nullable=True)
    prev_hash = Column(String, nullable=True)
    hash = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    actor = relationship("User")


class Evidence(Base):
    __tablename__ = "evidence"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String)
    media_type = Column(String)  # e.g., "audio/wav", "text/plain"
    evidence_type = Column(String)  # "audio", "text" 
    description = Column(Text)
    # Note: Migration renamed 'content' to 'encrypted_content', but we keep the column name as 'content'
    # to minimize code changes. This is safe as long as we don't run the migration again.
    content = Column(LargeBinary, nullable=False)  # Encrypted content
    encrypted_dek = Column(LargeBinary, nullable=True)  # Encrypted data encryption key
    is_encrypted = Column(Boolean, default=False, nullable=False)  # Flag to indicate if envelope encryption is used
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class StreamSource(Base):
    __tablename__ = "stream_sources"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    rtsp_url = Column(String, nullable=False)
    location = Column(String)
    is_active = Column(Boolean, default=True)


class Snapshot(Base):
    __tablename__ = "snapshots"
    id = Column(String, primary_key=True)  # UUID
    identity_id = Column(String, index=True)
    location = Column(String)
    timestamp = Column(DateTime(timezone=True))
    encrypted_image_data = Column(LargeBinary, nullable=False)  # Encrypted image data
    encrypted_dek = Column(LargeBinary, nullable=True)  # Encrypted data encryption key
    is_encrypted = Column(Boolean, default=False, nullable=False)  # Flag to indicate if envelope encryption is used


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    area_id = Column(String, nullable=False, index=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    crime_type = Column(String, nullable=False, index=True)
    yhat = Column(Float, nullable=False)
    yhat_lower = Column(Float, nullable=False)
    yhat_upper = Column(Float, nullable=False)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True) # e.g., 'recognition', 'reconstruction'
    version = Column(Integer, nullable=False)
    path = Column(String, nullable=False)
    sha256 = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    active = Column(Boolean, default=False, nullable=False)
    training_output_path = Column(String, nullable=True)  # New column to store path to training outputs

    __table_args__ = (UniqueConstraint('name', 'version', name='uq_model_name_version'),)


class PredictiveLog(Base):
    """Bias monitoring and predictive policing audit log"""
    __tablename__ = "predictive_logs"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    prediction_type = Column(String, nullable=False)  # e.g., 'bias_audit', 'hotspot', 'reappearance'
    prediction_data = Column(Text, nullable=False)  # JSON string with full prediction/audit data
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    bias_score = Column(Float, nullable=True)
    is_flagged = Column(Boolean, default=False, nullable=False)
    auditor_id = Column(String, nullable=True)
    audit_status = Column(String, nullable=True)  # 'pending', 'approved', 'rejected'
    audit_notes = Column(Text, nullable=True)


class DPIAAssessment(Base):
    """Data Protection Impact Assessment records"""
    __tablename__ = "dpia_assessments"
    id = Column(Integer, primary_key=True, index=True)
    assessment_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    assessment_version = Column(String, nullable=False)  # e.g., "1.0", "1.1"
    
    # Risk scoring
    risk_level = Column(String, nullable=False)  # 'LOW', 'MEDIUM', 'HIGH'
    overall_risk_score = Column(Float, nullable=False)  # 0.0 - 1.0
    privacy_risks = Column(Text, nullable=False)  # JSON array of identified risks
    
    # Compliance checks
    gdpr_compliant = Column(Boolean, default=False, nullable=False)
    uk_dpa_compliant = Column(Boolean, default=False, nullable=False)
    bias_threshold_met = Column(Boolean, default=False, nullable=False)
    encryption_verified = Column(Boolean, default=False, nullable=False)
    audit_logs_enabled = Column(Boolean, default=False, nullable=False)
    
    # Approval workflow
    approval_status = Column(String, nullable=False, default='pending')  # 'pending', 'approved', 'rejected'
    approved_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    approval_date = Column(DateTime(timezone=True), nullable=True)
    valid_until = Column(DateTime(timezone=True), nullable=True)  # Expiry date for re-assessment
    
    # Documentation
    assessment_notes = Column(Text, nullable=True)
    mitigation_measures = Column(Text, nullable=True)  # JSON array of implemented mitigations
    recommendations = Column(Text, nullable=True)
    
    # Relationships
    approved_by = relationship("User", foreign_keys=[approved_by_id])
    
    # Metadata
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_by = relationship("User", foreign_keys=[created_by_id])
