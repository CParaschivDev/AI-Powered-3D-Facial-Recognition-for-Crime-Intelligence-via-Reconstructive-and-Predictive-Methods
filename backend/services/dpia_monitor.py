"""
DPIA (Data Protection Impact Assessment) Monitoring Service

Automated compliance checking and risk assessment for biometric data processing.
Enforces GDPR Article 9, UK DPA 2018, and ethical AI principles.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database.models import DPIAAssessment, User, PredictiveLog

logger = logging.getLogger(__name__)


class DPIAMonitor:
    """
    Monitors and enforces Data Protection Impact Assessment compliance.
    """
    
    # Risk severity weights (0.0 - 1.0)
    RISK_WEIGHTS = {
        "unauthorized_access": 0.25,
        "algorithmic_bias": 0.25,
        "function_creep": 0.15,
        "data_breach": 0.20,
        "inaccurate_matches": 0.10,
        "lack_transparency": 0.05
    }
    
    # Compliance thresholds
    BIAS_THRESHOLD = 0.05  # 5% maximum accuracy gap
    MAX_RISK_SCORE = 0.60  # Overall risk must be below 60% for approval
    REASSESSMENT_DAYS = 90  # Re-assess every 90 days
    
    def __init__(self, db: Session):
        """
        Initialize DPIA monitor.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
    
    def compute_risk_score(
        self,
        bias_score: Optional[float] = None,
        encryption_enabled: bool = True,
        audit_logs_enabled: bool = True,
        rbac_enabled: bool = True,
        recent_breaches: int = 0
    ) -> Tuple[float, Dict[str, float], str]:
        """
        Compute overall DPIA risk score based on system state.
        
        Args:
            bias_score: Latest bias monitoring score (None if not available)
            encryption_enabled: Whether AES-256-GCM encryption is active
            audit_logs_enabled: Whether hash-chained audit logs are enabled
            rbac_enabled: Whether role-based access control is enforced
            recent_breaches: Number of security incidents in last 90 days
            
        Returns:
            Tuple of (overall_risk_score, risk_breakdown, risk_level)
        """
        risk_breakdown = {}
        
        # 1. Unauthorized Access Risk
        access_risk = 0.0
        if not encryption_enabled:
            access_risk += 0.5
        if not rbac_enabled:
            access_risk += 0.3
        if recent_breaches > 0:
            access_risk += min(0.2 * recent_breaches, 0.4)
        risk_breakdown["unauthorized_access"] = min(access_risk, 1.0)
        
        # 2. Algorithmic Bias Risk
        if bias_score is not None:
            # Scale bias score: 0% bias = 0.0 risk, 10% bias = 1.0 risk
            bias_risk = min(bias_score / 0.10, 1.0)
        else:
            # No bias monitoring = high risk
            bias_risk = 0.8
        risk_breakdown["algorithmic_bias"] = bias_risk
        
        # 3. Function Creep (Scope Expansion) Risk
        # Low if purpose limitation enforced (simulated - would check policy docs)
        risk_breakdown["function_creep"] = 0.2 if audit_logs_enabled else 0.6
        
        # 4. Data Breach Risk
        breach_risk = 0.0
        if not encryption_enabled:
            breach_risk += 0.6
        if recent_breaches > 0:
            breach_risk += min(0.3 * recent_breaches, 0.4)
        risk_breakdown["data_breach"] = min(breach_risk, 1.0)
        
        # 5. Inaccurate Matches Risk
        # Assumes human-in-the-loop verification (would check from config)
        risk_breakdown["inaccurate_matches"] = 0.3  # Moderate risk even with verification
        
        # 6. Lack of Transparency Risk
        transparency_risk = 0.0
        if not audit_logs_enabled:
            transparency_risk += 0.5
        # Would also check for explainability features (saliency maps, confidence scores)
        risk_breakdown["lack_transparency"] = transparency_risk
        
        # Compute weighted overall score
        overall_score = sum(
            risk_breakdown[risk] * self.RISK_WEIGHTS[risk]
            for risk in self.RISK_WEIGHTS.keys()
        )
        
        # Determine risk level
        if overall_score < 0.30:
            risk_level = "LOW"
        elif overall_score < 0.60:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return overall_score, risk_breakdown, risk_level
    
    def check_compliance(
        self,
        bias_score: Optional[float] = None,
        encryption_enabled: bool = True,
        audit_logs_enabled: bool = True
    ) -> Dict[str, bool]:
        """
        Check compliance with GDPR, UK DPA 2018, and internal policies.
        
        Args:
            bias_score: Latest bias score from bias monitoring
            encryption_enabled: Whether encryption is active
            audit_logs_enabled: Whether audit logs are enabled
            
        Returns:
            Dictionary of compliance checks
        """
        compliance = {
            "gdpr_article_9": encryption_enabled and audit_logs_enabled,  # Special category data safeguards
            "uk_dpa_part_3": audit_logs_enabled,  # Law enforcement processing requires audit trail
            "bias_threshold": bias_score is not None and bias_score < self.BIAS_THRESHOLD if bias_score else False,
            "encryption": encryption_enabled,
            "audit_logs": audit_logs_enabled,
        }
        
        # Overall GDPR/UK DPA compliance
        compliance["gdpr_compliant"] = compliance["gdpr_article_9"] and compliance["bias_threshold"]
        compliance["uk_dpa_compliant"] = compliance["uk_dpa_part_3"] and compliance["encryption"]
        
        return compliance
    
    def create_assessment(
        self,
        created_by: User,
        assessment_version: str = "1.0",
        bias_score: Optional[float] = None,
        encryption_enabled: bool = True,
        audit_logs_enabled: bool = True,
        rbac_enabled: bool = True,
        recent_breaches: int = 0,
        notes: Optional[str] = None
    ) -> DPIAAssessment:
        """
        Create a new DPIA assessment record.
        
        Args:
            created_by: User creating the assessment
            assessment_version: Version identifier (e.g., "1.0", "2.1")
            bias_score: Latest bias monitoring score
            encryption_enabled: Encryption status
            audit_logs_enabled: Audit log status
            rbac_enabled: RBAC status
            recent_breaches: Number of recent security incidents
            notes: Additional assessment notes
            
        Returns:
            Created DPIAAssessment object
        """
        # Compute risk score
        overall_risk, risk_breakdown, risk_level = self.compute_risk_score(
            bias_score=bias_score,
            encryption_enabled=encryption_enabled,
            audit_logs_enabled=audit_logs_enabled,
            rbac_enabled=rbac_enabled,
            recent_breaches=recent_breaches
        )
        
        # Check compliance
        compliance = self.check_compliance(
            bias_score=bias_score,
            encryption_enabled=encryption_enabled,
            audit_logs_enabled=audit_logs_enabled
        )
        
        # Build mitigation measures list
        mitigations = []
        if encryption_enabled:
            mitigations.append({
                "measure": "AES-256-GCM Envelope Encryption",
                "status": "implemented",
                "effectiveness": "high"
            })
        if audit_logs_enabled:
            mitigations.append({
                "measure": "SHA-256 Hash-Chained Audit Logs",
                "status": "implemented",
                "effectiveness": "high"
            })
        if bias_score is not None and bias_score < self.BIAS_THRESHOLD:
            mitigations.append({
                "measure": f"Bias Monitoring (Score: {bias_score:.4f})",
                "status": "passing",
                "effectiveness": "high"
            })
        if rbac_enabled:
            mitigations.append({
                "measure": "Role-Based Access Control (RBAC)",
                "status": "implemented",
                "effectiveness": "medium"
            })
        
        # Auto-approve if risk is low and compliance is met
        auto_approve = (
            risk_level == "LOW" and
            compliance["gdpr_compliant"] and
            compliance["uk_dpa_compliant"]
        )
        
        # Create assessment
        assessment = DPIAAssessment(
            assessment_version=assessment_version,
            risk_level=risk_level,
            overall_risk_score=overall_risk,
            privacy_risks=json.dumps(risk_breakdown, indent=2),
            gdpr_compliant=compliance["gdpr_compliant"],
            uk_dpa_compliant=compliance["uk_dpa_compliant"],
            bias_threshold_met=compliance["bias_threshold"],
            encryption_verified=encryption_enabled,
            audit_logs_enabled=audit_logs_enabled,
            approval_status="approved" if auto_approve else "pending",
            approved_by_id=created_by.id if auto_approve else None,
            approval_date=datetime.now(timezone.utc) if auto_approve else None,
            valid_until=datetime.now(timezone.utc) + timedelta(days=self.REASSESSMENT_DAYS) if auto_approve else None,
            assessment_notes=notes,
            mitigation_measures=json.dumps(mitigations, indent=2),
            created_by_id=created_by.id
        )
        
        # Generate recommendations
        recommendations = []
        if risk_level == "HIGH":
            recommendations.append("⚠️ High risk detected - address issues before proceeding")
        if not encryption_enabled:
            recommendations.append("Enable AES-256-GCM encryption immediately")
        if not audit_logs_enabled:
            recommendations.append("Enable hash-chained audit logging")
        if bias_score is None:
            recommendations.append("Run bias monitoring evaluation")
        elif bias_score >= self.BIAS_THRESHOLD:
            recommendations.append(f"Bias score ({bias_score:.4f}) exceeds threshold - retrain model")
        if recent_breaches > 0:
            recommendations.append(f"Investigate {recent_breaches} recent security incident(s)")
        if auto_approve:
            recommendations.append("✅ System meets compliance standards - approved for processing")
        else:
            recommendations.append("⏳ Manual review required before approval")
        
        assessment.recommendations = json.dumps(recommendations, indent=2)
        
        self.db.add(assessment)
        self.db.commit()
        self.db.refresh(assessment)
        
        logger.info(f"DPIA assessment created: ID={assessment.id}, Risk={risk_level}, Status={assessment.approval_status}")
        
        return assessment
    
    def get_active_assessment(self) -> Optional[DPIAAssessment]:
        """
        Get the currently active (approved and not expired) DPIA assessment.
        
        Returns:
            Active DPIAAssessment or None if no valid assessment exists
        """
        now = datetime.now(timezone.utc)
        assessment = (
            self.db.query(DPIAAssessment)
            .filter(
                DPIAAssessment.approval_status == "approved",
                DPIAAssessment.valid_until > now
            )
            .order_by(desc(DPIAAssessment.assessment_date))
            .first()
        )
        return assessment
    
    def is_processing_allowed(self) -> Tuple[bool, Optional[str]]:
        """
        Check if biometric data processing is currently allowed under DPIA.
        
        Returns:
            Tuple of (is_allowed, reason)
        """
        active_assessment = self.get_active_assessment()
        
        if active_assessment is None:
            return False, "No active DPIA assessment found - create new assessment before processing"
        
        if active_assessment.risk_level == "HIGH":
            return False, f"High risk assessment (ID={active_assessment.id}) - processing blocked until risks mitigated"
        
        if not active_assessment.gdpr_compliant or not active_assessment.uk_dpa_compliant:
            return False, "DPIA assessment shows non-compliance - resolve issues before processing"
        
        return True, f"Processing allowed under DPIA assessment ID={active_assessment.id}"
    
    def approve_assessment(
        self,
        assessment_id: int,
        approved_by: User,
        notes: Optional[str] = None
    ) -> DPIAAssessment:
        """
        Approve a pending DPIA assessment.
        
        Args:
            assessment_id: ID of assessment to approve
            approved_by: User approving the assessment
            notes: Optional approval notes
            
        Returns:
            Updated DPIAAssessment
        """
        assessment = self.db.query(DPIAAssessment).filter(DPIAAssessment.id == assessment_id).first()
        if not assessment:
            raise ValueError(f"DPIA assessment {assessment_id} not found")
        
        if assessment.approval_status != "pending":
            raise ValueError(f"Assessment {assessment_id} is already {assessment.approval_status}")
        
        if assessment.risk_level == "HIGH":
            raise ValueError(f"Cannot approve HIGH risk assessment - mitigate risks first")
        
        assessment.approval_status = "approved"
        assessment.approved_by_id = approved_by.id
        assessment.approval_date = datetime.now(timezone.utc)
        assessment.valid_until = datetime.now(timezone.utc) + timedelta(days=self.REASSESSMENT_DAYS)
        
        if notes:
            current_notes = assessment.assessment_notes or ""
            assessment.assessment_notes = f"{current_notes}\n\n[Approval Notes - {datetime.now(timezone.utc).isoformat()}]\n{notes}"
        
        self.db.commit()
        self.db.refresh(assessment)
        
        logger.info(f"DPIA assessment {assessment_id} approved by user {approved_by.username}")
        
        return assessment
    
    def reject_assessment(
        self,
        assessment_id: int,
        rejected_by: User,
        reason: str
    ) -> DPIAAssessment:
        """
        Reject a pending DPIA assessment.
        
        Args:
            assessment_id: ID of assessment to reject
            rejected_by: User rejecting the assessment
            reason: Reason for rejection
            
        Returns:
            Updated DPIAAssessment
        """
        assessment = self.db.query(DPIAAssessment).filter(DPIAAssessment.id == assessment_id).first()
        if not assessment:
            raise ValueError(f"DPIA assessment {assessment_id} not found")
        
        if assessment.approval_status != "pending":
            raise ValueError(f"Assessment {assessment_id} is already {assessment.approval_status}")
        
        assessment.approval_status = "rejected"
        
        current_notes = assessment.assessment_notes or ""
        assessment.assessment_notes = f"{current_notes}\n\n[Rejection - {datetime.now(timezone.utc).isoformat()} by {rejected_by.username}]\n{reason}"
        
        self.db.commit()
        self.db.refresh(assessment)
        
        logger.warning(f"DPIA assessment {assessment_id} rejected by user {rejected_by.username}: {reason}")
        
        return assessment
    
    def get_latest_bias_score(self) -> Optional[float]:
        """
        Fetch the most recent bias monitoring score from database.
        
        Returns:
            Latest bias score or None if no bias audits found
        """
        latest_bias = (
            self.db.query(PredictiveLog)
            .filter(PredictiveLog.prediction_type == "bias_audit")
            .order_by(desc(PredictiveLog.timestamp))
            .first()
        )
        
        if latest_bias and latest_bias.bias_score is not None:
            return latest_bias.bias_score
        
        return None
    
    def get_all_assessments(
        self,
        limit: int = 50,
        status_filter: Optional[str] = None
    ) -> List[DPIAAssessment]:
        """
        Get all DPIA assessments, optionally filtered by status.
        
        Args:
            limit: Maximum number of assessments to return
            status_filter: Filter by approval status ('pending', 'approved', 'rejected')
            
        Returns:
            List of DPIAAssessment objects
        """
        query = self.db.query(DPIAAssessment)
        
        if status_filter:
            query = query.filter(DPIAAssessment.approval_status == status_filter)
        
        assessments = (
            query
            .order_by(desc(DPIAAssessment.assessment_date))
            .limit(limit)
            .all()
        )
        
        return assessments
    
    def generate_compliance_report(self) -> Dict:
        """
        Generate a comprehensive compliance report.
        
        Returns:
            Dictionary with compliance status, active assessment, and recommendations
        """
        active_assessment = self.get_active_assessment()
        is_allowed, reason = self.is_processing_allowed()
        latest_bias = self.get_latest_bias_score()
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_allowed": is_allowed,
            "processing_status_reason": reason,
            "active_assessment": None,
            "latest_bias_score": latest_bias,
            "compliance_summary": {
                "gdpr_article_9": False,
                "uk_dpa_part_3": False,
                "bias_threshold_met": latest_bias is not None and latest_bias < self.BIAS_THRESHOLD if latest_bias else False
            },
            "recent_assessments_count": {
                "approved": self.db.query(DPIAAssessment).filter(DPIAAssessment.approval_status == "approved").count(),
                "pending": self.db.query(DPIAAssessment).filter(DPIAAssessment.approval_status == "pending").count(),
                "rejected": self.db.query(DPIAAssessment).filter(DPIAAssessment.approval_status == "rejected").count()
            }
        }
        
        if active_assessment:
            report["active_assessment"] = {
                "id": active_assessment.id,
                "version": active_assessment.assessment_version,
                "risk_level": active_assessment.risk_level,
                "risk_score": active_assessment.overall_risk_score,
                "approved_date": active_assessment.approval_date.isoformat() if active_assessment.approval_date else None,
                "valid_until": active_assessment.valid_until.isoformat() if active_assessment.valid_until else None,
                "gdpr_compliant": active_assessment.gdpr_compliant,
                "uk_dpa_compliant": active_assessment.uk_dpa_compliant
            }
            report["compliance_summary"]["gdpr_article_9"] = active_assessment.gdpr_compliant
            report["compliance_summary"]["uk_dpa_part_3"] = active_assessment.uk_dpa_compliant
        
        return report
