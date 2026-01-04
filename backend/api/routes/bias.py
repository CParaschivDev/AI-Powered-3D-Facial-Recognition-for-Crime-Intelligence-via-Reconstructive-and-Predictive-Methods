"""
Bias Monitoring API Routes

Provides endpoints for accessing fairness metrics and bias audit logs.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import json
from datetime import datetime, timedelta

from backend.database.dependencies import get_db
from backend.database.models import PredictiveLog, User
from backend.api.models.schemas import BiasAuditDecision
from backend.core.dependencies import get_current_user

router = APIRouter(prefix="/bias", tags=["Bias Monitoring"])


@router.get("/reports")
def get_bias_reports(
    limit: int = Query(10, ge=1, le=100),
    flagged_only: bool = Query(False),
    db: Session = Depends(get_db)
):
    """
    Retrieve bias monitoring reports.
    
    Args:
        limit: Maximum number of reports to return
        flagged_only: If True, only return flagged reports
        
    Returns:
        List of bias audit reports
    """
    query = db.query(PredictiveLog).filter(
        PredictiveLog.prediction_type == "bias_audit"
    )
    
    if flagged_only:
        query = query.filter(PredictiveLog.is_flagged == True)
    
    reports = query.order_by(PredictiveLog.timestamp.desc()).limit(limit).all()
    
    result = []
    for report in reports:
        result.append({
            "id": report.id,
            "model_name": report.model_name,
            "timestamp": report.timestamp.isoformat(),
            "bias_score": report.bias_score,
            "is_flagged": report.is_flagged,
            "audit_status": report.audit_status,
            "audit_notes": report.audit_notes,
            "auditor_id": report.auditor_id,
            "full_report": json.loads(report.prediction_data) if report.prediction_data else {}
        })
    
    return {
        "count": len(result),
        "reports": result
    }


@router.get("/reports/{report_id}")
def get_bias_report_detail(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    Retrieve detailed bias report by ID.
    
    Args:
        report_id: ID of the bias report
        
    Returns:
        Full bias report with all metrics
    """
    report = db.query(PredictiveLog).filter(
        PredictiveLog.id == report_id,
        PredictiveLog.prediction_type == "bias_audit"
    ).first()
    
    if not report:
        raise HTTPException(status_code=404, detail="Bias report not found")
    
    return {
        "id": report.id,
        "model_name": report.model_name,
        "timestamp": report.timestamp.isoformat(),
        "bias_score": report.bias_score,
        "is_flagged": report.is_flagged,
        "audit_status": report.audit_status,
        "audit_notes": report.audit_notes,
        "auditor_id": report.auditor_id,
        "full_report": json.loads(report.prediction_data)
    }


@router.post("/reports/{report_id}/audit")
def audit_bias_report(
    report_id: int,
    decision: BiasAuditDecision,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Submit an audit decision for a flagged bias report.
    
    Args:
        report_id: ID of the bias report
        decision: Audit decision (approve/reject) with notes
        
    Returns:
        Updated report
    """
    report = db.query(PredictiveLog).filter(
        PredictiveLog.id == report_id,
        PredictiveLog.prediction_type == "bias_audit"
    ).first()
    
    if not report:
        raise HTTPException(status_code=404, detail="Bias report not found")
    
    # Update audit fields
    report.audit_status = decision.decision
    report.audit_notes = decision.notes
    report.auditor_id = str(current_user.id)
    
    db.commit()
    db.refresh(report)
    
    return {
        "message": f"Bias report {decision.decision}",
        "report_id": report.id,
        "audit_status": report.audit_status,
        "auditor": current_user.username
    }


@router.get("/summary")
def get_bias_summary(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    Get summary statistics for bias monitoring.
    
    Args:
        days: Number of days to include in summary
        
    Returns:
        Summary statistics
    """
    since = datetime.utcnow() - timedelta(days=days)
    
    reports = db.query(PredictiveLog).filter(
        PredictiveLog.prediction_type == "bias_audit",
        PredictiveLog.timestamp >= since
    ).all()
    
    if not reports:
        return {
            "period_days": days,
            "total_audits": 0,
            "flagged_count": 0,
            "average_bias_score": 0.0,
            "pending_reviews": 0,
            "approved": 0,
            "rejected": 0
        }
    
    flagged = sum(1 for r in reports if r.is_flagged)
    pending = sum(1 for r in reports if r.audit_status == "pending")
    approved = sum(1 for r in reports if r.audit_status == "approved")
    rejected = sum(1 for r in reports if r.audit_status == "rejected")
    
    bias_scores = [r.bias_score for r in reports if r.bias_score is not None]
    avg_bias = sum(bias_scores) / len(bias_scores) if bias_scores else 0.0
    
    return {
        "period_days": days,
        "total_audits": len(reports),
        "flagged_count": flagged,
        "average_bias_score": round(avg_bias, 4),
        "max_bias_score": round(max(bias_scores), 4) if bias_scores else 0.0,
        "min_bias_score": round(min(bias_scores), 4) if bias_scores else 0.0,
        "pending_reviews": pending,
        "approved": approved,
        "rejected": rejected,
        "flagged_rate": round(flagged / len(reports), 2) if reports else 0.0
    }


@router.get("/trend")
def get_bias_trend(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    Get bias score trend over time.
    
    Args:
        days: Number of days to include
        
    Returns:
        Time series of bias scores
    """
    since = datetime.utcnow() - timedelta(days=days)
    
    reports = db.query(PredictiveLog).filter(
        PredictiveLog.prediction_type == "bias_audit",
        PredictiveLog.timestamp >= since,
        PredictiveLog.bias_score != None
    ).order_by(PredictiveLog.timestamp.asc()).all()
    
    trend_data = [
        {
            "timestamp": r.timestamp.isoformat(),
            "bias_score": r.bias_score,
            "is_flagged": r.is_flagged,
            "model_name": r.model_name
        }
        for r in reports
    ]
    
    return {
        "period_days": days,
        "data_points": len(trend_data),
        "trend": trend_data
    }
