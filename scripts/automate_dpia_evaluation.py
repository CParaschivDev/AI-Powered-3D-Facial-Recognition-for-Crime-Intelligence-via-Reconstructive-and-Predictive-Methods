"""
Automated DPIA Evaluation Script

Runs comprehensive Data Protection Impact Assessment with automated compliance checks.
Similar to bias monitoring - creates assessment, checks risks, and logs to database.

Usage:
    python scripts/automate_dpia_evaluation.py
    python scripts/automate_dpia_evaluation.py --version "2.0" --notes "Monthly review"
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from backend.database.dependencies import SessionLocal
from backend.database.models import User as DBUser
from backend.services.dpia_monitor import DPIAMonitor
import json


def main():
    parser = argparse.ArgumentParser(description="Run automated DPIA compliance evaluation")
    parser.add_argument(
        "--version",
        type=str,
        default="auto",
        help="Assessment version identifier (default: 'auto')"
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="Automated DPIA evaluation",
        help="Assessment notes"
    )
    parser.add_argument(
        "--encryption",
        action="store_true",
        default=True,
        help="Mark encryption as enabled (default: True)"
    )
    parser.add_argument(
        "--no-encryption",
        action="store_false",
        dest="encryption",
        help="Mark encryption as disabled"
    )
    parser.add_argument(
        "--audit-logs",
        action="store_true",
        default=True,
        help="Mark audit logs as enabled (default: True)"
    )
    parser.add_argument(
        "--no-audit-logs",
        action="store_false",
        dest="audit_logs",
        help="Mark audit logs as disabled"
    )
    parser.add_argument(
        "--rbac",
        action="store_true",
        default=True,
        help="Mark RBAC as enabled (default: True)"
    )
    parser.add_argument(
        "--breaches",
        type=int,
        default=0,
        help="Number of recent security breaches (default: 0)"
    )
    parser.add_argument(
        "--skip-database",
        action="store_true",
        help="Don't save to database (just print to console)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DPIA (Data Protection Impact Assessment) Evaluation")
    print("=" * 80)
    
    # Get database session
    if not args.skip_database:
        db = SessionLocal()
        monitor = DPIAMonitor(db)
        
        # Get or create admin user for assessment
        admin_user = db.query(DBUser).filter(DBUser.role == "admin").first()
        if not admin_user:
            print("WARNING: No admin user found - creating default admin for assessment")
            from backend.core.security import get_password_hash
            admin_user = DBUser(
                username="admin",
                hashed_password=get_password_hash("admin_password"),
                role="admin",
                disabled=False
            )
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)
        
        print(f"Assessment Creator: {admin_user.username} (ID: {admin_user.id})")
        print()
        
        # Fetch latest bias score from database
        print("Fetching latest bias monitoring score...")
        latest_bias = monitor.get_latest_bias_score()
        if latest_bias is not None:
            print(f"   Bias Score: {latest_bias:.4f} ({latest_bias * 100:.2f}%)")
        else:
            print("   WARNING: No bias audits found in database")
        print()
        
        # Create DPIA assessment
        print("Running automated compliance checks...")
        print(f"   Encryption: {'Enabled' if args.encryption else 'Disabled'}")
        print(f"   Audit Logs: {'Enabled' if args.audit_logs else 'Disabled'}")
        print(f"   RBAC: {'Enabled' if args.rbac else 'Disabled'}")
        print(f"   Security Breaches (90d): {args.breaches}")
        print()
        
        assessment = monitor.create_assessment(
            created_by=admin_user,
            assessment_version=args.version,
            bias_score=latest_bias,
            encryption_enabled=args.encryption,
            audit_logs_enabled=args.audit_logs,
            rbac_enabled=args.rbac,
            recent_breaches=args.breaches,
            notes=args.notes
        )
        
        # Display results
        print("=" * 80)
        print("DPIA ASSESSMENT RESULTS")
        print("=" * 80)
        print(f"Assessment ID: {assessment.id}")
        print(f"Version: {assessment.assessment_version}")
        print(f"Date: {assessment.assessment_date}")
        print()
        
        # Risk assessment
        print(f"RISK ASSESSMENT")
        print(f"   Overall Score: {assessment.overall_risk_score:.4f} ({assessment.overall_risk_score * 100:.1f}%)")
        print(f"   Risk Level: {assessment.risk_level}")
        print()
        
        # Risk breakdown
        risks = json.loads(assessment.privacy_risks)
        print("   Risk Breakdown:")
        for risk_name, risk_score in risks.items():
            risk_pct = risk_score * 100
            status = "[HIGH]" if risk_score > 0.6 else "[MED]" if risk_score > 0.3 else "[LOW]"
            print(f"      {status} {risk_name.replace('_', ' ').title()}: {risk_pct:.1f}%")
        print()
        
        # Compliance
        print(f"COMPLIANCE STATUS")
        print(f"   GDPR Article 9 (Special Category Data): {'PASS' if assessment.gdpr_compliant else 'FAIL'}")
        print(f"   UK DPA 2018 Part 3 (Law Enforcement): {'PASS' if assessment.uk_dpa_compliant else 'FAIL'}")
        print(f"   Bias Threshold (<5%): {'PASS' if assessment.bias_threshold_met else 'FAIL'}")
        print(f"   Encryption (AES-256-GCM): {'VERIFIED' if assessment.encryption_verified else 'NOT VERIFIED'}")
        print(f"   Audit Logs (SHA-256): {'ENABLED' if assessment.audit_logs_enabled else 'DISABLED'}")
        print()
        
        # Approval status
        print(f"APPROVAL STATUS")
        print(f"   Status: {assessment.approval_status.upper()}")
        if assessment.approval_status == "approved":
            print(f"   Approved By: {assessment.approved_by.username if assessment.approved_by else 'Auto-approved'}")
            print(f"   Approved Date: {assessment.approval_date}")
            print(f"   Valid Until: {assessment.valid_until}")
            print(f"   Biometric processing is ALLOWED under this assessment")
        elif assessment.approval_status == "pending":
            print(f"   Awaiting manual review by admin")
        else:
            print(f"   Processing BLOCKED - assessment rejected")
        print()
        
        # Mitigation measures
        mitigations = json.loads(assessment.mitigation_measures)
        if mitigations:
            print(f"MITIGATION MEASURES ({len(mitigations)} implemented):")
            for measure in mitigations:
                status_mark = "[OK]" if measure["status"] in ["implemented", "passing"] else "[WARN]"
                print(f"   {status_mark} {measure['measure']} (Effectiveness: {measure['effectiveness']})")
            print()
        
        # Recommendations
        recommendations = json.loads(assessment.recommendations)
        if recommendations:
            print(f"RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   - {rec}")
            print()
        
        print("=" * 80)
        print(f"DPIA assessment saved to database (ID: {assessment.id})")
        print(f"View in API: GET /api/v1/dpia/assessments/{assessment.id}")
        print("=" * 80)
        
        # Close database session
        db.close()
        
    else:
        # Skip database mode - just compute risk score
        print("WARNING: --skip-database mode: Results will NOT be saved")
        print()
        
        monitor = DPIAMonitor(None)  # No DB session
        
        overall_risk, risk_breakdown, risk_level = monitor.compute_risk_score(
            bias_score=None,  # Would fetch from DB normally
            encryption_enabled=args.encryption,
            audit_logs_enabled=args.audit_logs,
            rbac_enabled=args.rbac,
            recent_breaches=args.breaches
        )
        
        print(f"RISK ASSESSMENT (Dry Run)")
        print(f"   Overall Score: {overall_risk:.4f} ({overall_risk * 100:.1f}%)")
        print(f"   Risk Level: {risk_level}")
        print()
        
        print("   Risk Breakdown:")
        for risk_name, risk_score in risk_breakdown.items():
            risk_pct = risk_score * 100
            status = "[HIGH]" if risk_score > 0.6 else "[MED]" if risk_score > 0.3 else "[LOW]"
            print(f"      {status} {risk_name.replace('_', ' ').title()}: {risk_pct:.1f}%")
        print()
        
        print("Run without --skip-database to save assessment to database")


if __name__ == "__main__":
    main()
