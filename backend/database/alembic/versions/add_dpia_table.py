"""add dpia assessments table

Revision ID: add_dpia_table
Revises: 
Create Date: 2025-12-08

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import func

# revision identifiers, used by Alembic.
revision = 'add_dpia_table'
down_revision = '2bdc5e24a9c9'
branch_labels = None
depends_on = None


def upgrade():
    # Create dpia_assessments table
    op.create_table(
        'dpia_assessments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('assessment_date', sa.DateTime(timezone=True), server_default=func.now(), nullable=False),
        sa.Column('assessment_version', sa.String(), nullable=False),
        sa.Column('risk_level', sa.String(), nullable=False),
        sa.Column('overall_risk_score', sa.Float(), nullable=False),
        sa.Column('privacy_risks', sa.Text(), nullable=False),
        sa.Column('gdpr_compliant', sa.Boolean(), nullable=False),
        sa.Column('uk_dpa_compliant', sa.Boolean(), nullable=False),
        sa.Column('bias_threshold_met', sa.Boolean(), nullable=False),
        sa.Column('encryption_verified', sa.Boolean(), nullable=False),
        sa.Column('audit_logs_enabled', sa.Boolean(), nullable=False),
        sa.Column('approval_status', sa.String(), nullable=False),
        sa.Column('approved_by_id', sa.Integer(), nullable=True),
        sa.Column('approval_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('valid_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('assessment_notes', sa.Text(), nullable=True),
        sa.Column('mitigation_measures', sa.Text(), nullable=True),
        sa.Column('recommendations', sa.Text(), nullable=True),
        sa.Column('created_by_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['approved_by_id'], ['users.id']),
        sa.ForeignKeyConstraint(['created_by_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_dpia_assessments_id'), 'dpia_assessments', ['id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_dpia_assessments_id'), table_name='dpia_assessments')
    op.drop_table('dpia_assessments')
