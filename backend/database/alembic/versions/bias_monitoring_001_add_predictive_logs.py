"""add predictive logs table for bias monitoring

Revision ID: bias_monitoring_001
Revises: 
Create Date: 2025-12-08 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'bias_monitoring_001'
down_revision = None  # Set this to your latest migration revision
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create predictive_logs table for bias monitoring
    op.create_table('predictive_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('prediction_type', sa.String(), nullable=False),
        sa.Column('prediction_data', sa.Text(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('bias_score', sa.Float(), nullable=True),
        sa.Column('is_flagged', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('auditor_id', sa.String(), nullable=True),
        sa.Column('audit_status', sa.String(), nullable=True),
        sa.Column('audit_notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_predictive_logs_id'), 'predictive_logs', ['id'], unique=False)
    op.create_index(op.f('ix_predictive_logs_model_name'), 'predictive_logs', ['model_name'], unique=False)
    op.create_index(op.f('ix_predictive_logs_is_flagged'), 'predictive_logs', ['is_flagged'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_predictive_logs_is_flagged'), table_name='predictive_logs')
    op.drop_index(op.f('ix_predictive_logs_model_name'), table_name='predictive_logs')
    op.drop_index(op.f('ix_predictive_logs_id'), table_name='predictive_logs')
    op.drop_table('predictive_logs')
