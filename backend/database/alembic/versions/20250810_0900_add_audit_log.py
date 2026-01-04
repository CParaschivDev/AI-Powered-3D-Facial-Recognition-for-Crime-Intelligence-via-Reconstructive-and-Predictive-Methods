"""Add immutable audit log table

Revision ID: a1b2c3d4e5f6
Revises: 9c8d7e6f5a4b
Create Date: 2025-08-10 09:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '9c8d7e6f5a4b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the old, simple audit log table if it exists
    op.drop_table('audit_logs')

    # Create the new, hash-chained audit log table
    op.create_table('audit_log',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('case_id', sa.String(), nullable=False),
    sa.Column('actor_id', sa.Integer(), nullable=False),
    sa.Column('action', sa.String(), nullable=False),
    sa.Column('payload_json', sa.String(), nullable=False),
    sa.Column('file_hash', sa.String(), nullable=True),
    sa.Column('prev_hash', sa.String(), nullable=True),
    sa.Column('hash', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
    sa.ForeignKeyConstraint(['actor_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('hash')
    )
    op.create_index(op.f('ix_audit_log_case_id'), 'audit_log', ['case_id'], unique=False)
    op.create_index(op.f('ix_audit_log_id'), 'audit_log', ['id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_audit_log_id'), table_name='audit_log')
    op.drop_index(op.f('ix_audit_log_case_id'), table_name='audit_log')
    op.drop_table('audit_log')

    # Restore the old, simple audit log table
    op.create_table('audit_logs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
    sa.Column('user', sa.String(), nullable=True),
    sa.Column('ip_address', sa.String(), nullable=True),
    sa.Column('endpoint', sa.String(), nullable=True),
    sa.Column('method', sa.String(), nullable=True),
    sa.Column('status_code', sa.Integer(), nullable=True),
    sa.Column('response_time_ms', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
