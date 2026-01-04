"""Add missing tables: evidence, snapshots, stream_sources

Revision ID: d0ce2c180103
Revises: 21f2de674dac
Create Date: 2025-11-17 00:14:54.054618+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd0ce2c180103'
down_revision = '21f2de674dac'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create evidence table
    op.create_table('evidence',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('file_name', sa.String(), nullable=True),
        sa.Column('media_type', sa.String(), nullable=True),
        sa.Column('evidence_type', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('content', sa.LargeBinary(), nullable=False),
        sa.Column('encrypted_dek', sa.LargeBinary(), nullable=True),
        sa.Column('is_encrypted', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_evidence_id'), 'evidence', ['id'], unique=False)

    # Create snapshots table
    op.create_table('snapshots',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('identity_id', sa.String(), nullable=True),
        sa.Column('location', sa.String(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('encrypted_image_data', sa.LargeBinary(), nullable=False),
        sa.Column('encrypted_dek', sa.LargeBinary(), nullable=True),
        sa.Column('is_encrypted', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create stream_sources table
    op.create_table('stream_sources',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('rtsp_url', sa.String(), nullable=False),
        sa.Column('location', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index(op.f('ix_stream_sources_id'), 'stream_sources', ['id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_stream_sources_id'), table_name='stream_sources')
    op.drop_table('stream_sources')
    op.drop_table('snapshots')
    op.drop_index(op.f('ix_evidence_id'), table_name='evidence')
    op.drop_table('evidence')
