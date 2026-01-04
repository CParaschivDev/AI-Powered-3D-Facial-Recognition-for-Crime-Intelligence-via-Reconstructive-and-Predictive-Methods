"""add_training_output_path_to_model_versions

Revision ID: 21f2de674dac
Revises: e5d6f7a8b9c0
Create Date: 2025-10-06 23:24:50.167058+01:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '21f2de674dac'
down_revision = 'e5d6f7a8b9c0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add training_output_path column to model_versions table
    op.add_column('model_versions', sa.Column('training_output_path', sa.String(), nullable=True))


def downgrade() -> None:
    # Remove training_output_path column from model_versions table
    op.drop_column('model_versions', 'training_output_path')
