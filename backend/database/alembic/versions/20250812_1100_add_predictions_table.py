"""Add predictions table

Revision ID: f3a4b5c6d7e8
Revises: 1a2b3c4d5e6f
Create Date: 2025-08-12 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f3a4b5c6d7e8'
down_revision = '1a2b3c4d5e6f'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('predictions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('area_id', sa.String(), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('crime_type', sa.String(), nullable=False),
    sa.Column('yhat', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_predictions_area_id'), 'predictions', ['area_id'], unique=False)
    op.create_index(op.f('ix_predictions_crime_type'), 'predictions', ['crime_type'], unique=False)
    op.create_index(op.f('ix_predictions_id'), 'predictions', ['id'], unique=False)
    op.create_index(op.f('ix_predictions_ts'), 'predictions', ['ts'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_predictions_ts'), table_name='predictions')
    op.drop_index(op.f('ix_predictions_id'), table_name='predictions')
    op.drop_index(op.f('ix_predictions_crime_type'), table_name='predictions')
    op.drop_index(op.f('ix_predictions_area_id'), table_name='predictions')
    op.drop_table('predictions')
