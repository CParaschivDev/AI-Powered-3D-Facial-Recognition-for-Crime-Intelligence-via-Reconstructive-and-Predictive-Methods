"""merge_bias_monitoring

Revision ID: 2bdc5e24a9c9
Revises: bias_monitoring_001, d0ce2c180103
Create Date: 2025-12-08 10:32:57.372402+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2bdc5e24a9c9'
down_revision = ('bias_monitoring_001', 'd0ce2c180103')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
