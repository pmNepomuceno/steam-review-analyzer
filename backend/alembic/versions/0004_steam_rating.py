"""steam store rating on games

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-02 17:00:00

Adds Steam's own store rating; it starts NULL. aspects_processed_at is not backfilled:
for a done game without one, the real analysis date is unknown, and the homepage shows
no date rather than a made-up one.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0004'
down_revision: str | Sequence[str] | None = '0003'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('games', sa.Column('steam_score_desc', sa.Text(), nullable=True))
    op.add_column('games', sa.Column('steam_positive', sa.Integer(), nullable=True))
    op.add_column('games', sa.Column('steam_total', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('games', 'steam_total')
    op.drop_column('games', 'steam_positive')
    op.drop_column('games', 'steam_score_desc')
