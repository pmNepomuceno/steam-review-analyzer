"""stored aspect quotes, steam rating check time

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-02 19:00:00

Quotes and Steam's rating move from request time to processing time (process_reviews).
Both new columns start NULL, so games already "done" are served without quotes or a
rating until `scripts/process_reviews.py --force` reruns them; nothing is backfilled here
because quotes need the sentiment model.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0005'
down_revision: str | Sequence[str] | None = '0004'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('games', sa.Column('aspect_quotes', postgresql.JSONB(), nullable=True))
    op.add_column(
        'games', sa.Column('steam_rating_checked_at', sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('games', 'steam_rating_checked_at')
    op.drop_column('games', 'aspect_quotes')
