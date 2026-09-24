"""aspect status, sentiment per review

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-24 23:30:00

Adds games.aspects_status/aspects_error next to aspects_processed_at (which keeps meaning
"last successful run") and moves predicted_sentiment from review_aspects to reviews. Every
game comes back "pending", so the next /aspects request (or scripts/process_reviews.py)
reprocesses it; that run replaces the old review_aspects rows. Downgrading deletes
review_aspects rows for the same reason.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0003'
down_revision: str | Sequence[str] | None = '0002'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('reviews', sa.Column('predicted_sentiment', sa.String(length=8), nullable=True))
    op.drop_column('review_aspects', 'predicted_sentiment')
    op.add_column(
        'games',
        sa.Column('aspects_status', sa.String(length=16), server_default='pending', nullable=False),
    )
    op.add_column('games', sa.Column('aspects_error', sa.Text(), nullable=True))
    op.create_check_constraint(
        'ck_games_aspects_status',
        'games',
        "aspects_status IN ('pending', 'processing', 'done', 'failed')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute('UPDATE games SET aspects_processed_at = NULL')  # its rows are deleted below
    op.drop_constraint('ck_games_aspects_status', 'games', type_='check')
    op.drop_column('games', 'aspects_error')
    op.drop_column('games', 'aspects_status')
    op.execute('DELETE FROM review_aspects')
    op.add_column(
        'review_aspects', sa.Column('predicted_sentiment', sa.String(length=8), nullable=False)
    )
    op.drop_column('reviews', 'predicted_sentiment')
