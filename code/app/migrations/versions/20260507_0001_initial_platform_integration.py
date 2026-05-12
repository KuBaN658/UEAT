"""initial: conspect_jobs

Revision ID: 3e4b8e0d6e21
Revises:
Create Date: 2026-05-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "3e4b8e0d6e21"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "conspect_jobs",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column(
            "status",
            sa.Text,
            nullable=False,
            server_default=sa.text("'queued'"),
        ),
        sa.Column("payload", postgresql.JSONB, nullable=False),
        sa.Column("result", postgresql.JSONB, nullable=True),
        sa.Column("error_code", sa.Text, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("retry_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("posted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'done', 'failed')",
            name="conspect_jobs_status_check",
        ),
    )

    op.create_index(
        "ix_conspect_jobs_status_created",
        "conspect_jobs",
        ["status", "created_at"],
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )

    op.create_index(
        "ix_conspect_jobs_user_created",
        "conspect_jobs",
        ["user_id", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_conspect_jobs_user_created", table_name="conspect_jobs")
    op.drop_index("ix_conspect_jobs_status_created", table_name="conspect_jobs")
    op.drop_table("conspect_jobs")
