"""
SQLAlchemy ``MetaData`` / ``Table`` definitions used only by Alembic for
migration autogeneration. Runtime access is plain asyncpg.
"""

from __future__ import annotations

import sqlalchemy as sa

metadata = sa.MetaData()


# -- Flow 2: conspect generation queue ---------------------------------

conspect_jobs = sa.Table(
    "conspect_jobs",
    metadata,
    sa.Column("job_id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column("user_id", sa.Text, nullable=False),
    sa.Column(
        "status",
        sa.Text,
        nullable=False,
        server_default=sa.text("'queued'"),
        comment="queued | running | done | failed",
    ),
    sa.Column(
        "payload",
        sa.dialects.postgresql.JSONB,
        nullable=False,
        comment="Full JobCreateRequest body (submissions[], callback, etc.)",
    ),
    sa.Column(
        "result",
        sa.dialects.postgresql.JSONB,
        nullable=True,
        comment="Generated conspect + auto-check fields, after worker completes",
    ),
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
)

sa.Index(
    "ix_conspect_jobs_status_created",
    conspect_jobs.c.status,
    conspect_jobs.c.created_at,
    postgresql_where=sa.text("status IN ('queued', 'running')"),
)
