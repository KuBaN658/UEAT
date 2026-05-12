"""Receive callbacks from our workers (Flow 1 tags, Flow 2 conspect)."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Request

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/submissions/{submission_id}/diagnostic-tags", status_code=200)
async def recv_diagnostic_tags(submission_id: UUID, request: Request):
    body = await request.json()
    tags = body.get("tags") or []
    log.info(
        "<- diagnostic-tags submission=%s tags=%s model=%s",
        submission_id, tags, body.get("model"),
    )
    request.app.state.submission_store.patch_tags(str(submission_id), tags)
    return {"status": "ok"}


@router.post("/v1/personal-conspect", status_code=200)
async def recv_conspect(request: Request):
    body = await request.json()
    log.info(
        "<- personal-conspect job=%s user=%s len=%d",
        body.get("job_id"), body.get("user_id"), len(body.get("text", "")),
    )
    return {"status": "ok"}
