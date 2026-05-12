"""
Proxy ``/jobs`` to the generator API so the dashboard is same-origin
(no CORS, single port-forward). Injects the real service token; the
browser only sends a placeholder.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.core.config import get_settings

router = APIRouter()


def _service_headers() -> dict[str, str]:
    h = {"content-type": "application/json"}
    token = get_settings().platform_auth_token
    if token:
        h["authorization"] = f"Bearer {token}"
    return h


@router.post("/jobs")
async def proxy_create_job(request: Request):
    raw = await request.body()
    body = json.loads(raw) if raw else {}
    # Default the callback to admin so the worker can deliver back here.
    body.setdefault("callback", {}).setdefault("url", "http://ege-admin/v1/personal-conspect")
    r = await request.app.state.api_client.post("/jobs", json=body, headers=_service_headers())
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


@router.get("/jobs/{job_id}")
async def proxy_get_job(job_id: str, request: Request):
    r = await request.app.state.api_client.get(f"/jobs/{job_id}", headers=_service_headers())
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
