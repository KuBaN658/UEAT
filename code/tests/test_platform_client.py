"""
Tests for ``PlatformClient`` -- uses ``respx`` to intercept httpx calls
without a real network.
"""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
import respx

from app.infrastructure.http.platform_client import (
    PermanentDeliveryError,
    PlatformClient,
    TransientDeliveryError,
)

pytestmark = pytest.mark.asyncio


@respx.mock
async def test_post_conspect_sends_idempotency_key_and_bearer():
    submission_id = uuid4()
    callback_url = "https://platform.example.com/v1/personal-conspect"
    route = respx.post(callback_url).mock(return_value=httpx.Response(204))

    client = PlatformClient()
    try:
        await client.post_conspect(
            callback_url=callback_url,
            callback_token="job-specific-token",
            job_id=submission_id,
            payload={"job_id": str(submission_id), "text": "..."},
        )
    finally:
        await client.aclose()

    assert route.called
    req = route.calls.last.request
    assert req.headers["Idempotency-Key"] == str(submission_id)
    assert req.headers["Authorization"] == "Bearer job-specific-token"


@respx.mock
async def test_post_conspect_4xx_is_permanent():
    callback_url = "https://platform.example.com/v1/personal-conspect"
    respx.post(callback_url).mock(return_value=httpx.Response(400, text="bad payload"))

    client = PlatformClient()
    try:
        with pytest.raises(PermanentDeliveryError) as exc:
            await client.post_conspect(
                callback_url=callback_url,
                callback_token="t",
                job_id=uuid4(),
                payload={},
            )
        assert exc.value.status_code == 400
    finally:
        await client.aclose()


@respx.mock
async def test_post_conspect_5xx_retries_then_transient():
    callback_url = "https://platform.example.com/v1/personal-conspect"
    route = respx.post(callback_url).mock(return_value=httpx.Response(500))

    client = PlatformClient()
    try:
        with pytest.raises(TransientDeliveryError):
            await client.post_conspect(
                callback_url=callback_url,
                callback_token="t",
                job_id=uuid4(),
                payload={},
            )
        # tenacity stop_after_attempt(3) -> 3 calls total
        assert route.call_count == 3
    finally:
        await client.aclose()


@respx.mock
async def test_post_diagnostic_tags_uses_base_url():
    submission_id = uuid4()
    base = "https://platform.example.com"
    expected_url = f"{base}/v1/submissions/{submission_id}/diagnostic-tags"
    route = respx.post(expected_url).mock(return_value=httpx.Response(204))

    client = PlatformClient(base_url=base, default_token="default-token")
    try:
        await client.post_diagnostic_tags(
            submission_id=submission_id,
            payload={"tags": ["x"]},
        )
    finally:
        await client.aclose()

    assert route.called
    req = route.calls.last.request
    assert req.headers["Idempotency-Key"] == str(submission_id)
    assert req.headers["Authorization"] == "Bearer default-token"


@respx.mock
async def test_post_diagnostic_tags_429_treated_as_transient():
    submission_id = uuid4()
    base = "https://platform.example.com"
    url = f"{base}/v1/submissions/{submission_id}/diagnostic-tags"
    route = respx.post(url).mock(return_value=httpx.Response(429))

    client = PlatformClient(base_url=base, default_token="t")
    try:
        with pytest.raises(TransientDeliveryError):
            await client.post_diagnostic_tags(
                submission_id=submission_id, payload={}
            )
        assert route.call_count == 3  # retried
    finally:
        await client.aclose()
