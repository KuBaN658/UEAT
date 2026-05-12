"""
HTTP client for the platform's callback endpoints (per ``app/spec/platform.openapi.yaml``).

Two surfaces:
- ``POST /v1/personal-conspect`` -- Flow 2 result callback (full URL per job)
- ``POST /v1/submissions/{id}/diagnostic-tags`` -- Flow 1 result callback

Reliability: ``Idempotency-Key`` on every POST, Bearer auth, tenacity retries
(3× exponential backoff on 5xx/network/408/429). 4xx -> ``PermanentDeliveryError``.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)


class PermanentDeliveryError(RuntimeError):
    """4xx response -- platform rejected the payload, do not retry."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body[:300]}")
        self.status_code = status_code
        self.body = body


class TransientDeliveryError(RuntimeError):
    """5xx, timeout, or network error -- retry exhausted."""


_RETRY = retry(
    retry=retry_if_exception_type(TransientDeliveryError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)


class PlatformClient:
    def __init__(self, base_url: str = "", default_token: str = "", timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_token = default_token
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def post_conspect(
        self,
        *,
        callback_url: str,
        callback_token: str | None,
        job_id: UUID,
        payload: dict[str, Any],
    ) -> None:
        """Deliver Flow 2 conspect. URL comes from the per-job callback config."""
        token = callback_token or self._default_token
        headers = self._headers(idempotency_key=str(job_id), token=token)
        await self._post_with_retry(callback_url, headers=headers, json=payload)

    async def post_diagnostic_tags(
        self, *, submission_id: UUID, payload: dict[str, Any],
    ) -> None:
        """Deliver Flow 1 tags. URL is built from base_url (path is contract-fixed)."""
        if not self._base_url:
            raise RuntimeError("PlatformClient.base_url is empty -- set PLATFORM_BASE_URL")
        url = f"{self._base_url}/v1/submissions/{submission_id}/diagnostic-tags"
        headers = self._headers(idempotency_key=str(submission_id), token=self._default_token)
        await self._post_with_retry(url, headers=headers, json=payload)

    @staticmethod
    def _headers(*, idempotency_key: str, token: str) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Idempotency-Key": idempotency_key}
        if token:
            h["Authorization"] = f"Bearer {token}"
        return h

    @_RETRY
    async def _post_with_retry(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any],
    ) -> None:
        try:
            r = await self._client.post(url, headers=headers, json=json)
        except (httpx.NetworkError, httpx.TimeoutException) as exc:
            log.warning("platform callback network error: %s url=%s", exc, url)
            raise TransientDeliveryError(str(exc)) from exc

        if r.status_code in (200, 201, 202, 204):
            return
        # 4xx is permanent -- except 408 (timeout) and 429 (rate-limit).
        if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
            log.warning(
                "platform callback rejected: %s url=%s body=%s",
                r.status_code, url, r.text[:200],
            )
            raise PermanentDeliveryError(r.status_code, r.text)
        log.warning(
            "platform callback transient: %s url=%s body=%s",
            r.status_code, url, r.text[:200],
        )
        raise TransientDeliveryError(f"HTTP {r.status_code}: {r.text[:200]}")
