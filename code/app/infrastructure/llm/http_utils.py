"""
Shared urllib HTTP retry helper with exponential back-off.

Used by both the LLM and embedding clients to avoid duplicating retry logic.
"""

from __future__ import annotations

import logging
import socket
import time
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


def http_retryable_status(code: int | None) -> bool:
    """Return True for HTTP status codes that are safe to retry."""
    return code in (429, 502, 503, 504)


def urlopen_with_retries(
    req: urllib.request.Request,
    *,
    timeout: float,
    max_retries: int,
    url: str,
    log_prefix: str = "HTTP",
) -> bytes:
    """Perform ``urllib.request.urlopen`` with retries on transient errors.

    Args:
        req: Pre-built ``urllib.request.Request`` object.
        timeout: Socket timeout in seconds.
        max_retries: Number of additional attempts after the first failure.
        url: URL string used only for error messages.
        log_prefix: Prefix for log messages (e.g. ``"Groq"``).

    Returns:
        Response body as bytes.

    Raises:
        RuntimeError: On non-retryable HTTP error or exhausted retries.
    """
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")[:1200]
            if http_retryable_status(e.code) and attempt < max_retries:
                delay = min(2.0**attempt, 30.0)
                log.warning(
                    "%s HTTP %s (attempt %s/%s), retry in %.1fs: %s",
                    log_prefix,
                    e.code,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    err_body[:200],
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"{log_prefix} HTTP {e.code}: {err_body}") from e
        except (OSError, socket.timeout) as e:
            if attempt < max_retries:
                delay = min(2.0**attempt, 30.0)
                log.warning(
                    "%s connection error (attempt %s): %s, retry in %.1fs",
                    log_prefix,
                    attempt + 1,
                    e,
                    delay,
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"Cannot reach {url}: {e}") from e
    raise RuntimeError(f"Exhausted retries for {url}")  # should never reach here
