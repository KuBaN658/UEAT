"""
LLM chat clients: Groq, OpenRouter (OpenAI-compatible), and Ollama (local).

The active backend is controlled by ``AppSettings.llm_backend``
(``"groq"`` | ``"openrouter"`` | ``"ollama"``).

Public helpers
--------------
- ``build_chat_client_for_conspect()`` — required client; raises on missing key.
- ``build_chat_client_optional()``     — optional client; returns ``None`` when unavailable.
- ``get_llm_provider_and_model()``     — metadata for the ``/meta/models`` endpoint.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.core.config import get_settings
from app.infrastructure.llm.http_utils import urlopen_with_retries

log = logging.getLogger(__name__)


# ── Protocol / value types ───────────────────────────────────────────


@dataclass(frozen=True)
class ToolCall:
    """A single LLM tool-call request."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LlmResponse:
    """Normalised response from any LLM backend."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()


@runtime_checkable
class ChatClient(Protocol):
    """Minimal unified interface for diagnosis and conspect generation."""

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> LlmResponse: ...


# ── Shared helpers ───────────────────────────────────────────────────


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _response_snippet(text: str, limit: int = 600) -> str:
    compact = text.strip().replace("\r", "\\r")
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}... [truncated {len(compact) - limit} chars]"


def _stream_chunks_to_chat_completion(chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Convert OpenAI SSE or Ollama NDJSON chunks into a minimal chat response."""
    if len(chunks) == 1 and isinstance(chunks[0].get("choices"), list):
        return chunks[0]

    content_parts: list[str] = []
    for chunk in chunks:
        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice.get("delta") if isinstance(choice, dict) else {}
            message = choice.get("message") if isinstance(choice, dict) else {}
            if isinstance(delta, dict):
                content_parts.append(str(delta.get("content") or ""))
            if isinstance(message, dict):
                content_parts.append(str(message.get("content") or ""))
            continue

        message = chunk.get("message")
        if isinstance(message, dict):
            content_parts.append(str(message.get("content") or ""))
        content_parts.append(str(chunk.get("response") or ""))

    content = "".join(content_parts).strip()
    if not content:
        return None
    return {"choices": [{"message": {"content": content}}]}


def _parse_streaming_json_response(text: str) -> dict[str, Any] | None:
    chunks: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line.removeprefix("data:").strip()
        if line == "[DONE]":
            continue
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            chunks.append(parsed)
    return _stream_chunks_to_chat_completion(chunks) if chunks else None


def _decode_json_response(raw_bytes: bytes, log_prefix: str) -> dict[str, Any]:
    text = raw_bytes.decode("utf-8", errors="replace").lstrip("\ufeff")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        streamed = _parse_streaming_json_response(text)
        if streamed is not None:
            return streamed

        start = text.find("{")
        if start != -1:
            try:
                data, _ = json.JSONDecoder().raw_decode(text[start:])
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(data, dict):
                    return data

        message = (
            f"{log_prefix} returned a non-JSON response "
            + f"({e.msg} at line {e.lineno}, column {e.colno}): {_response_snippet(text)!r}"
        )
        raise RuntimeError(message) from e

    if not isinstance(data, dict):
        raise RuntimeError(f"{log_prefix} returned JSON {type(data).__name__}, expected object")
    return data


def _groq_assistant_text(msg: Any) -> str:
    """Use message.content; fall back to reasoning when content is empty (reasoning models)."""
    content = getattr(msg, "content", None) or ""
    if str(content).strip():
        return str(content)
    reasoning = getattr(msg, "reasoning", None)
    if reasoning is None:
        return ""
    return reasoning if isinstance(reasoning, str) else str(reasoning)


def _tool_call_message(call: ToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.arguments, ensure_ascii=False),
        },
    }


def serialize_tool_calls(tool_calls: tuple[ToolCall, ...]) -> list[dict[str, Any]]:
    """Convert normalised tool calls back to OpenAI-compatible message payloads."""
    return [_tool_call_message(call) for call in tool_calls]


def get_llm_backend() -> str:
    """Return the active LLM backend slug (``"groq"``, ``"openrouter"``, or ``"ollama"``)."""
    return get_settings().llm_backend


# ── Groq client ──────────────────────────────────────────────────────


class GroqClient:
    """Groq cloud chat client using the official ``groq`` SDK.

    Args:
        api_key: Override for ``GROQ_API_KEY`` from settings.
        model: Override for ``GROQ_MODEL`` from settings.
        max_retries: Number of SDK-level retries.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        s = get_settings()
        self.api_key = api_key or s.groq_api_key
        self.model = model or s.groq_model
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        from groq import Groq  # type: ignore

        self._client = Groq(api_key=self.api_key, max_retries=max_retries)

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> LlmResponse:
        """Send a two-message (system + user) chat request."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_format="hidden",
        )
        return LlmResponse(text=_groq_assistant_text(resp.choices[0].message))

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> LlmResponse:
        """Send a multi-turn chat request with tool definitions."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_format": "hidden",
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        for idx, call in enumerate(getattr(msg, "tool_calls", None) or []):
            fn = getattr(call, "function", None)
            calls.append(
                ToolCall(
                    id=str(getattr(call, "id", "") or f"tool_call_{idx}"),
                    name=str(getattr(fn, "name", "") if fn else ""),
                    arguments=_parse_tool_arguments(getattr(fn, "arguments", "") if fn else ""),
                )
            )
        return LlmResponse(
            text=_groq_assistant_text(msg),
            tool_calls=tuple(calls),
        )


# ── OpenRouter client ────────────────────────────────────────────────


class OpenRouterClient:
    """OpenRouter OpenAI-compatible API client (uses ``urllib`` only, no extra deps).

    See https://openrouter.ai/docs/api/reference/overview

    Args:
        api_key: Override for ``OPENROUTER_API_KEY``.
        model: Override for ``OPENROUTER_MODEL``.
        base_url: Override for ``OPENROUTER_BASE_URL``.
        timeout_sec: Override for ``OPENROUTER_TIMEOUT``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        s = get_settings()
        self.api_key = (api_key or s.openrouter_api_key).strip()
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.model = model or s.openrouter_model
        self.base_url = (base_url or s.openrouter_base_url).rstrip("/")
        self.timeout_sec = timeout_sec if timeout_sec is not None else s.openrouter_timeout
        self._referer = s.openrouter_http_referer
        self._title = s.openrouter_app_title
        self._max_retries = s.openrouter_max_retries

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self._referer,
            "X-Title": self._title,
        }

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> LlmResponse:
        """Send a two-message chat request."""
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload = json.dumps(body).encode("utf-8")
        url = f"{self.base_url}/chat/completions"

        req = urllib.request.Request(url, data=payload, headers=self._headers(), method="POST")
        raw_bytes = urlopen_with_retries(
            req,
            timeout=self.timeout_sec,
            max_retries=self._max_retries,
            url=url,
            log_prefix="OpenRouter",
        )

        data = _decode_json_response(raw_bytes, "OpenRouter")
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {data!r}")
        msg = choices[0].get("message") or {}
        return LlmResponse(text=str(msg.get("content") or "").strip())

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> LlmResponse:
        """Send a multi-turn chat request with tool definitions."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        payload = json.dumps(body).encode("utf-8")
        url = f"{self.base_url}/chat/completions"

        req = urllib.request.Request(url, data=payload, headers=self._headers(), method="POST")
        raw_bytes = urlopen_with_retries(
            req,
            timeout=self.timeout_sec,
            max_retries=self._max_retries,
            url=url,
            log_prefix="OpenRouter",
        )

        data = _decode_json_response(raw_bytes, "OpenRouter")
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {data!r}")
        msg = choices[0].get("message") or {}
        calls: list[ToolCall] = []
        for idx, call in enumerate(msg.get("tool_calls") or []):
            fn = call.get("function") or {}
            calls.append(
                ToolCall(
                    id=str(call.get("id") or f"tool_call_{idx}"),
                    name=str(fn.get("name") or ""),
                    arguments=_parse_tool_arguments(fn.get("arguments")),
                )
            )
        return LlmResponse(
            text=str(msg.get("content") or "").strip(),
            tool_calls=tuple(calls),
        )


# ── Ollama client ────────────────────────────────────────────────────


def _format_ollama_http_error(base_url: str, code: int, err_body: str) -> str:
    msg = f"Ollama HTTP {code}: {err_body}"
    if "ollama.com" in err_body or "unexpected EOF" in err_body:
        msg += (
            "\n\nHint: Ollama is calling ollama.com (cloud) and the connection dropped. "
            "Run `ollama pull <model>` so inference is local; check VPN/firewall; or "
            "set OLLAMA_BASE_URL=http://127.0.0.1:11434."
        )
    elif "127.0.0.1" in base_url or "localhost" in base_url:
        msg += (
            "\n\nHint: ensure the Ollama service is running and the model is "
            "pulled (`ollama list`)."
        )
    return msg


class OllamaClient:
    """Ollama HTTP API client (``/api/chat``).

    Args:
        base_url: Override for ``OLLAMA_BASE_URL``.
        model: Override for ``OLLAMA_MODEL``.
        timeout_sec: Override for ``OLLAMA_TIMEOUT``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        s = get_settings()
        self.base_url = (base_url or s.ollama_base_url).rstrip("/")
        self.model = model or s.ollama_model
        self.timeout_sec = timeout_sec if timeout_sec is not None else s.ollama_timeout
        self._max_retries = s.ollama_max_retries

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> LlmResponse:
        """Send a two-message chat request.

        ``think=false`` prevents qwen3 from using all ``num_predict`` tokens
        on internal reasoning.
        """
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        payload = json.dumps(body).encode("utf-8")
        url = f"{self.base_url}/api/chat"

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            raw_bytes = urlopen_with_retries(
                req,
                timeout=self.timeout_sec,
                max_retries=self._max_retries,
                url=url,
                log_prefix="Ollama",
            )
        except RuntimeError as e:
            m = re.match(r"Ollama HTTP (\d+):\s*(.*)$", str(e), re.DOTALL)
            if m:
                raise RuntimeError(
                    _format_ollama_http_error(self.base_url, int(m.group(1)), m.group(2)[:800])
                ) from e
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}: {e}. "
                "Start Ollama Desktop / `ollama serve`, or set OLLAMA_BASE_URL."
            ) from e

        data = _decode_json_response(raw_bytes, "Ollama")
        msg = data.get("message") or {}
        text = str(msg.get("content") or "").strip()
        if not text:
            text = str(msg.get("thinking") or "").strip()
        if not text:
            text = str(data.get("response") or "").strip()
        return LlmResponse(text=text)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> LlmResponse:
        """Degrade gracefully: concatenate messages and call ``chat``
        (Ollama does not support OpenAI tool-call spec reliably)."""
        system_parts: list[str] = []
        user_parts: list[str] = []
        for msg in messages:
            role = str(msg.get("role") or "user")
            content = str(msg.get("content") or "")
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(f"{role}: {content}")
        if tools:
            user_parts.append(
                "Доступные инструменты отключены для локального backend; "
                "используй только предоставленный контекст."
            )
        return self.chat(
            system="\n\n".join(system_parts),
            user="\n\n".join(user_parts),
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ── Factory helpers ──────────────────────────────────────────────────


def build_chat_client_for_conspect() -> ChatClient:
    """Return a ``ChatClient`` for conspect generation (required).

    Raises:
        ValueError: If the configured backend is missing an API key.
    """
    backend = get_llm_backend()
    if backend == "ollama":
        return OllamaClient()
    if backend == "groq":
        if not get_settings().groq_api_key.strip():
            raise ValueError("GROQ_API_KEY is not set (LLM_BACKEND=groq)")
        return GroqClient()
    if backend == "openrouter":
        if not get_settings().openrouter_api_key.strip():
            raise ValueError("OPENROUTER_API_KEY is not set (LLM_BACKEND=openrouter)")
        return OpenRouterClient()
    raise ValueError(f"Unknown LLM_BACKEND={backend!r}; use 'groq', 'openrouter', or 'ollama'")


def build_chat_client_for_judge() -> ChatClient:
    """Return a ``ChatClient`` for retrieval evaluation / judge calls.

    If ``JUDGE_MODEL`` is set in the environment, it overrides the model for
    the active backend. Otherwise falls back to ``build_chat_client_for_conspect()``.
    """
    s = get_settings()
    judge_model = s.judge_model.strip()
    if not judge_model:
        return build_chat_client_for_conspect()

    backend = get_llm_backend()
    if backend == "ollama":
        return OllamaClient(model=judge_model)
    if backend == "groq":
        if not s.groq_api_key.strip():
            raise ValueError("GROQ_API_KEY is not set (LLM_BACKEND=groq)")
        return GroqClient(model=judge_model)
    if backend == "openrouter":
        if not s.openrouter_api_key.strip():
            raise ValueError("OPENROUTER_API_KEY is not set (LLM_BACKEND=openrouter)")
        return OpenRouterClient(model=judge_model)
    raise ValueError(f"Unknown LLM_BACKEND={backend!r}; use 'groq', 'openrouter', or 'ollama'")


def build_chat_client_optional() -> ChatClient | None:
    """Return a ``ChatClient`` for mistake diagnosis, or ``None`` if unavailable.

    Returns ``None`` rather than raising so callers can gracefully skip diagnosis.
    """
    backend = get_llm_backend()
    s = get_settings()
    if backend == "ollama":
        return OllamaClient()
    if backend == "groq" and s.groq_api_key.strip():
        try:
            return GroqClient()
        except Exception as exc:
            log.warning("Groq init failed: %s", exc)
            return None
    if backend == "openrouter" and s.openrouter_api_key.strip():
        try:
            return OpenRouterClient()
        except Exception as exc:
            log.warning("OpenRouter init failed: %s", exc)
            return None
    return None


def get_llm_provider_and_model() -> tuple[str, str]:
    """Return ``(provider, model)`` for the ``/meta/models`` endpoint."""
    b = get_llm_backend()
    s = get_settings()
    if b == "ollama":
        return "ollama", s.ollama_model
    if b == "openrouter":
        return "openrouter", s.openrouter_model
    return "groq", s.groq_model
