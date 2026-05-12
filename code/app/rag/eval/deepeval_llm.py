"""
DeepEval LLM adapter: wraps the project's ChatClient as a DeepEvalBaseLLM.

This lets G-Eval metrics use the same judge backend (Groq / OpenRouter / Ollama)
that is already configured via LLM_BACKEND / JUDGE_MODEL env vars.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.infrastructure.llm.clients import ChatClient, build_chat_client_for_judge
from deepeval.models import DeepEvalBaseLLM


class JudgeLLM(DeepEvalBaseLLM):
    """Thin wrapper around ChatClient for use in DeepEval G-Eval metrics.

    Args:
        client: An already-built ChatClient. When *None*, creates one via
                ``build_chat_client_for_judge()``.
        model_name: Display name returned by ``get_model_name()``.
    """

    def __init__(
        self,
        client: ChatClient | None = None,
        model_name: str = "judge",
    ) -> None:
        self._client: ChatClient = client or build_chat_client_for_judge()
        self._model_name = model_name
        # Skip the parent __init__ so it doesn't call load_model immediately
        # with auto-detected name. We manage state ourselves.

    def load_model(self, *args: Any, **kwargs: Any) -> "JudgeLLM":
        return self

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        """Synchronous generation used by DeepEval when async_mode=False."""
        resp = self._client.chat(
            system="",
            user=prompt,
            temperature=0.0,
            max_tokens=4096,
        )
        return resp.text

    async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        """Async generation: delegates to the sync method via thread executor."""
        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self._model_name
