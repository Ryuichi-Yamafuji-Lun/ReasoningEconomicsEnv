"""API-backed LLM baseline using an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import requests

from baselines.llm.base import BaseLLMBaseline


class APIChatBaseline(BaseLLMBaseline):
    """Calls an OpenAI-compatible API endpoint for responses."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.0,
    ):
        resolved_base_url = base_url or self.get_required_env("BASELINE_API_BASE_URL")
        resolved_api_key = api_key or self.get_required_env("BASELINE_API_KEY")
        resolved_model = model or self.get_required_env("BASELINE_API_MODEL")
        super().__init__(
            model=resolved_model,
            timeout_s=timeout_s,
            max_retries=max_retries,
            temperature=temperature,
        )
        self.base_url = resolved_base_url.rstrip("/")
        self.api_key = resolved_api_key

    def _complete(self, *, prompt: str, max_new_tokens: int | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if max_new_tokens is not None:
            payload["max_tokens"] = int(max_new_tokens)

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
