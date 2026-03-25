"""Local/self-hosted LLM baseline via OpenAI-compatible endpoint."""

from __future__ import annotations

from baselines.llm.api_chat import APIChatBaseline


class LocalVLLMBaseline(APIChatBaseline):
    """Connect to local vLLM/OpenAI-compatible server for inference."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.0,
    ):
        # Many local servers ignore API keys but still require the header.
        super().__init__(
            base_url=base_url or self.get_required_env(
                "BASELINE_LOCAL_BASE_URL",
                "http://127.0.0.1:8001/v1",
            ),
            api_key=self.get_required_env("BASELINE_LOCAL_API_KEY", "local"),
            model=model or self.get_required_env("BASELINE_LOCAL_MODEL"),
            timeout_s=timeout_s,
            max_retries=max_retries,
            temperature=temperature,
        )
