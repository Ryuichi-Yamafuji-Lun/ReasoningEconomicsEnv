"""Shared base helpers for LLM-backed baselines."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any

from env.models import ReasonBudgetObservation


class BaseLLMBaseline(ABC):
    """Shared prompt + retry scaffolding for LLM-backed baselines."""

    def __init__(
        self,
        *,
        model: str,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.0,
    ):
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.temperature = temperature

    @staticmethod
    def get_required_env(name: str, default: str | None = None) -> str:
        value = os.getenv(name, default)
        if value is None or not value.strip():
            raise ValueError(f"Missing required env var: {name}")
        return value

    def _build_prompt(
        self,
        observation: ReasonBudgetObservation,
        problem_type: str | None = None,
    ) -> str:
        history_lines = []
        for i, h in enumerate(observation.history or [], 1):
            status = "correct" if h.get("was_correct") else "wrong"
            history_lines.append(
                f"Q{i}: {h.get('question_summary', '')} [{h.get('tokens_used', '?')} tokens, {status}]"
            )
        history_text = "\n".join(history_lines) if history_lines else "(none yet)"
        ptype_line = f"Problem type: {problem_type}\n" if problem_type else ""
        return (
            "You are solving math problems under a shared token budget.\n"
            "Show concise but sufficient reasoning, then end with final answer in \\boxed{}.\n\n"
            f"{ptype_line}"
            f"Remaining budget: {int(observation.remaining_budget)}\n"
            f"Questions remaining: {observation.questions_remaining}\n"
            f"Budget per remaining question: {observation.budget_per_remaining:.0f}\n"
            f"Accuracy so far: {observation.accuracy_so_far:.0%}\n"
            f"History:\n{history_text}\n\n"
            f"Question:\n{observation.question}\n"
        )

    def select_action(
        self,
        observation: ReasonBudgetObservation,
        *,
        problem_type: str | None = None,
        max_new_tokens: int | None = None,
        **_context: Any,
    ) -> str:
        prompt = self._build_prompt(observation, problem_type=problem_type)
        retries = max(0, int(self.max_retries))
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                text = self._complete(prompt=prompt, max_new_tokens=max_new_tokens)
                if text and text.strip():
                    return text
                return "I could not complete reasoning. \\boxed{0}"
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < retries:
                    time.sleep(min(2.0, 0.5 * (attempt + 1)))
        raise RuntimeError(f"LLM baseline request failed after retries: {last_exc}") from last_exc

    @abstractmethod
    def _complete(self, *, prompt: str, max_new_tokens: int | None = None) -> str:
        """Return completion text for prompt."""
