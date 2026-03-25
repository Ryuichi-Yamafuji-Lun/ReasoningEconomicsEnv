"""Uniform baseline: generate a template response at fair-share verbosity."""

from env.models import ReasonBudgetObservation


class UniformBaseline:
    """Always produce a response with moderate verbosity.

    Returns template text with answer placeholder ``\\boxed{0}``.
    Length is scaled to roughly the fair-share token budget.
    """

    def __init__(self, min_tokens: int = 10, max_tokens: int = 800):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def select_action(self, observation: ReasonBudgetObservation, **_context) -> str:
        remaining = float(observation.remaining_budget)
        q_rem = int(observation.questions_remaining)
        if q_rem <= 0:
            return "I cannot answer. \\boxed{0}"
        target_tokens = int(remaining / q_rem)
        target_tokens = max(self.min_tokens, min(self.max_tokens, target_tokens))
        # ~4 chars per token approximation
        filler_len = max(0, target_tokens * 4 - 80)
        reasoning = "Let me think step by step. " + ("step " * (filler_len // 5))
        return f"{reasoning.strip()}\n\n\\boxed{{0}}"
