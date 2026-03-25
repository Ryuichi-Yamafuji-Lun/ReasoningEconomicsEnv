"""Greedy-max baseline: generate maximal-length template response."""

from env.models import ReasonBudgetObservation


class GreedyMaxBaseline:
    """Use as much of the fair-share budget as possible in the response."""

    def __init__(self, min_tokens: int = 10, max_tokens: int = 800):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def select_action(self, observation: ReasonBudgetObservation, **_context) -> str:
        remaining = float(observation.remaining_budget)
        q_rem = int(observation.questions_remaining)
        if q_rem <= 0:
            return "I cannot answer. \\boxed{0}"
        cap = int(remaining / q_rem)
        cap = max(self.min_tokens, min(self.max_tokens, cap))
        filler_len = max(0, cap * 4 - 80)
        reasoning = (
            "I will solve this problem very carefully with detailed reasoning. "
            + ("detail " * (filler_len // 7))
        )
        return f"{reasoning.strip()}\n\n\\boxed{{0}}"
