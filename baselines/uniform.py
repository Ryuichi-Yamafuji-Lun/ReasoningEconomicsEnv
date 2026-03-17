"""Uniform baseline: allocate fair-share tokens directly."""

from env.models import ReasonBudgetObservation


class UniformBaseline:
    """Always allocate fair share clamped to configured token bounds."""

    def __init__(self, min_tokens: int, max_tokens: int):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def select_action(self, observation: ReasonBudgetObservation) -> int:
        remaining = float(observation.remaining_budget)
        q_rem = int(observation.questions_remaining)
        if q_rem <= 0:
            return self.min_tokens
        target = int(remaining / q_rem)
        return max(self.min_tokens, min(self.max_tokens, target))
