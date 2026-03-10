"""Greedy-max baseline: allocate max that fits in remaining_budget / questions_remaining."""

from reasonbudget_gym.env.models import ReasonBudgetObservation


class GreedyMaxBaseline:
    """Pick the largest fair-share token allocation within bounds."""

    def __init__(self, min_tokens: int, max_tokens: int):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def select_action(self, observation: ReasonBudgetObservation) -> int:
        remaining = float(observation.remaining_budget)
        q_rem = int(observation.questions_remaining)
        if q_rem <= 0:
            return self.min_tokens
        cap = int(remaining / q_rem)
        return max(self.min_tokens, min(self.max_tokens, cap))
