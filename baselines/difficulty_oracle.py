"""Difficulty oracle: knows true difficulty, allocates proportionally (upper-bound heuristic)."""

from reasonbudget_gym.env.models import ReasonBudgetObservation


class DifficultyOracleBaseline:
    """Maps difficulty to direct token allocations."""

    def __init__(self, min_tokens: int, max_tokens: int):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def _clamp(self, tokens: int):
        return max(self.min_tokens, min(self.max_tokens, tokens))

    def select_action(
        self,
        _observation: ReasonBudgetObservation,
        difficulty: str | None = None,
    ):
        if difficulty is None:
            return self._clamp(300)  # default medium
        if difficulty == "gsm8k":
            return self._clamp(50)
        if difficulty == "math_l1_l2":
            return self._clamp(150)
        if difficulty == "math_l3":
            return self._clamp(300)
        if difficulty == "math_l4_l5":
            return self._clamp(700)
        return self._clamp(300)
