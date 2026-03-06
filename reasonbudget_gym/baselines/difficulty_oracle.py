"""Difficulty oracle: knows true difficulty, allocates proportionally (upper-bound heuristic)."""

from reasonbudget_gym.env.models import ReasonBudgetObservation


class DifficultyOracleBaseline:
    """Maps difficulty to tier: gsm8k->0, math_l1_l2->1, math_l3->2, math_l4_l5->3 or 4."""

    def __init__(self, budget_tiers: list[int]):
        self.budget_tiers = budget_tiers

    def select_action(
        self,
        observation: ReasonBudgetObservation,
        difficulty: str | None = None,
    ) -> int:
        if difficulty is None:
            return 2  # default medium
        if difficulty == "gsm8k":
            return 0
        if difficulty == "math_l1_l2":
            return 1
        if difficulty == "math_l3":
            return 2
        if difficulty == "math_l4_l5":
            return 4 if len(self.budget_tiers) > 4 else 3
        return 2
