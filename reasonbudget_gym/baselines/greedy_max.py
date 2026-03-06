"""Greedy-max baseline: allocate max that fits in remaining_budget / questions_remaining."""

import numpy as np

from reasonbudget_gym.env.models import ReasonBudgetObservation


class GreedyMaxBaseline:
    """Pick highest tier that does not exceed remaining_budget / questions_remaining."""

    def __init__(self, budget_tiers: list[int]):
        self.budget_tiers = budget_tiers

    def select_action(self, observation: ReasonBudgetObservation) -> int:
        remaining = float(observation.remaining_budget)
        q_rem = int(observation.questions_remaining)
        if q_rem <= 0:
            return 0
        cap = remaining / q_rem
        for i in range(len(self.budget_tiers) - 1, -1, -1):
            if self.budget_tiers[i] <= cap:
                return i
        return 0
