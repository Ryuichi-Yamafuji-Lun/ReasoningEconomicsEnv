"""Uniform baseline: a_t = tier closest to remaining_budget / questions_remaining."""

import numpy as np

from reasonbudget_gym.env.models import ReasonBudgetObservation


class UniformBaseline:
    """Always allocate fair share: pick tier closest to remaining_budget / questions_remaining."""

    def __init__(self, budget_tiers: list[int]):
        self.budget_tiers = budget_tiers

    def select_action(self, observation: ReasonBudgetObservation) -> int:
        remaining = float(observation.remaining_budget)
        q_rem = int(observation.questions_remaining)
        if q_rem <= 0:
            return 0
        target = remaining / q_rem
        idx = int(np.argmin(np.abs(np.array(self.budget_tiers) - target)))
        return min(idx, len(self.budget_tiers) - 1)
