"""LinUCB contextual bandit over budget tiers."""

import numpy as np

from reasonbudget_gym.env.models import ReasonBudgetObservation


class BanditBaseline:
    """LinUCB over 5 arms; features = question_embedding + budget state."""

    def __init__(
        self,
        budget_tiers: list[int],
        embedding_dim: int = 384,
        alpha: float = 1.0,
    ):
        self.budget_tiers = budget_tiers
        self.n_arms = len(budget_tiers)
        self.d = embedding_dim + 3  # embedding + remaining_budget_norm, q_rem_norm, budget_per_remaining_norm
        self.alpha = alpha
        self.A = [np.eye(self.d) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.d) for _ in range(self.n_arms)]

    def _feature(self, observation: ReasonBudgetObservation) -> np.ndarray:
        emb = np.asarray(observation.question_embedding, dtype=np.float64).ravel()
        if emb.size != 384:
            emb = np.zeros(384)
        remaining = float(observation.remaining_budget)
        q_rem = float(observation.questions_remaining)
        budget_per = float(observation.budget_per_remaining)
        remaining_norm = min(1.0, remaining / 5000.0) if remaining else 0.0
        q_rem_norm = min(1.0, q_rem / 20.0) if q_rem else 0.0
        budget_per_norm = min(1.0, budget_per / 1000.0) if budget_per else 0.0
        return np.concatenate([
            emb[:384] if len(emb) >= 384 else np.pad(emb, (0, 384 - len(emb))),
            [remaining_norm, q_rem_norm, budget_per_norm],
        ]).astype(np.float64).reshape(-1, 1)

    def select_action(self, observation: ReasonBudgetObservation) -> int:
        x = self._feature(observation)
        x = x.reshape(-1)
        if x.size != self.d:
            x = np.zeros(self.d)
            x[-3:] = [0.5, 0.5, 0.5]
        x = x.reshape(-1, 1)
        ucb = []
        for a in range(self.n_arms):
            Aa = self.A[a]
            ba = self.b[a]
            theta = np.linalg.solve(Aa, ba)
            xt = x.ravel()
            u = theta @ xt
            bonus = self.alpha * np.sqrt(np.maximum(0, (x.T @ np.linalg.solve(Aa, x))[0, 0]))
            ucb.append(u + bonus)
        return int(np.argmax(ucb))

    def update(
        self,
        observation: ReasonBudgetObservation,
        action: int,
        reward: float,
    ) -> None:
        x = self._feature(observation).reshape(-1)
        if x.size != self.d:
            return
        x = x.reshape(-1, 1)
        self.A[action] += x @ x.T
        self.b[action] += reward * x.ravel()
