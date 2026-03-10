"""LinUCB contextual bandit over token-allocation bins."""

import numpy as np

from reasonbudget_gym.env.models import ReasonBudgetObservation


class BanditBaseline:
    """LinUCB over token bins; features = question_embedding + budget state."""

    def __init__(
        self,
        min_tokens: int,
        max_tokens: int,
        num_bins: int = 16,
        embedding_dim: int = 384,
        alpha: float = 1.0,
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.token_bins = np.linspace(min_tokens, max_tokens, num=num_bins).astype(int).tolist()
        self.n_arms = len(self.token_bins)
        self.d = embedding_dim + 3  # embedding + remaining_budget_norm, q_rem_norm, budget_per_remaining_norm
        self.alpha = alpha
        self.A = [np.eye(self.d) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.d) for _ in range(self.n_arms)]

    def _arm_to_tokens(self, arm_idx: int) -> int:
        return int(self.token_bins[arm_idx])

    def _tokens_to_arm(self, token_allocation: int) -> int:
        arr = np.asarray(self.token_bins, dtype=np.int32)
        return int(np.argmin(np.abs(arr - int(token_allocation))))

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
            a_mat = self.A[a]
            ba = self.b[a]
            theta = np.linalg.solve(a_mat, ba)
            xt = x.ravel()
            u = theta @ xt
            bonus = self.alpha * np.sqrt(np.maximum(0, (x.T @ np.linalg.solve(a_mat, x))[0, 0]))
            ucb.append(u + bonus)
        arm = int(np.argmax(ucb))
        return self._arm_to_tokens(arm)

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
        arm = self._tokens_to_arm(action)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x.ravel()
