"""Environment configuration (EnvConfig) for v2 post-training RL environment."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvConfig:
    """Fully configurable episode and environment parameters."""

    # If True, run full experiment settings. If False, run baby test settings.
    prod: bool = False
    # Baby-test subset window on MetaMathQA: [subset_start_idx, subset_start_idx + subset_size)
    subset_start_idx: int = 0
    subset_size: int = 500
    # Same indexing for NuminaMath-TIR baby runs
    numina_subset_start_idx: int = 0
    numina_subset_size: int = 500

    # Budget policy: hard-cap preserves existing clipping/early-stop behavior.
    hard_cap_mode: bool = True
    # Soft-budget controls (used when hard_cap_mode=False).
    soft_allow_negative_budget: bool = True
    soft_overspend_penalty: float = 0.25

    num_questions: int = 10
    total_budget: Optional[int] = None
    budget_ratio: float = 2.0
    min_tokens: int = 10
    max_tokens: int = 800
    max_tokens_per_step: int = 2048
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    beta: float = 0.05
    gamma: float = 0.1
    lambda_ep: float = 0.5
    target_utilization: float = 0.9
    seed: Optional[int] = None

    def get_total_budget(self) -> int:
        """Compute total_budget from budget_ratio if not set."""
        if self.total_budget is not None:
            return self.total_budget
        avg_tokens = (self.min_tokens + self.max_tokens) / 2.0
        return int(self.budget_ratio * self.num_questions * avg_tokens)
