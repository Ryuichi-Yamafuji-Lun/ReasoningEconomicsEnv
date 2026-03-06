"""Environment configuration (EnvConfig) per Section 2.1 of the implementation plan."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    """Fully configurable episode and environment parameters."""

    num_questions: int = 10
    total_budget: Optional[int] = None
    budget_ratio: float = 2.0
    difficulty_mix: dict = field(
        default_factory=lambda: {
            "gsm8k": 0.3,
            "math_l1_l2": 0.2,
            "math_l3": 0.2,
            "math_l4_l5": 0.3,
        }
    )
    action_space: str = "discrete"
    budget_tiers: list = field(default_factory=lambda: [50, 100, 200, 400, 800])
    solver_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    use_cache: bool = True
    cache_path: str = "data/response_cache.json"
    embedding_model: str = "all-MiniLM-L6-v2"
    beta: float = 0.05
    gamma: float = 0.1
    seed: Optional[int] = None

    def get_total_budget(self) -> int:
        """Compute total_budget from budget_ratio if not set."""
        if self.total_budget is not None:
            return self.total_budget
        avg_tokens = sum(self.budget_tiers) / len(self.budget_tiers)
        return int(self.budget_ratio * self.num_questions * avg_tokens)
