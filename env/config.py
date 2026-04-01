"""Environment configuration (EnvConfig) for v2 post-training RL environment."""

import os
import warnings
from dataclasses import dataclass, replace
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
        """Compute total_budget from budget_ratio if not set.

        NOTE: This formula uses min_tokens/max_tokens as abstract units. It does
        NOT tokenize any content with ``tokenizer_name``. When a policy tokenizer
        is active, prefer ``ReasonBudgetEnvironment._compute_tokenizer_native_budget``
        (called automatically on ``reset`` when ``tokenizer_name`` is provided).
        """
        if self.total_budget is not None:
            return self.total_budget
        avg_tokens = (self.min_tokens + self.max_tokens) / 2.0
        derived = int(self.budget_ratio * self.num_questions * avg_tokens)
        if self.tokenizer_name and self.tokenizer_name != "Qwen/Qwen2.5-0.5B-Instruct":
            warnings.warn(
                f"EnvConfig.tokenizer_name is set to {self.tokenizer_name!r} but "
                f"total_budget is derived from min_tokens/max_tokens ({derived} abstract "
                "units), not from the tokenizer. The budget cap and per-step token "
                "counts may be in different unit systems. Pass total_budget explicitly "
                "or use tokenizer_name on reset() for tokenizer-native budgets.",
                UserWarning,
                stacklevel=2,
            )
        return derived


def env_config_for_server() -> EnvConfig:
    """Defaults for new OpenEnv sessions, with optional Docker/deploy overrides.

    ``REE_DEFAULT_TOKENIZER_NAME``: Hugging Face model id used when the client
    does not send ``tokenizer_name`` on reset (post-training clients should send it).
    """
    cfg = EnvConfig()
    tok = os.environ.get("REE_DEFAULT_TOKENIZER_NAME", "").strip()
    if tok:
        cfg = replace(cfg, tokenizer_name=tok)
    prod = os.environ.get("REE_PROD", "").strip().lower()
    if prod in ("1", "true", "yes"):
        cfg = replace(cfg, prod=True)
    return cfg
