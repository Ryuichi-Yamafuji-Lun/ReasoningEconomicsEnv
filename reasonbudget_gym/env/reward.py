"""Step reward computation per Section 2.4 of the implementation plan."""


def compute_reward(
    was_correct: bool,
    tokens_allocated: int,
    total_budget: int,
    num_questions: int,
    beta: float = 0.05,
    gamma: float = 0.1,
) -> float:
    """Compute step reward for the question just answered.

    Reward = correctness - cost_penalty + efficiency_bonus.
    """
    correctness = 1.0 if was_correct else -0.1

    fair_share = total_budget / num_questions
    spend_ratio = tokens_allocated / fair_share if fair_share > 0 else 0.0
    cost_penalty = beta * max(0.0, spend_ratio - 1.0)

    efficiency_bonus = (
        gamma * (1.0 - tokens_allocated / total_budget) if was_correct and total_budget > 0 else 0.0
    )

    return correctness - cost_penalty + efficiency_bonus
