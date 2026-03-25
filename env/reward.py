"""Reward computation: per-step reward and episode-level bonus."""


def compute_reward(
    was_correct: bool,
    tokens_used: int,
    total_budget: int,
    num_questions: int,
    beta: float = 0.05,
    gamma: float = 0.1,
    overspend_tokens: int = 0,
    soft_overspend_penalty: float = 0.25,
    hard_cap_mode: bool = True,
) -> float:
    """Compute per-step reward for the question just answered.

    r_step = correctness + efficiency_bonus - cost_penalty

    - correctness: +1.0 if correct, -0.1 if wrong
    - efficiency_bonus: gamma * (1 - tokens_used / fair_share) when correct
    - cost_penalty: beta * max(0, tokens_used / fair_share - 1)
    """
    correctness = 1.0 if was_correct else -0.1

    fair_share = total_budget / num_questions if num_questions > 0 else 1.0
    spend_ratio = tokens_used / fair_share if fair_share > 0 else 0.0
    cost_penalty = beta * max(0.0, spend_ratio - 1.0)

    efficiency_bonus = (
        gamma * (1.0 - spend_ratio) if was_correct and fair_share > 0 else 0.0
    )

    overspend_penalty = 0.0
    if not hard_cap_mode and overspend_tokens > 0 and fair_share > 0:
        overspend_penalty = soft_overspend_penalty * (overspend_tokens / fair_share)

    return correctness - cost_penalty + efficiency_bonus - overspend_penalty


def compute_episode_bonus(
    total_correct: int,
    num_questions: int,
    total_spent: int,
    total_budget: int,
    lambda_ep: float = 0.5,
    target_utilization: float = 0.9,
) -> float:
    """Compute episode-level bonus added to the final step reward.

    r_episode = lambda_ep * (episode_accuracy * budget_utilization_score)

    - episode_accuracy: total_correct / num_questions
    - budget_utilization: 1 - |spent/total_budget - target_util|
    """
    if num_questions <= 0 or total_budget <= 0:
        return 0.0
    episode_accuracy = total_correct / num_questions
    budget_util = 1.0 - abs(total_spent / total_budget - target_utilization)
    return lambda_ep * (episode_accuracy * max(0.0, budget_util))
