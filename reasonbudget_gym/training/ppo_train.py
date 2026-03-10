"""Minimal OpenEnv-native training: run N episodes with a baseline policy and log metrics."""

import argparse
from pathlib import Path

from reasonbudget_gym.env import EnvConfig, ReasonBudgetEnvironment
from reasonbudget_gym.env.models import ReasonBudgetAction
from reasonbudget_gym.baselines import UniformBaseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenEnv episodes with a baseline policy")
    parser.add_argument("--output", type=str, default="runs/openenv_train")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--budget_ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = EnvConfig(
        num_questions=args.num_questions,
        budget_ratio=args.budget_ratio,
        seed=args.seed,
    )
    env = ReasonBudgetEnvironment(config=config)
    policy = UniformBaseline(config.min_tokens, config.max_tokens)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    rewards = []
    accuracies = []

    for ep in range(args.num_episodes):
        obs = env.reset(seed=args.seed + ep)
        episode_reward = 0.0
        while not obs.done:
            action = policy.select_action(obs)
            obs = env.step(ReasonBudgetAction(token_allocation=action))
            episode_reward += float(obs.reward or 0.0)
        state = env.state
        rewards.append(episode_reward)
        acc = state.total_correct / state.questions_answered if state.questions_answered else 0.0
        accuracies.append(acc)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{args.num_episodes}  reward={episode_reward:.2f}  accuracy={acc:.2%}")

    mean_reward = sum(rewards) / len(rewards)
    mean_acc = sum(accuracies) / len(accuracies)
    print(f"Done. Mean reward={mean_reward:.2f}  Mean accuracy={mean_acc:.2%}")


if __name__ == "__main__":
    main()
