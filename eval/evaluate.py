"""Run all baselines over N episodes; collect metrics (OpenEnv)."""

import argparse
import json
from pathlib import Path

import numpy as np

from reasonbudget_gym.env import EnvConfig, ReasonBudgetEnvironment
from reasonbudget_gym.env.models import ReasonBudgetAction
from reasonbudget_gym.baselines import (
    UniformBaseline,
    GreedyMaxBaseline,
    DifficultyOracleBaseline,
    BanditBaseline,
)


def evaluate_baseline(
    name: str,
    env: ReasonBudgetEnvironment,
    baseline,
    n_episodes: int,
    seed: int,
) -> list[dict]:
    results = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        total_reward = 0.0
        allocations = []
        while not obs.done:
            if (
                name == "oracle"
                and hasattr(env, "_questions")
                and env._questions
                and env._step_idx < len(env._questions)
            ):
                diff = env._questions[env._step_idx].difficulty
                action = baseline.select_action(obs, difficulty=diff)
            else:
                action = baseline.select_action(obs)
            obs = env.step(ReasonBudgetAction(token_allocation=action))
            total_reward += float(obs.reward or 0.0)
            allocations.append(action)
            if name == "bandit":
                baseline.update(obs, action, float(obs.reward or 0.0))
        state = env.state
        results.append({
            "total_reward": total_reward,
            "total_correct": state.total_correct,
            "questions_answered": state.questions_answered,
            "accuracy": state.total_correct / state.questions_answered if state.questions_answered else 0,
            "budget_utilization": 1.0 - state.budget_remaining_ratio,
            "allocations": allocations,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    config = EnvConfig(num_questions=10, budget_ratio=2.0, seed=args.seed)
    env = ReasonBudgetEnvironment(config=config)

    min_tokens = config.min_tokens
    max_tokens = config.max_tokens
    all_results = {}

    for name, baseline in [
        ("uniform", UniformBaseline(min_tokens, max_tokens)),
        ("greedy_max", GreedyMaxBaseline(min_tokens, max_tokens)),
        ("oracle", DifficultyOracleBaseline(min_tokens, max_tokens)),
        ("bandit", BanditBaseline(min_tokens, max_tokens, embedding_dim=384)),
    ]:
        res = evaluate_baseline(name, env, baseline, args.n_episodes, args.seed)
        all_results[name] = res

    summary = {}
    for agent, runs in all_results.items():
        accs = [r["accuracy"] for r in runs]
        rewards = [r["total_reward"] for r in runs]
        summary[agent] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
        }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "raw": dict(all_results)}, f, indent=1)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
