"""Run all baselines over N episodes; collect metrics (v2: text-response actions)."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from env import EnvConfig, ReasonBudgetEnvironment
from env.models import ReasonBudgetAction
from baselines import (
    APIChatBaseline,
    DifficultyOracleBaseline,
    GreedyMaxBaseline,
    LocalVLLMBaseline,
    UniformBaseline,
)


def _select_response(
    env: ReasonBudgetEnvironment,
    obs,
    baseline: Any,
    max_new_tokens: int | None = None,
) -> str:
    ptype = None
    if hasattr(env, "_questions") and env._questions and env._step_idx < len(env._questions):
        ptype = env._questions[env._step_idx].problem_type
    return baseline.select_action(
        obs,
        problem_type=ptype,
        max_new_tokens=max_new_tokens,
    )


def _parse_csv_names(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _build_baselines(
    args: argparse.Namespace,
    *,
    min_tokens: int,
    max_tokens: int,
) -> dict[str, Any]:
    baselines: dict[str, Any] = {
        "uniform": UniformBaseline(min_tokens, max_tokens),
        "greedy_max": GreedyMaxBaseline(min_tokens, max_tokens),
        "oracle": DifficultyOracleBaseline(min_tokens, max_tokens),
    }

    requested = _parse_csv_names(args.baselines)
    needs_llm = args.include_llm or any(name.startswith("llm_") for name in requested)
    if needs_llm:
        baselines["llm_api"] = APIChatBaseline(
            timeout_s=args.llm_timeout_s,
            max_retries=args.llm_max_retries,
            temperature=args.llm_temperature,
        )
        baselines["llm_local"] = LocalVLLMBaseline(
            timeout_s=args.llm_timeout_s,
            max_retries=args.llm_max_retries,
            temperature=args.llm_temperature,
        )

    if not requested:
        return baselines

    unknown = [name for name in requested if name not in baselines]
    if unknown:
        available = ", ".join(sorted(baselines.keys()))
        raise ValueError(
            f"Unknown baselines: {unknown}. Available choices: {available}"
        )

    return {name: baselines[name] for name in requested}


def evaluate_baseline(
    env: ReasonBudgetEnvironment,
    baseline,
    n_episodes: int,
    seed: int,
    llm_max_new_tokens: int | None = None,
) -> list[dict]:
    results = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        total_reward = 0.0
        tokens_per_step: list[int] = []
        while not obs.done:
            response = _select_response(
                env,
                obs,
                baseline,
                max_new_tokens=llm_max_new_tokens,
            )
            obs = env.step(ReasonBudgetAction(response=response))
            total_reward += float(obs.reward or 0.0)
            if obs.history:
                tokens_per_step.append(obs.history[-1].get("tokens_used", 0))
        state = env.state
        spent = int(state.spent_budget)
        budget = int(state.total_budget)
        budget_util = (spent / budget) if budget > 0 else 0.0
        overspend_tokens = max(0, spent - budget)
        results.append({
            "total_reward": total_reward,
            "total_correct": state.total_correct,
            "questions_answered": state.questions_answered,
            "accuracy": (
                state.total_correct / state.questions_answered
                if state.questions_answered
                else 0
            ),
            "budget_utilization": budget_util,
            "budget_utilization_clamped": min(1.0, budget_util),
            "overspend_tokens": overspend_tokens,
            "went_over_budget": overspend_tokens > 0,
            "tokens_per_step": tokens_per_step,
            "mean_tokens_per_question": (
                float(np.mean(tokens_per_step)) if tokens_per_step else 0.0
            ),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument(
        "--include_llm",
        action="store_true",
        help="Include llm_api and llm_local baselines (requires env vars).",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="",
        help="Comma-separated baseline names to run (default: all enabled).",
    )
    parser.add_argument("--llm_max_new_tokens", type=int, default=512)
    parser.add_argument("--llm_timeout_s", type=float, default=30.0)
    parser.add_argument("--llm_max_retries", type=int, default=2)
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    args = parser.parse_args()

    config = EnvConfig(num_questions=10, budget_ratio=2.0, seed=args.seed)
    env = ReasonBudgetEnvironment(config=config)

    min_tokens = config.min_tokens
    max_tokens = config.max_tokens
    all_results: dict[str, list[dict]] = {}
    selected_baselines = _build_baselines(
        args,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )

    for name, baseline in selected_baselines.items():
        res = evaluate_baseline(
            env,
            baseline,
            args.n_episodes,
            args.seed,
            llm_max_new_tokens=args.llm_max_new_tokens,
        )
        all_results[name] = res

    summary = {}
    for agent, runs in all_results.items():
        accs = [r["accuracy"] for r in runs]
        rewards = [r["total_reward"] for r in runs]
        budget_utils = [r["budget_utilization"] for r in runs]
        budget_utils_clamped = [r.get("budget_utilization_clamped", 0.0) for r in runs]
        overspends = [r.get("overspend_tokens", 0) for r in runs]
        went_over = [1.0 if r.get("went_over_budget", False) else 0.0 for r in runs]
        mean_tpq = [r["mean_tokens_per_question"] for r in runs]
        summary[agent] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "budget_utilization_mean": float(np.mean(budget_utils)),
            "budget_utilization_clamped_mean": float(np.mean(budget_utils_clamped)),
            "overspend_tokens_mean": float(np.mean(overspends)),
            "episodes_over_budget_rate": float(np.mean(went_over)),
            "mean_tokens_per_question": float(np.mean(mean_tpq)),
        }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip tokens_per_step from raw output to keep file manageable
    raw_compact = {}
    for agent, runs in all_results.items():
        raw_compact[agent] = [
            {k: v for k, v in r.items() if k != "tokens_per_step"} for r in runs
        ]
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "raw": raw_compact}, f, indent=1)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
