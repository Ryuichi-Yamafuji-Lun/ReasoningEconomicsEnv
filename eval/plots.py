"""Plotting: agent comparison, budget pacing, tokens-per-question distribution."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def agent_comparison(eval_json_path: str, output_path: str | None = None) -> None:
    """Bar chart: mean accuracy per agent."""
    with open(eval_json_path) as f:
        data = json.load(f)
    summary = data.get("summary", data)
    agents = list(summary.keys())
    means = [summary[a]["accuracy_mean"] for a in agents]
    stds = [summary[a]["accuracy_std"] for a in agents]
    _, ax = plt.subplots()
    x = np.arange(len(agents))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Accuracy")
    ax.set_title("Agent comparison (mean +/- std)")
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def budget_utilization_comparison(
    eval_json_path: str, output_path: str | None = None
) -> None:
    """Bar chart: mean clamped budget utilization per agent."""
    with open(eval_json_path) as f:
        data = json.load(f)
    summary = data.get("summary", data)
    agents = list(summary.keys())
    utils = [summary[a].get("budget_utilization_clamped_mean", 0) for a in agents]
    _, ax = plt.subplots()
    x = np.arange(len(agents))
    ax.bar(x, utils, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Budget utilization")
    ax.set_title("Budget utilization by agent (clamped to [0,1])")
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def budget_pacing(eval_json_path: str, output_path: str | None = None) -> None:
    """Cumulative tokens spent over episode steps per agent (mean across episodes)."""
    with open(eval_json_path) as f:
        data = json.load(f)
    raw = data.get("raw", data)
    _, ax = plt.subplots()
    for agent, runs in raw.items():
        if not runs:
            continue
        # Use mean_tokens_per_question as proxy; detailed tokens_per_step may not be in raw
        token_lists = [r.get("tokens_per_step", []) for r in runs]
        if not any(token_lists):
            continue
        max_steps = max(len(t) for t in token_lists)
        cumulative = np.zeros(max_steps)
        count = np.zeros(max_steps)
        for tl in token_lists:
            for i, t in enumerate(tl):
                cumulative[i] += t
                count[i] += 1
        mean_per_step = np.divide(
            cumulative, count, out=np.zeros_like(cumulative), where=count > 0
        )
        cumsum = np.cumsum(mean_per_step)
        ax.plot(np.arange(1, len(cumsum) + 1), cumsum, label=agent)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative tokens spent")
    ax.set_title("Budget pacing")
    ax.legend()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def tokens_per_question_distribution(
    eval_json_path: str, output_path: str | None = None
) -> None:
    """Histogram of tokens used per question across all agents."""
    with open(eval_json_path) as f:
        data = json.load(f)
    raw = data.get("raw", data)
    _, ax = plt.subplots()
    for agent, runs in raw.items():
        tpq = [r.get("mean_tokens_per_question", 0) for r in runs]
        if tpq:
            ax.hist(tpq, bins=20, alpha=0.5, label=agent)
    ax.set_xlabel("Mean tokens per question")
    ax.set_ylabel("Frequency")
    ax.set_title("Tokens per question distribution")
    ax.legend()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
