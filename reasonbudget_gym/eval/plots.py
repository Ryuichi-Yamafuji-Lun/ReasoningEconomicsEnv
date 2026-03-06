"""Plotting: budget-accuracy curves, agent comparison, pacing, allocation heatmap."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_cache_entries(cache_path: str) -> dict:
    """Load cache JSON and return entries dict."""
    with open(cache_path) as f:
        data = json.load(f)
    return data.get("entries", data)


def budget_accuracy_curve(cache_path: str, output_path: str | None = None) -> None:
    """Plot accuracy vs budget tier per difficulty (from cache). First result."""
    entries = load_cache_entries(cache_path)
    tiers = [50, 100, 200, 400, 800]
    # We don't have difficulty in cache keys; aggregate over all questions
    tier_correct = {t: 0 for t in tiers}
    tier_total = {t: 0 for t in tiers}
    for qid, qdata in entries.items():
        for t in tiers:
            key = str(t)
            if key not in qdata:
                continue
            tier_total[t] += 1
            if qdata[key].get("was_correct"):
                tier_correct[t] += 1
    accs = [tier_correct[t] / tier_total[t] if tier_total[t] else 0 for t in tiers]
    fig, ax = plt.subplots()
    ax.plot(tiers, accs, "o-")
    ax.set_xlabel("Budget (tokens)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Budget-Accuracy Curve (from response cache)")
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def agent_comparison(eval_json_path: str, output_path: str | None = None) -> None:
    """Bar chart: mean accuracy per agent."""
    with open(eval_json_path) as f:
        data = json.load(f)
    summary = data.get("summary", data)
    agents = list(summary.keys())
    means = [summary[a]["accuracy_mean"] for a in agents]
    stds = [summary[a]["accuracy_std"] for a in agents]
    fig, ax = plt.subplots()
    x = np.arange(len(agents))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Accuracy")
    ax.set_title("Agent comparison (mean ± std)")
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def budget_pacing(eval_json_path: str, output_path: str | None = None) -> None:
    """Cumulative budget spent over episode steps per agent (mean across episodes)."""
    with open(eval_json_path) as f:
        data = json.load(f)
    raw = data.get("raw", data)
    tiers = [50, 100, 200, 400, 800]
    fig, ax = plt.subplots()
    for agent, runs in raw.items():
        if not runs:
            continue
        max_steps = max(len(r["allocations"]) for r in runs)
        cumulative = np.zeros(max_steps)
        for r in runs:
            spent = [tiers[a] for a in r["allocations"]]
            for i, s in enumerate(spent):
                cumulative[i] += s
        cumulative /= len(runs)
        cumsum = np.cumsum(cumulative)
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


def allocation_heatmap(eval_json_path: str, output_path: str | None = None) -> None:
    """Heatmap: tier (y) vs step (x) for one agent, or average allocation per step."""
    with open(eval_json_path) as f:
        data = json.load(f)
    raw = data.get("raw", data)
    tiers = [50, 100, 200, 400, 800]
    # For uniform agent: average allocation index per step
    agent = "uniform"
    if agent not in raw or not raw[agent]:
        agent = list(raw.keys())[0] if raw else None
    if agent is None:
        return
    runs = raw[agent]
    max_steps = max(len(r["allocations"]) for r in runs)
    grid = np.zeros((len(tiers), max_steps))
    for r in runs:
        for step, a in enumerate(r["allocations"]):
            if a < len(tiers):
                grid[a, step] += 1
    grid /= len(runs)
    fig, ax = plt.subplots()
    sns.heatmap(grid, xticklabels=range(1, max_steps + 1), yticklabels=tiers, ax=ax)
    ax.set_xlabel("Step")
    ax.set_ylabel("Budget tier (tokens)")
    ax.set_title(f"Allocation heatmap ({agent})")
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
