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
    tier_correct = dict.fromkeys(tiers, 0)
    tier_total = dict.fromkeys(tiers, 0)
    for qid, qdata in entries.items():
        for t in tiers:
            key = str(t)
            if key not in qdata:
                continue
            tier_total[t] += 1
            if qdata[key].get("was_correct"):
                tier_correct[t] += 1
    accs = [tier_correct[t] / tier_total[t] if tier_total[t] else 0 for t in tiers]
    _, ax = plt.subplots()
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
    _, ax = plt.subplots()
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
    _, ax = plt.subplots()
    for agent, runs in raw.items():
        if not runs:
            continue
        max_steps = max(len(r["allocations"]) for r in runs)
        cumulative = np.zeros(max_steps)
        for r in runs:
            spent = [int(a) for a in r["allocations"]]
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
    """Heatmap: token-allocation density (y) vs step (x) for one agent."""
    with open(eval_json_path) as f:
        data = json.load(f)
    raw = data.get("raw", data)
    agent = "uniform"
    if agent not in raw or not raw[agent]:
        agent = list(raw.keys())[0] if raw else None
    if agent is None:
        return
    runs = raw[agent]
    max_steps = max(len(r["allocations"]) for r in runs)
    all_allocs = [int(a) for r in runs for a in r["allocations"]]
    if not all_allocs:
        return
    token_min = min(all_allocs)
    token_max = max(all_allocs)
    n_token_bins = 20
    edges = np.linspace(token_min, token_max, n_token_bins + 1)
    grid = np.zeros((n_token_bins, max_steps))
    for step in range(max_steps):
        vals = [int(r["allocations"][step]) for r in runs if step < len(r["allocations"])]
        if not vals:
            continue
        hist, _ = np.histogram(vals, bins=edges)
        grid[:, step] = hist / len(vals)
    yticklabels = [int((edges[i] + edges[i + 1]) / 2.0) for i in range(n_token_bins)]
    _, ax = plt.subplots()
    sns.heatmap(grid, xticklabels=range(1, max_steps + 1), yticklabels=yticklabels, ax=ax)
    ax.set_xlabel("Step")
    ax.set_ylabel("Token allocation (bin center)")
    ax.set_title(f"Allocation heatmap ({agent})")
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
