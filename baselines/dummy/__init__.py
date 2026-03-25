"""Deterministic baselines for smoke testing env/reward behavior."""

from baselines.dummy.uniform import UniformBaseline
from baselines.dummy.greedy_max import GreedyMaxBaseline
from baselines.dummy.difficulty_oracle import DifficultyOracleBaseline

__all__ = [
    "UniformBaseline",
    "GreedyMaxBaseline",
    "DifficultyOracleBaseline",
]
