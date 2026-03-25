"""Baselines: dummy deterministic and LLM-backed strategies."""

from baselines.dummy import (
    UniformBaseline,
    GreedyMaxBaseline,
    DifficultyOracleBaseline,
)
from baselines.llm import (
    APIChatBaseline,
    LocalVLLMBaseline,
)

__all__ = [
    "UniformBaseline",
    "GreedyMaxBaseline",
    "DifficultyOracleBaseline",
    "APIChatBaseline",
    "LocalVLLMBaseline",
]
