"""Episode question record (shared by loaders and env)."""

from dataclasses import dataclass


@dataclass
class Question:
    """Single question in an episode."""

    id: str
    text: str
    answer: str
    problem_type: str
    source: str
