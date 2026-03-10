"""Episode sampler: load MetaMathQA, difficulty stratification, and optional MATH L4-L5 supplement."""

import re
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset

from reasonbudget_gym.data.difficulty_labels import classify_question


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content of last \\boxed{...} in text, or None."""
    if not text:
        return None
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


@dataclass
class Question:
    """Single question in an episode."""

    id: str
    text: str
    answer: str
    difficulty: str
    source: str


class EpisodeSampler:
    """Samples episodes from MetaMathQA with optional MATH L4-L5 supplement."""

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._meta_math_by_difficulty: Optional[dict[str, list[Question]]] = None
        self._math_l4_l5: Optional[list[Question]] = None

    def _load_meta_math(self) -> dict[str, list[Question]]:
        if self._meta_math_by_difficulty is not None:
            return self._meta_math_by_difficulty
        ds = load_dataset("meta-math/MetaMathQA", "default", split="train", trust_remote_code=True)
        by_diff: dict[str, list[Question]] = {
            "gsm8k": [],
            "math_l1_l2": [],
            "math_l3": [],
            "math_l4_l5": [],
        }
        for i, row in enumerate(ds):
            qid = f"metamath_{i}"
            query = row.get("query") or row.get("question", "")
            response = row.get("response", "")
            answer = _extract_boxed(response)
            if not answer:
                answer = response.strip().split("\n")[-1] if response else ""
            diff = classify_question(dict(row))
            if diff not in by_diff:
                by_diff[diff] = []
            by_diff[diff].append(
                Question(id=qid, text=query, answer=answer, difficulty=diff, source="metamath")
            )
        self._meta_math_by_difficulty = by_diff
        return self._meta_math_by_difficulty

    def _load_math_l4_l5(self) -> list[Question]:
        if self._math_l4_l5 is not None:
            return self._math_l4_l5
        try:
            ds = load_dataset(
                "EleutherAI/hendrycks_math",
                "all",
                split="train",
                trust_remote_code=True,
            )
        except Exception:
            self._math_l4_l5 = []
            return self._math_l4_l5
        out = []
        for i, row in enumerate(ds):
            level = row.get("level")
            if level is None:
                continue
            try:
                l = int(level)
            except (TypeError, ValueError):
                continue
            if l < 4:
                continue
            problem = row.get("problem", "")
            solution = row.get("solution", "")
            answer = _extract_boxed(solution)
            if not answer:
                answer = solution.strip().split("\n")[-1] if solution else ""
            diff = "math_l4_l5"
            out.append(
                Question(
                    id=f"math_l45_{i}",
                    text=problem,
                    answer=answer,
                    difficulty=diff,
                    source="hendrycks_math",
                )
            )
        self._math_l4_l5 = out
        return self._math_l4_l5

    def sample_episode(
        self,
        num_questions: int,
        difficulty_mix: dict[str, float],
        seed: Optional[int] = None,
    ) -> list[Question]:
        """Sample num_questions questions according to difficulty_mix.

        difficulty_mix: e.g. {'gsm8k': 0.3, 'math_l1_l2': 0.2, 'math_l3': 0.2, 'math_l4_l5': 0.3}
        """
        import random

        rng = random.Random(seed if seed is not None else self._seed)
        by_diff = self._load_meta_math()
        math_l45 = self._load_math_l4_l5()
        if math_l45:
            by_diff["math_l4_l5"] = list(by_diff.get("math_l4_l5", [])) + math_l45

        counts = {}
        for diff, frac in difficulty_mix.items():
            if frac <= 0:
                continue
            n = max(0, int(round(num_questions * frac)))
            if n > 0 and diff in by_diff and by_diff[diff]:
                counts[diff] = n
        total = sum(counts.values())
        if total < num_questions:
            for diff in difficulty_mix:
                if diff not in counts and diff in by_diff and by_diff[diff]:
                    counts[diff] = counts.get(diff, 0) + 1
                    total += 1
                    if total >= num_questions:
                        break
        if total > num_questions:
            for diff in list(counts.keys()):
                if counts[diff] > 0 and total > num_questions:
                    counts[diff] -= 1
                    total -= 1

        chosen = []
        for diff, n in counts.items():
            pool = by_diff.get(diff, [])
            if not pool or n <= 0:
                continue
            chosen.extend(rng.sample(pool, min(n, len(pool))))
        rng.shuffle(chosen)
        return chosen[:num_questions]
