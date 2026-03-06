"""Cached solver: lookup from precomputed response cache (no API/GPU)."""

import json
from pathlib import Path
from typing import Any

from reasonbudget_gym.solver.base_solver import BaseSolver, SolverResult


class CachedSolver(BaseSolver):
    """Instant lookup from cache[question_id][budget_tier] -> SolverResult."""

    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        self._cache: dict[str, dict[int, dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        with open(self.cache_path) as f:
            data = json.load(f)
        if "entries" in data:
            self._cache = data["entries"]
        else:
            self._cache = data

    def _nearest_tier(self, question_id: str, budget_tokens: int) -> int:
        tiers = list(self._cache.get(question_id, {}).keys())
        if not tiers:
            return budget_tokens
        tiers_int = [int(t) for t in tiers]
        best = min(tiers_int, key=lambda t: abs(t - budget_tokens))
        return best

    def solve(self, question: str, ground_truth: str, budget_tokens: int) -> SolverResult:
        """Look up cache by question_id. Requires question_id to be passed via info or we use hash(question)."""
        # Cache is keyed by question_id; when used from env, env must pass question_id in info.
        # For backward compatibility we also support keying by question text hash.
        import hashlib
        qid = getattr(self, "_last_question_id", None) or hashlib.sha256(question.encode()).hexdigest()[:16]
        tier_key = self._nearest_tier(qid, budget_tokens)
        entry = (self._cache.get(qid) or {}).get(str(tier_key)) or (self._cache.get(qid) or {}).get(tier_key)
        if entry is None:
            return SolverResult(
                answer="",
                tokens_used=0,
                was_correct=False,
                response_text="",
            )
        return SolverResult(
            answer=entry.get("answer", ""),
            tokens_used=int(entry.get("tokens_used", 0)),
            was_correct=bool(entry.get("was_correct", False)),
            response_text=entry.get("response_text", ""),
        )

    def set_question_id(self, question_id: str) -> None:
        """Set current question id for next solve() call (used by env)."""
        self._last_question_id = question_id
