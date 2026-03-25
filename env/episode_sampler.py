"""Episode sampling: even mix across MetaMathQA problem types + NuminaMath-TIR."""

from __future__ import annotations

import random
from typing import Optional

from data.loaders import (
    METAMATHQA_PROBLEM_TYPES,
    NUMINA_PROBLEM_TYPE,
    load_metamathqa_by_type,
    load_numinamath_tir,
)
from data.question import Question

ALL_MIX_TYPES: tuple[str, ...] = (*METAMATHQA_PROBLEM_TYPES, NUMINA_PROBLEM_TYPE)


def _even_type_counts(
    num_questions: int,
    active_types: list[str],
    rng: random.Random,
):
    """Split `num_questions` across `active_types` as evenly as possible."""
    k = len(active_types)
    if k == 0 or num_questions <= 0:
        return []
    types = list(active_types)
    rng.shuffle(types)
    if num_questions <= k:
        return [(t, 1) for t in rng.sample(types, num_questions)]
    base, rem = divmod(num_questions, k)
    out: list[tuple[str, int]] = []
    for i, t in enumerate(types):
        c = base + (1 if i < rem else 0)
        out.append((t, c))
    return out


class EpisodeSampler:
    """Samples episodes with an even mix of MetaMathQA `type` values + NuminaMath-TIR."""

    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        prod: bool = False,
        subset_start_idx: int = 0,
        subset_size: int = 500,
        numina_subset_start_idx: int = 0,
        numina_subset_size: int = 500,
    ):
        self._seed = seed
        self._prod = prod
        self._subset_start_idx = max(0, int(subset_start_idx))
        self._subset_size = max(1, int(subset_size))
        self._numina_subset_start_idx = max(0, int(numina_subset_start_idx))
        self._numina_subset_size = max(1, int(numina_subset_size))
        self._pools: Optional[dict[str, list[Question]]] = None

    def _fallback_pools(self):
        """Tiny offline pools (one row per type) when HF load fails."""
        fb: dict[str, list[Question]] = {
            "MATH_AnsAug": [
                Question("fb_ma", "Compute 1+1.", "2", "MATH_AnsAug", "fallback"),
            ],
            "GSM_Rephrased": [
                Question("fb_gr", "What is 3*4?", "12", "GSM_Rephrased", "fallback"),
            ],
            "GSM_SV": [
                Question("fb_gsv", "What is 10-3?", "7", "GSM_SV", "fallback"),
            ],
            "GSM_FOBAR": [
                Question("fb_gf", "What is 8/2?", "4", "GSM_FOBAR", "fallback"),
            ],
            "GSM_AnsAug": [
                Question("fb_ga", "What is 5+6?", "11", "GSM_AnsAug", "fallback"),
            ],
            "MATH_FOBAR": [
                Question("fb_mf", "What is 9^2?", "81", "MATH_FOBAR", "fallback"),
            ],
            "MATH_Rephrased": [
                Question("fb_mr", "Solve x if x+2=5.", "3", "MATH_Rephrased", "fallback"),
            ],
            "MATH_SV": [
                Question("fb_ms", "What is sqrt(16)?", "4", "MATH_SV", "fallback"),
            ],
            NUMINA_PROBLEM_TYPE: [
                Question("fb_nm", "What is 2^3?", "8", NUMINA_PROBLEM_TYPE, "fallback"),
            ],
        }
        return fb

    def _load_pools(self):
        if self._pools is not None:
            return self._pools
        try:
            meta = load_metamathqa_by_type(
                prod=self._prod,
                subset_start_idx=self._subset_start_idx,
                subset_size=self._subset_size,
            )
            numina = load_numinamath_tir(
                prod=self._prod,
                subset_start_idx=self._numina_subset_start_idx,
                subset_size=self._numina_subset_size,
            )
            pools: dict[str, list[Question]] = {}
            for t in METAMATHQA_PROBLEM_TYPES:
                if meta.get(t):
                    pools[t] = list(meta[t])
            if numina:
                pools[NUMINA_PROBLEM_TYPE] = numina
            self._pools = pools if pools else self._fallback_pools()
        except Exception:
            self._pools = self._fallback_pools()
        return self._pools

    def sample_episode(
        self,
        num_questions: int,
        seed: Optional[int] = None,
    ):
        """Sample `num_questions` with an even split across available problem types."""
        rng = random.Random(seed if seed is not None else self._seed)
        pools = self._load_pools()
        active = [t for t in ALL_MIX_TYPES if pools.get(t)]
        if not active:
            return []

        counts = _even_type_counts(num_questions, active, rng)
        chosen: list[Question] = []
        for ptype, n in counts:
            pool = pools[ptype]
            if not pool or n <= 0:
                continue
            if n <= len(pool):
                chosen.extend(rng.sample(pool, n))
            else:
                chosen.extend(rng.sample(pool, len(pool)))
                chosen.extend(rng.choices(pool, k=n - len(pool)))
        rng.shuffle(chosen)
        return chosen[:num_questions]
