"""Dataset-specific loaders: MetaMathQA (by `type`) and NuminaMath-TIR."""

from __future__ import annotations

import re
from typing import Optional

from datasets import load_dataset

from data.question import Question

# Canonical MetaMathQA `type` values (see dataset card).
METAMATHQA_PROBLEM_TYPES: tuple[str, ...] = (
    "MATH_AnsAug",
    "GSM_Rephrased",
    "GSM_SV",
    "GSM_FOBAR",
    "GSM_AnsAug",
    "MATH_FOBAR",
    "MATH_Rephrased",
    "MATH_SV",
)

METAMATHQA_TYPE_SET = frozenset(METAMATHQA_PROBLEM_TYPES)

# Single label for NuminaMath-TIR rows (mixed into episodes alongside MetaMath types).
NUMINA_PROBLEM_TYPE = "NuminaMath_TIR"


def _extract_boxed(text: str) -> Optional[str]:
    r"""Extract content of last \boxed{...} in text, or None."""
    if not text:
        return None
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def _answer_from_solution(solution: str) -> str:
    answer = _extract_boxed(solution)
    if answer:
        return answer
    return solution.strip().split("\n")[-1] if solution else ""


def _canonical_metamath_type(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip()
    if s in METAMATHQA_TYPE_SET:
        return s
    upper = s.upper()
    for k in METAMATHQA_PROBLEM_TYPES:
        if k.upper() == upper:
            return k
    return None


def _metamath_split_name(
    prod: bool,
    subset_start_idx: int,
    subset_size: int,
) -> str:
    if prod:
        return "train"
    start = max(0, int(subset_start_idx))
    end = start + max(1, int(subset_size))
    return f"train[{start}:{end}]"


def load_metamathqa_by_type(
    *,
    prod: bool,
    subset_start_idx: int,
    subset_size: int,
) -> dict[str, list[Question]]:
    """Load MetaMathQA and bucket rows strictly by `type` (known types only)."""
    split_name = _metamath_split_name(prod, subset_start_idx, subset_size)
    ds = load_dataset("meta-math/MetaMathQA", "default", split=split_name)
    pools: dict[str, list[Question]] = {t: [] for t in METAMATHQA_PROBLEM_TYPES}
    for i, row in enumerate(ds):
        ptype = _canonical_metamath_type(row.get("type"))
        if ptype is None:
            continue
        qid = f"metamath_{i}"
        query = row.get("query") or row.get("question", "")
        response = row.get("response", "")
        answer = _answer_from_solution(response)
        pools[ptype].append(
            Question(
                id=qid,
                text=query,
                answer=answer,
                problem_type=ptype,
                source="metamath",
            )
        )
    return pools


def _numina_split_name(
    prod: bool,
    subset_start_idx: int,
    subset_size: int,
) -> str:
    if prod:
        return "train"
    start = max(0, int(subset_start_idx))
    end = start + max(1, int(subset_size))
    return f"train[{start}:{end}]"


def load_numinamath_tir(
    *,
    prod: bool,
    subset_start_idx: int,
    subset_size: int,
) -> list[Question]:
    """Load NuminaMath-TIR; one problem_type label for all rows."""
    split_name = _numina_split_name(prod, subset_start_idx, subset_size)
    ds = load_dataset("AI-MO/NuminaMath-TIR", split=split_name)
    out: list[Question] = []
    for i, row in enumerate(ds):
        problem = row.get("problem", "") or row.get("question", "")
        solution = row.get("solution", "") or row.get("answer", "")
        answer = _answer_from_solution(solution)
        if not problem or not answer:
            continue
        out.append(
            Question(
                id=f"numina_{i}",
                text=problem,
                answer=answer,
                problem_type=NUMINA_PROBLEM_TYPE,
                source="numina_math_tir",
            )
        )
    return out
