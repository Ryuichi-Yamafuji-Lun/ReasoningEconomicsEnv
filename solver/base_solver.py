"""Base solver interface, SolverResult, and answer grading."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sympy import simplify, sympify

try:
    from sympy.parsing.latex import parse_latex
except ImportError:
    parse_latex = None


@dataclass
class SolverResult:
    """Result of solving one question with a given budget."""

    answer: str
    tokens_used: int
    was_correct: bool
    response_text: str


def extract_boxed_answer(text: str) -> str:
    """Extract content of last \\boxed{...} in text, or empty string."""
    if not text:
        return ""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    lines = text.strip().split("\n")
    return lines[-1].strip() if lines else ""


def _normalize_math(s: str) -> str:
    """Normalize for comparison: strip and collapse whitespace."""
    return " ".join(s.strip().split())


def _try_sympy_equal(a: str, b: str) -> bool:
    """Compare two math expressions symbolically (including LaTeX-style)."""
    a, b = _normalize_math(a), _normalize_math(b)
    if a == b:
        return True
    for expr_a, expr_b in [(a, b), (a.replace("\\frac", "frac"), b)]:
        try:
            # Try standard sympify (handles 1/2, 0.5, etc.)
            va = simplify(sympify(expr_a))
            vb = simplify(sympify(expr_b))
            if va == vb:
                return True
        except Exception:
            pass
    if parse_latex is not None:
        try:
            va = simplify(parse_latex(a))
            vb = simplify(parse_latex(b))
            return va == vb
        except Exception:
            pass
    return False


def grade_answer(predicted: str, ground_truth: str) -> bool:
    """Grade predicted answer against ground truth: exact match or sympy equivalence."""
    pred = extract_boxed_answer(predicted) if "\boxed" in predicted or "\\boxed" in predicted else predicted.strip()
    gt = extract_boxed_answer(ground_truth) if "\boxed" in ground_truth or "\\boxed" in ground_truth else ground_truth.strip()
    pred = _normalize_math(pred)
    gt = _normalize_math(gt)
    if pred == gt:
        return True
    return _try_sympy_equal(pred, gt)


class BaseSolver(ABC):
    """Abstract solver: question + budget -> SolverResult."""

    @abstractmethod
    def solve(self, question: str, ground_truth: str, budget_tokens: int) -> SolverResult:
        """Solve the question with at most budget_tokens thinking tokens."""
        pass
