"""Answer extraction and grading: boxed-answer parsing, exact match, SymPy equivalence."""

import re

from sympy import simplify, sympify

try:
    from sympy.parsing.latex import parse_latex
except ImportError:
    parse_latex = None


def extract_boxed_answer(text: str):
    r"""Extract content of last \boxed{...} in text, or fall back to last line."""
    if not text:
        return ""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    lines = text.strip().split("\n")
    return lines[-1].strip() if lines else ""


def _normalize_math(s: str):
    """Strip and collapse whitespace for comparison."""
    return " ".join(s.strip().split())


def _try_sympy_equal(a: str, b: str):
    """Compare two math expressions symbolically (including LaTeX-style)."""
    a, b = _normalize_math(a), _normalize_math(b)
    if a == b:
        return True
    for expr_a, expr_b in [(a, b), (a.replace("\\frac", "frac"), b)]:
        try:
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


def _try_numeric_equal(a: str, b: str, tol: float = 1e-6):
    """Compare two strings as floats within tolerance."""
    try:
        return abs(float(a) - float(b)) < tol
    except (ValueError, TypeError):
        return False


def grade_answer(predicted: str, ground_truth: str):
    """Grade predicted answer against ground truth.

    Pipeline: extract boxed answer -> exact match -> numeric tolerance -> SymPy equivalence.
    """
    pred = extract_boxed_answer(predicted) if "\\boxed" in predicted else predicted.strip()
    gt = extract_boxed_answer(ground_truth) if "\\boxed" in ground_truth else ground_truth.strip()
    pred = _normalize_math(pred)
    gt = _normalize_math(gt)
    if pred == gt:
        return True
    if _try_numeric_equal(pred, gt):
        return True
    return _try_sympy_equal(pred, gt)
