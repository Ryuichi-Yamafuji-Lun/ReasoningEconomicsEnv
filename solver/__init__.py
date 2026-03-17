"""Solver: base interface, cached solver, and live solver."""

from solver.base_solver import BaseSolver, SolverResult, grade_answer
from solver.cached_solver import CachedSolver
from solver.live_solver import LiveSolver

__all__ = ["BaseSolver", "SolverResult", "grade_answer", "CachedSolver", "LiveSolver"]
