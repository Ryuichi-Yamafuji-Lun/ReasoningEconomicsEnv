"""Problem-type oracle baseline for verbosity only (not answer correctness)."""

from env.models import ReasonBudgetObservation

# Heuristic target lengths (tokens) — env reward is still driven by global budget only.
PROBLEM_TYPE_TOKEN_MAP = {
    "MATH_AnsAug": 400,
    "GSM_Rephrased": 120,
    "GSM_SV": 100,
    "GSM_FOBAR": 140,
    "GSM_AnsAug": 120,
    "MATH_FOBAR": 500,
    "MATH_Rephrased": 380,
    "MATH_SV": 350,
    "NuminaMath_TIR": 700,
}


class DifficultyOracleBaseline:
    """Uses known problem_type to scale response length (upper-bound heuristic)."""

    def __init__(self, min_tokens: int = 10, max_tokens: int = 800):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def _clamp(self, tokens: int) -> int:
        return max(self.min_tokens, min(self.max_tokens, tokens))

    def select_action(
        self,
        _observation: ReasonBudgetObservation,
        problem_type: str | None = None,
        **_context,
    ) -> str:
        target = PROBLEM_TYPE_TOKEN_MAP.get(problem_type or "", 300)
        target = self._clamp(target)
        filler_len = max(0, target * 4 - 80)
        reasoning = "Let me work through this. " + ("work " * (filler_len // 5))
        return f"{reasoning.strip()}\n\n\\boxed{{0}}"
