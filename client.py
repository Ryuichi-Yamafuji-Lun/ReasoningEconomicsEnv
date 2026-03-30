"""Typed OpenEnv client for the ReasonBudget environment (v2)."""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

try:
    from .env.models import (
        ReasonBudgetAction,
        ReasonBudgetObservation,
        ReasonBudgetState,
    )
except ImportError:
    from env.models import (
        ReasonBudgetAction,
        ReasonBudgetObservation,
        ReasonBudgetState,
    )


class ReasonBudgetEnvClient(
    EnvClient[ReasonBudgetAction, ReasonBudgetObservation, ReasonBudgetState]
):
    """WebSocket client for interacting with a ReasonBudget OpenEnv server."""

    def _step_payload(self, action: ReasonBudgetAction) -> Dict[str, Any]:
        """Convert a typed action to the step payload."""
        return {"response": action.response}

    def _parse_result(self, payload: Dict[str, Any]):
        """Parse server step/reset payload to a typed StepResult."""
        obs_data = payload.get("observation")
        if not isinstance(obs_data, dict):
            obs_data = payload if isinstance(payload, dict) else {}
        done = payload.get("done", obs_data.get("done", False))
        reward = payload.get("reward", obs_data.get("reward"))
        observation = ReasonBudgetObservation(
            remaining_budget=obs_data.get("remaining_budget", 0.0),
            questions_remaining=obs_data.get("questions_remaining", 0),
            step_idx=obs_data.get("step_idx", 0),
            budget_per_remaining=obs_data.get("budget_per_remaining", 0.0),
            accuracy_so_far=obs_data.get("accuracy_so_far", 0.0),
            question=obs_data.get("question", ""),
            history=obs_data.get("history", []),
            done=done,
            reward=reward,
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]):
        """Parse server state payload to a typed ReasonBudgetState."""
        state_data = payload.get("state")
        if not isinstance(state_data, dict):
            state_data = payload if isinstance(payload, dict) else {}
        return ReasonBudgetState(
            episode_id=state_data.get("episode_id"),
            step_count=state_data.get("step_count", 0),
            total_budget=state_data.get("total_budget", 0),
            spent_budget=state_data.get("spent_budget", 0),
            questions_answered=state_data.get("questions_answered", 0),
            total_correct=state_data.get("total_correct", 0),
            current_accuracy=state_data.get("current_accuracy", 0.0),
            budget_remaining_ratio=state_data.get("budget_remaining_ratio", 0.0),
        )
