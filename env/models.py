"""OpenEnv Pydantic models: Action, Observation, State for ReasonBudget environment."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server.types import Action as _ActionBase
    from openenv.core.env_server.types import Observation as _ObservationBase
    from openenv.core.env_server.types import State as _StateBase
except ImportError:
    _ActionBase = BaseModel
    _ObservationBase = BaseModel
    _StateBase = BaseModel


class ReasonBudgetAction(_ActionBase):
    """Action: LLM's full text output (reasoning trace + answer)."""

    if _ActionBase is BaseModel:
        model_config = ConfigDict(extra="forbid")
        metadata: dict[str, Any] = Field(default_factory=dict)

    response: str = Field(..., description="LLM's full text output (reasoning trace + answer)")


class ReasonBudgetObservation(_ObservationBase):
    """Observation: question text, budget state, history, and step result (reward, done)."""

    if _ObservationBase is BaseModel:
        model_config = ConfigDict(extra="forbid")
        done: bool = Field(default=False)
        reward: float | None = Field(default=None)
        metadata: dict[str, Any] = Field(default_factory=dict)

    remaining_budget: float = Field(..., description="Tokens remaining in episode budget")
    questions_remaining: int = Field(..., ge=0, description="Questions left in episode")
    step_idx: int = Field(..., ge=0, description="Current step index")
    budget_per_remaining: float = Field(..., description="remaining_budget / questions_remaining")
    accuracy_so_far: float = Field(..., ge=0, le=1, description="Fraction of correct answers so far")
    question: str = Field(default="", description="Current math question text")
    history: list[dict[str, Any]] = Field(default_factory=list, description="Past step summaries")


class ReasonBudgetState(_StateBase):
    """Episode-level state metadata."""

    if _StateBase is BaseModel:
        model_config = ConfigDict(extra="allow")
        episode_id: str | None = Field(default=None)
        step_count: int = Field(default=0, ge=0)

    total_budget: int = Field(..., ge=0)
    spent_budget: int = Field(..., ge=0)
    questions_answered: int = Field(..., ge=0)
    total_correct: int = Field(..., ge=0)
    current_accuracy: float = Field(..., ge=0, le=1)
    budget_remaining_ratio: float = Field(..., ge=0, le=1)
