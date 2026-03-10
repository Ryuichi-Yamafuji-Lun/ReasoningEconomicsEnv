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
    """Action: direct token allocation for current question."""

    if _ActionBase is BaseModel:
        model_config = ConfigDict(extra="forbid")
        metadata: dict[str, Any] = Field(default_factory=dict)

    token_allocation: int = Field(..., ge=1, description="Direct token count to allocate")


class ReasonBudgetObservation(_ObservationBase):
    """Observation: question embedding, budget state, and step result (reward, done)."""

    if _ObservationBase is BaseModel:
        model_config = ConfigDict(extra="forbid")
        done: bool = Field(default=False)
        reward: float | None = Field(default=None)
        metadata: dict[str, Any] = Field(default_factory=dict)

    question_embedding: list[float] = Field(..., description="Frozen encoder embedding (e.g. 384-dim)")
    remaining_budget: float = Field(..., ge=0, description="Tokens remaining in episode budget")
    questions_remaining: int = Field(..., ge=0, description="Questions left in episode")
    step_idx: int = Field(..., ge=0, description="Current step index")
    budget_per_remaining: float = Field(..., ge=0, description="remaining_budget / questions_remaining")
    accuracy_so_far: float = Field(..., ge=0, le=1, description="Fraction of correct answers so far")
    question: str = Field(default="", description="Raw question text (optional)")
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
