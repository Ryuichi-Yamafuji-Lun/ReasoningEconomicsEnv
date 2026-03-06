"""Core OpenEnv environment: ReasonBudgetEnvironment (reset/step/state)."""

import uuid
from typing import Optional

import numpy as np

from reasonbudget_gym.env.config import EnvConfig
from reasonbudget_gym.env.episode_sampler import EpisodeSampler, Question
from reasonbudget_gym.env.models import ReasonBudgetAction, ReasonBudgetObservation, ReasonBudgetState
from reasonbudget_gym.env.reward import compute_reward

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    from abc import ABC, abstractmethod
    from typing import Generic, TypeVar
    ActT = TypeVar("ActT")
    ObsT = TypeVar("ObsT")
    StateT = TypeVar("StateT")
    class Environment(ABC, Generic[ActT, ObsT, StateT]):
        @abstractmethod
        def reset(self, seed=None, episode_id=None, **kwargs): ...
        @abstractmethod
        def step(self, action, timeout_s=None, **kwargs): ...
        @property
        @abstractmethod
        def state(self): ...


def _obs_from_internals(
    *,
    embeddings: list,
    step_idx: int,
    questions: list,
    remaining_budget: int,
    total_correct: int,
    history: list,
    embedding_dim: int = 384,
) -> ReasonBudgetObservation:
    if step_idx >= len(questions):
        q_rem = 0
        emb = [0.0] * embedding_dim
        budget_per = 0.0
        question_text = ""
    else:
        q_rem = len(questions) - step_idx
        emb = embeddings[step_idx].tolist() if hasattr(embeddings[step_idx], "tolist") else list(embeddings[step_idx])
        if len(emb) != embedding_dim:
            emb = list(emb)[:embedding_dim] + [0.0] * (embedding_dim - len(emb))
        budget_per = remaining_budget / q_rem if q_rem > 0 else 0.0
        question_text = questions[step_idx].text if step_idx < len(questions) else ""
    acc = total_correct / step_idx if step_idx > 0 else 0.0
    return ReasonBudgetObservation(
        question_embedding=emb,
        remaining_budget=float(remaining_budget),
        questions_remaining=q_rem,
        step_idx=step_idx,
        budget_per_remaining=budget_per,
        accuracy_so_far=acc,
        question=question_text,
        history=list(history),
        done=False,
        reward=None,
    )


class ReasonBudgetEnvironment(Environment[ReasonBudgetAction, ReasonBudgetObservation, ReasonBudgetState]):
    """OpenEnv environment: sequential reasoning budget allocation."""

    def __init__(self, config: Optional[EnvConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or EnvConfig()
        self._sampler = EpisodeSampler(seed=self.config.seed)
        self._solver = None
        self._encoder = None
        self.num_questions = self.config.num_questions
        self.budget_tiers = self.config.budget_tiers
        self.total_budget = self.config.get_total_budget()
        self.embedding_dim = 384
        self._episode_id: Optional[str] = None
        self._questions: list[Question] = []
        self._embeddings: list = []
        self._step_idx: int = 0
        self._remaining_budget: int = 0
        self._history: list[dict] = []
        self._total_correct: int = 0

    def _get_solver(self):
        if self._solver is not None:
            return self._solver
        if self.config.use_cache:
            from reasonbudget_gym.solver.cached_solver import CachedSolver
            self._solver = CachedSolver(self.config.cache_path)
        else:
            from reasonbudget_gym.solver.live_solver import LiveSolver
            self._solver = LiveSolver(self.config.solver_model)
        return self._solver

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(self.config.embedding_model)
        return self._encoder

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> ReasonBudgetObservation:
        if seed is not None:
            self._sampler = EpisodeSampler(seed=seed)
        self._episode_id = episode_id or str(uuid.uuid4())
        self._questions = self._sampler.sample_episode(
            self.num_questions,
            self.config.difficulty_mix,
            seed=seed,
        )
        if len(self._questions) < self.num_questions:
            self.num_questions = len(self._questions)
        self.total_budget = self.config.get_total_budget()
        self._remaining_budget = self.total_budget
        self._step_idx = 0
        self._history = []
        self._total_correct = 0
        encoder = self._get_encoder()
        texts = [q.text for q in self._questions]
        self._embeddings = encoder.encode(texts, convert_to_numpy=True)
        obs = _obs_from_internals(
            embeddings=self._embeddings,
            step_idx=self._step_idx,
            questions=self._questions,
            remaining_budget=self._remaining_budget,
            total_correct=self._total_correct,
            history=self._history,
            embedding_dim=self.embedding_dim,
        )
        obs.reward = 0.0
        obs.done = False
        return obs

    def step(
        self,
        action: ReasonBudgetAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> ReasonBudgetObservation:
        if self._step_idx >= len(self._questions):
            obs = _obs_from_internals(
                embeddings=self._embeddings,
                step_idx=self._step_idx,
                questions=self._questions,
                remaining_budget=self._remaining_budget,
                total_correct=self._total_correct,
                history=self._history,
                embedding_dim=self.embedding_dim,
            )
            obs.reward = 0.0
            obs.done = True
            return obs
        tier_tokens = self.budget_tiers[action.budget_allocation]
        spend = min(tier_tokens, self._remaining_budget)
        if spend < self.budget_tiers[0]:
            obs = _obs_from_internals(
                embeddings=self._embeddings,
                step_idx=self._step_idx,
                questions=self._questions,
                remaining_budget=self._remaining_budget,
                total_correct=self._total_correct,
                history=self._history,
                embedding_dim=self.embedding_dim,
            )
            obs.reward = 0.0
            obs.done = True
            return obs
        question = self._questions[self._step_idx]
        solver = self._get_solver()
        if hasattr(solver, "set_question_id"):
            solver.set_question_id(question.id)
        result = solver.solve(question.text, question.answer, spend)
        self._total_correct += 1 if result.was_correct else 0
        reward = compute_reward(
            result.was_correct,
            spend,
            self.total_budget,
            self.num_questions,
            beta=self.config.beta,
            gamma=self.config.gamma,
        )
        self._remaining_budget -= spend
        self._history.append({
            "tokens_allocated": spend,
            "tokens_used": result.tokens_used,
            "was_correct": result.was_correct,
        })
        self._step_idx += 1
        terminated = self._step_idx >= len(self._questions)
        truncated = self._remaining_budget < self.budget_tiers[0] and not terminated
        if truncated and self._step_idx < len(self._questions):
            terminated = True
        obs = _obs_from_internals(
            embeddings=self._embeddings,
            step_idx=self._step_idx,
            questions=self._questions,
            remaining_budget=self._remaining_budget,
            total_correct=self._total_correct,
            history=self._history,
            embedding_dim=self.embedding_dim,
        )
        obs.reward = reward
        obs.done = terminated
        return obs

    @property
    def state(self) -> ReasonBudgetState:
        spent = self.total_budget - self._remaining_budget
        return ReasonBudgetState(
            episode_id=self._episode_id,
            step_count=self._step_idx,
            total_budget=self.total_budget,
            spent_budget=spent,
            questions_answered=self._step_idx,
            total_correct=self._total_correct,
            current_accuracy=self._total_correct / self._step_idx if self._step_idx > 0 else 0.0,
            budget_remaining_ratio=self._remaining_budget / self.total_budget if self.total_budget > 0 else 0.0,
        )


# Backward compatibility alias (deprecated): use ReasonBudgetEnvironment
ReasonBudgetEnv = ReasonBudgetEnvironment
