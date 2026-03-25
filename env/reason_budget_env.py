"""Core OpenEnv environment: ReasonBudgetEnvironment (reset/step/state).

v2: The environment is a grader, not a solver-wrapper. It receives the LLM's
text output, tokenizes it, extracts and grades the answer, and returns rewards.
The LLM call happens in rollout_func, not here.
"""

import uuid
from typing import Optional

from env.config import EnvConfig
from data.question import Question
from env.episode_sampler import EpisodeSampler
from env.grading import extract_boxed_answer, grade_answer
from env.models import ReasonBudgetAction, ReasonBudgetObservation, ReasonBudgetState
from env.reward import compute_episode_bonus, compute_reward

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
    step_idx: int,
    questions: list,
    remaining_budget: int,
    total_correct: int,
    history: list,
) -> ReasonBudgetObservation:
    """Build an observation dict from internal episode state."""
    if step_idx >= len(questions):
        q_rem = 0
        budget_per = 0.0
        question_text = ""
    else:
        q_rem = len(questions) - step_idx
        budget_per = remaining_budget / q_rem if q_rem > 0 else 0.0
        question_text = questions[step_idx].text
    acc = total_correct / step_idx if step_idx > 0 else 0.0
    return ReasonBudgetObservation(
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


class ReasonBudgetEnvironment(
    Environment[ReasonBudgetAction, ReasonBudgetObservation, ReasonBudgetState]
):
    """OpenEnv environment: sequential reasoning budget allocation (v2).

    The environment serves math questions, tokenizes the LLM's response to
    count tokens_used, extracts and grades the answer, and returns per-step
    rewards plus an episode-level bonus on the final step.
    """

    def __init__(self, config: Optional[EnvConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or EnvConfig()
        self._sampler = EpisodeSampler(
            seed=self.config.seed,
            prod=self.config.prod,
            subset_start_idx=self.config.subset_start_idx,
            subset_size=self.config.subset_size,
            numina_subset_start_idx=self.config.numina_subset_start_idx,
            numina_subset_size=self.config.numina_subset_size,
        )
        self._tokenizer = None
        self.num_questions = self.config.num_questions
        self.min_tokens = self.config.min_tokens
        self.max_tokens = self.config.max_tokens
        self.total_budget = self.config.get_total_budget()

        self._episode_id: Optional[str] = None
        self._questions: list[Question] = []
        self._step_idx: int = 0
        self._remaining_budget: int = 0
        self._history: list[dict] = []
        self._total_correct: int = 0

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name, trust_remote_code=True
            )
        except Exception:
            self._tokenizer = None
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the LLM's tokenizer, or approximate by whitespace."""
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            return len(tokenizer.encode(text, add_special_tokens=False))
        # Rough fallback: ~0.75 words per token (conservative)
        return max(1, int(len(text.split()) * 1.33))

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> ReasonBudgetObservation:
        if seed is not None:
            self._sampler = EpisodeSampler(
                seed=seed,
                prod=self.config.prod,
                subset_start_idx=self.config.subset_start_idx,
                subset_size=self.config.subset_size,
                numina_subset_start_idx=self.config.numina_subset_start_idx,
                numina_subset_size=self.config.numina_subset_size,
            )
        self._episode_id = episode_id or str(uuid.uuid4())
        self._questions = self._sampler.sample_episode(
            self.num_questions,
            seed=seed,
        )
        if len(self._questions) < self.num_questions:
            self.num_questions = len(self._questions)
        self.total_budget = self.config.get_total_budget()
        self._remaining_budget = self.total_budget
        self._step_idx = 0
        self._history = []
        self._total_correct = 0
        obs = _obs_from_internals(
            step_idx=self._step_idx,
            questions=self._questions,
            remaining_budget=self._remaining_budget,
            total_correct=self._total_correct,
            history=self._history,
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
        # Already past all questions
        if self._step_idx >= len(self._questions):
            obs = _obs_from_internals(
                step_idx=self._step_idx,
                questions=self._questions,
                remaining_budget=self._remaining_budget,
                total_correct=self._total_correct,
                history=self._history,
            )
            obs.reward = 0.0
            obs.done = True
            return obs

        # In hard-cap mode, terminate if not enough budget remains for a minimum step.
        if self.config.hard_cap_mode and self._remaining_budget < self.min_tokens:
            obs = _obs_from_internals(
                step_idx=self._step_idx,
                questions=self._questions,
                remaining_budget=self._remaining_budget,
                total_correct=self._total_correct,
                history=self._history,
            )
            obs.reward = 0.0
            obs.done = True
            return obs

        question = self._questions[self._step_idx]

        # 1. Tokenize the response to count tokens_used
        tokens_raw = self._count_tokens(action.response)
        if self.config.hard_cap_mode:
            tokens_used = min(tokens_raw, max(0, self._remaining_budget))
        else:
            tokens_used = tokens_raw

        # 2. Extract and grade the answer
        predicted = extract_boxed_answer(action.response)
        was_correct = grade_answer(predicted, question.answer)

        self._total_correct += 1 if was_correct else 0

        # 3. Compute per-step reward
        step_total_spent = (self.total_budget - self._remaining_budget) + tokens_used
        overspend_tokens = max(0, step_total_spent - self.total_budget)
        reward = compute_reward(
            was_correct,
            tokens_used,
            self.total_budget,
            self.num_questions,
            beta=self.config.beta,
            gamma=self.config.gamma,
            overspend_tokens=overspend_tokens,
            soft_overspend_penalty=self.config.soft_overspend_penalty,
            hard_cap_mode=self.config.hard_cap_mode,
        )

        # 4. Update budget and history
        self._remaining_budget -= tokens_used
        if not self.config.soft_allow_negative_budget:
            self._remaining_budget = max(0, self._remaining_budget)
        self._history.append(
            {
                "tokens_used": tokens_used,
                "was_correct": was_correct,
                "question_summary": question.text[:80],
            }
        )
        self._step_idx += 1

        # 5. Check termination
        terminated = self._step_idx >= len(self._questions)
        truncated = (
            self.config.hard_cap_mode
            and self._remaining_budget < self.min_tokens
            and not terminated
        )
        if truncated:
            terminated = True

        # 6. Episode-level bonus on the final step
        if terminated:
            total_spent = self.total_budget - self._remaining_budget
            reward += compute_episode_bonus(
                self._total_correct,
                self.num_questions,
                total_spent,
                self.total_budget,
                lambda_ep=self.config.lambda_ep,
                target_utilization=self.config.target_utilization,
            )

        obs = _obs_from_internals(
            step_idx=self._step_idx,
            questions=self._questions,
            remaining_budget=self._remaining_budget,
            total_correct=self._total_correct,
            history=self._history,
        )
        obs.reward = reward
        obs.done = terminated
        return obs

    @property
    def state(self) -> ReasonBudgetState:
        spent = self.total_budget - self._remaining_budget
        if self.total_budget > 0:
            ratio = self._remaining_budget / self.total_budget
            budget_remaining_ratio = max(0.0, min(1.0, ratio))
        else:
            budget_remaining_ratio = 0.0
        return ReasonBudgetState(
            episode_id=self._episode_id,
            step_count=self._step_idx,
            total_budget=self.total_budget,
            spent_budget=spent,
            questions_answered=self._step_idx,
            total_correct=self._total_correct,
            current_accuracy=(
                self._total_correct / self._step_idx if self._step_idx > 0 else 0.0
            ),
            budget_remaining_ratio=budget_remaining_ratio,
        )
