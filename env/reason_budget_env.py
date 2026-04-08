"""Core OpenEnv environment: ReasonBudgetEnvironment (reset/step/state).

v2: The environment is a grader, not a solver-wrapper. It receives the LLM's
text output, tokenizes it, extracts and grades the answer, and returns rewards.
The LLM generation is driven by the trainer (via environment_factory).
"""

import uuid
import warnings
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
    config: EnvConfig,
):
    """Build an observation dict from internal episode state."""
    if step_idx >= len(questions):
        q_rem = 0
        budget_per = 0.0
        question_text = ""
        problem_type = None
    else:
        q_rem = len(questions) - step_idx
        budget_per = remaining_budget / q_rem if q_rem > 0 else 0.0
        question_text = questions[step_idx].text
        problem_type = getattr(questions[step_idx], "problem_type", None)
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
        metadata={
            "problem_type": problem_type,
            "min_tokens": config.min_tokens,
            "max_tokens": config.max_tokens,
            "num_questions": len(questions),
            "budget_ratio": config.budget_ratio,
        },
    )


class ReasonBudgetEnvironment(
    Environment[ReasonBudgetAction, ReasonBudgetObservation, ReasonBudgetState]
):
    """OpenEnv environment: sequential reasoning budget allocation (v2).

    The environment serves math questions, tokenizes the LLM's response to
    count tokens_used, extracts and grades the answer, and returns per-step
    rewards plus an episode-level bonus on the final step.
    """

    # Required by openenv HTTPEnvServer when create_app(..., max_concurrent_envs>1):
    # validation uses getattr(env_cls, "SUPPORTS_CONCURRENT_SESSIONS", False).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

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
        self._tokenizer_cache_key: Optional[str] = None
        self._active_tokenizer_name: Optional[str] = None
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
        self._budget_source: str = "config"

    def _resolved_tokenizer_name(self):
        if self._active_tokenizer_name:
            return self._active_tokenizer_name
        return self.config.tokenizer_name

    def _invalidate_tokenizer_cache(self):
        self._tokenizer = None
        self._tokenizer_cache_key = None

    def _get_tokenizer(self):
        name = self._resolved_tokenizer_name()
        if self._tokenizer is not None and self._tokenizer_cache_key == name:
            return self._tokenizer
        self._invalidate_tokenizer_cache()
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                name, trust_remote_code=True
            )
            self._tokenizer_cache_key = name
        except Exception:
            self._tokenizer = None
            self._tokenizer_cache_key = None
        return self._tokenizer

    def _count_tokens(self, text: str):
        """Count tokens in text using the LLM's tokenizer, or approximate by whitespace."""
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            return len(tokenizer.encode(text, add_special_tokens=False))
        # Rough fallback: ~0.75 words per token (conservative)
        return max(1, int(len(text.split()) * 1.33))

    def _compute_tokenizer_native_budget(self, questions: list[Question]):
        """Compute total budget by tokenizing the actual episode questions.

        Budget = budget_ratio * sum(token_count(question_i.text)) for all sampled
        questions, measured in the active policy tokenizer's units.  Falls back to
        the config-derived formula if the tokenizer cannot be loaded.
        """
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            warnings.warn(
                "Could not load tokenizer for tokenizer-native budget computation; "
                "falling back to config-derived budget (abstract units). "
                f"Attempted tokenizer: {self._resolved_tokenizer_name()!r}",
                UserWarning,
                stacklevel=2,
            )
            return self.config.get_total_budget()
        total_question_tokens = sum(
            len(tokenizer.encode(q.text, add_special_tokens=False))
            for q in questions
        )
        return max(1, int(self.config.budget_ratio * total_question_tokens))

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        total_budget: Optional[int] = None,
        **kwargs,
    ):
        tokenizer_name = tokenizer_name or kwargs.pop("tokenizer_name", None)
        total_budget = total_budget or kwargs.pop("total_budget", None)
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
        self._step_idx = 0
        self._history = []
        self._total_correct = 0

        # --- Tokenizer setup (must precede budget computation) ---
        tn = (tokenizer_name or "").strip()
        if tn:
            self._active_tokenizer_name = tn
        else:
            self._active_tokenizer_name = None
        self._invalidate_tokenizer_cache()

        # --- Budget computation (priority: client override > tokenizer-native > config) ---
        if total_budget is not None:
            self.total_budget = int(total_budget)
            self._budget_source = "client"
        elif self._active_tokenizer_name:
            self.total_budget = self._compute_tokenizer_native_budget(self._questions)
            self._budget_source = "tokenizer_native"
        else:
            warnings.warn(
                "No tokenizer_name provided on reset and no explicit total_budget; "
                "budget cap is derived from config min_tokens/max_tokens in abstract "
                "units, not aligned to any policy tokenizer. Set tokenizer_name on "
                "reset or pass total_budget explicitly for tokenizer-aligned budgets.",
                UserWarning,
                stacklevel=2,
            )
            self.total_budget = self.config.get_total_budget()
            self._budget_source = "config"

        self._remaining_budget = self.total_budget
        obs = _obs_from_internals(
            step_idx=self._step_idx,
            questions=self._questions,
            remaining_budget=self._remaining_budget,
            total_correct=self._total_correct,
            history=self._history,
            config=self.config,
        )
        obs.reward = 0.0
        obs.done = False
        obs.metadata["total_budget"] = self.total_budget
        obs.metadata["budget_source"] = self._budget_source
        return obs

    def step(
        self,
        action: ReasonBudgetAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ):
        # Already past all questions
        if self._step_idx >= len(self._questions):
            obs = _obs_from_internals(
                step_idx=self._step_idx,
                questions=self._questions,
                remaining_budget=self._remaining_budget,
                total_correct=self._total_correct,
                history=self._history,
                config=self.config,
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
                config=self.config,
            )
            obs.reward = 0.0
            obs.done = True
            return obs

        question = self._questions[self._step_idx]

        md = action.metadata or {}
        meta_tn = md.get("tokenizer_name")
        if isinstance(meta_tn, str) and meta_tn.strip():
            mts = meta_tn.strip()
            if mts != (self._active_tokenizer_name or ""):
                self._active_tokenizer_name = mts
                self._invalidate_tokenizer_cache()

        # 1. Tokenize the response to count tokens_used
        tokens_raw = self._count_tokens(action.response)
        if self.config.hard_cap_mode:
            tokens_used = min(tokens_raw, max(0, self._remaining_budget))
        else:
            tokens_used = tokens_raw

        # 2. Extract and grade the answer (optional visible-only tail for hybrid think models)
        grading_extra = md.get("grading_response")
        grading_text = (
            grading_extra.strip()
            if isinstance(grading_extra, str) and grading_extra.strip()
            else action.response
        )
        predicted = extract_boxed_answer(grading_text)
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
            config=self.config,
        )
        obs.reward = reward
        obs.done = terminated
        return obs

    @property
    def state(self):
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
