"""GRPO training script: wire GRPOTrainer with ReasonBudgetEnvironment via rollout_func."""

import argparse
from typing import Any

from env import EnvConfig, ReasonBudgetEnvironment
from env.models import ReasonBudgetAction, ReasonBudgetObservation


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are solving math problems under a shared token budget. "
    "Show your reasoning, then give your final answer in \\boxed{}."
)


def format_observation_prompt(obs: ReasonBudgetObservation):
    """Format an environment observation into a natural language prompt for the LLM."""
    history_lines = ""
    if obs.history:
        entries = []
        for i, h in enumerate(obs.history, 1):
            status = "correct" if h.get("was_correct") else "wrong"
            tokens = h.get("tokens_used", "?")
            summary = h.get("question_summary", "")
            entries.append(f"  Q{i}: {summary}... [{tokens} tokens, {status}]")
        history_lines = "\n".join(entries)
    else:
        history_lines = "  (none yet)"

    return (
        f"Remaining budget: {int(obs.remaining_budget)} tokens\n"
        f"Questions remaining: {obs.questions_remaining} (including this one)\n"
        f"Budget per remaining question: {obs.budget_per_remaining:.0f} tokens\n"
        f"Your accuracy so far: {obs.accuracy_so_far:.0%}\n"
        f"\nPrevious questions:\n{history_lines}\n"
        f"\nCurrent question:\n{obs.question}\n"
        f"\nSolve this problem. Show your reasoning, then give your final answer in \\boxed{{}}."
    )


# ---------------------------------------------------------------------------
# Reward function for GRPOTrainer
# ---------------------------------------------------------------------------


def reward_from_env(completions: list, **kwargs: Any):
    """Extract environment rewards for GRPOTrainer's reward_funcs interface."""
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards:
        return [float(r) for r in env_rewards]
    return [0.0] * len(completions or [])


# ---------------------------------------------------------------------------
# Rollout function
# ---------------------------------------------------------------------------


def rollout_func(prompts: list[str], trainer: Any):
    """Core rollout loop: run episodes against the ReasonBudgetEnvironment.

    For each prompt in the batch, run a full episode:
    1. Reset the env
    2. For each step, format observation -> generate completion -> step env
    3. Collect prompt_ids, completion_ids, logprobs, and rewards
    """
    from trl.trainer.grpo_trainer import generate_rollout_completions

    env = ReasonBudgetEnvironment(config=EnvConfig())
    tokenizer = trainer.processing_class

    all_prompt_ids: list = []
    all_completion_ids: list = []
    all_logprobs: list = []
    all_rewards: list[float] = []

    for _prompt in prompts:
        obs = env.reset()
        episode_reward = 0.0

        while not obs.done:
            user_prompt = format_observation_prompt(obs)

            if env.config.hard_cap_mode:
                max_new_tokens = min(
                    env.config.max_tokens_per_step,
                    max(0, int(obs.remaining_budget)),
                )
                max_new_tokens = max(max_new_tokens, env.min_tokens)
            else:
                max_new_tokens = env.config.max_tokens_per_step

            outputs = generate_rollout_completions(
                trainer,
                [user_prompt],
                max_new_tokens=max_new_tokens,
            )[0]

            completion_text = tokenizer.decode(
                outputs["completion_ids"], skip_special_tokens=True
            )

            obs = env.step(ReasonBudgetAction(response=completion_text))
            episode_reward += float(obs.reward or 0.0)

            all_prompt_ids.append(outputs["prompt_ids"])
            all_completion_ids.append(outputs["completion_ids"])
            all_logprobs.append(outputs["logprobs"])

        all_rewards.append(episode_reward)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": all_rewards,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GRPO training with ReasonBudgetEnvironment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--vllm_mode", type=str, default="colocate", choices=["colocate", "server"])
    parser.add_argument("--output_dir", type=str, default="runs/grpo_train")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    args = parser.parse_args()

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    # Placeholder dataset -- rollout_func drives the actual episode data
    dataset = Dataset.from_dict({"prompt": ["placeholder"] * 100})

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_from_env,
        train_dataset=dataset,
        rollout_func=rollout_func,
        args=grpo_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
