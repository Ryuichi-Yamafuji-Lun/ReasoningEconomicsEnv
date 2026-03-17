"""Build response cache: (question_id, budget_tier) -> {answer, was_correct, tokens_used, response_text}."""

import argparse
import json
import os
import random
import time
from pathlib import Path

from solver.base_solver import grade_answer, extract_boxed_answer


def call_together_api(
    question: str,
    budget_tokens: int,
    api_key: str,
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
) -> tuple[str, int]:
    """Call Together AI (OpenAI-compatible) for one question with max_tokens=budget_tokens.
    Returns (response_text, tokens_used).
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required for cache_builder. pip install openai")

    client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
    prompt = f"<|User|>{question}<|Assistant|><think>"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=budget_tokens + 128,
        temperature=0.0,
    )
    choice = resp.choices[0] if resp.choices else None
    if not choice:
        return "", 0
    text = choice.message.content or ""
    usage = getattr(resp, "usage", None)
    tokens_used = usage.completion_tokens if usage else 0
    return text, tokens_used


def build_cache(
    questions: list,
    budget_tiers: list[int],
    api_key: str,
    output_path: str,
    checkpoint_every: int = 50,
    holdout_ratio: float = 0.2,
) -> None:
    """For each question and tier, call API, grade, and write cache."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n_holdout = int(len(questions) * holdout_ratio)
    eval_ids = set(q.id for q in questions[:n_holdout]) if n_holdout else set()
    entries = {}
    for qi, q in enumerate(questions):
        qid = q.id
        entries[qid] = {}
        for tier in budget_tiers:
            try:
                text, tokens_used = call_together_api(q.text, tier, api_key)
            except Exception as e:
                text, tokens_used = "", 0
                print(f"API error q={qid} tier={tier}: {e}")
            answer = extract_boxed_answer(text)
            was_correct = grade_answer(answer, q.answer)
            entries[qid][str(tier)] = {
                "answer": answer,
                "was_correct": was_correct,
                "tokens_used": tokens_used,
                "response_text": text[:2000],
            }
            time.sleep(0.5)
        if (qi + 1) % checkpoint_every == 0:
            out = {"entries": entries, "eval_ids": list(eval_ids)}
            with open(output_path, "w") as f:
                json.dump(out, f, indent=1)
            print(f"Checkpoint: {qi + 1}/{len(questions)}")
    out = {"entries": entries, "eval_ids": list(eval_ids)}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote cache to {output_path}")


def main() -> None:
    from env.episode_sampler import EpisodeSampler

    parser = argparse.ArgumentParser(description="Build response cache for ReasoningEconomicsEnv")
    parser.add_argument("--num_questions", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/response_cache.json")
    parser.add_argument("--api_key", type=str, default=os.environ.get("TOGETHER_API_KEY", ""))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    args = parser.parse_args()
    if not args.api_key:
        print("Set TOGETHER_API_KEY or pass --api_key")
        return

    sampler = EpisodeSampler(seed=args.seed)
    by_diff = sampler._load_meta_math()
    all_q = []
    rng = random.Random(args.seed)
    for diff, pool in by_diff.items():
        n = max(1, args.num_questions // 4)
        all_q.extend(rng.sample(pool, min(n, len(pool))))
    rng = random.Random(args.seed)
    rng.shuffle(all_q)
    questions = all_q[: args.num_questions]
    if len(questions) < args.num_questions:
        print(f"Only {len(questions)} questions available")
    build_cache(
        questions,
        budget_tiers=[50, 100, 200, 400, 800],
        api_key=args.api_key,
        output_path=args.output,
        checkpoint_every=args.checkpoint_every,
        holdout_ratio=0.2,
    )


if __name__ == "__main__":
    main()
