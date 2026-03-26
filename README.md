---
title: Reasoning Economics Env
emoji: 📐
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# ReasoningEconomicsEnv v2

A **post-training RL environment** for the OpenEnv Challenge (Meta PyTorch + HuggingFace + Unsloth). The LLM being trained is both the allocator and the solver. The environment serves math questions under a shared token budget, grades answers, counts tokens, and returns rewards.

## Design

The LLM receives math questions, generates reasoning traces and answers, and is rewarded or penalized by the environment based on correctness and token efficiency. There is no separate MLP policy, no frozen solver, and no cache.

Through the reward signal across long-horizon episodes, the LLM learns **meta-reasoning**:
- How to spend a shared token budget across a **mixed** sequence of problems (MetaMathQA `type` variants + NuminaMath-TIR)
- How to trade off reasoning length vs correctness under that global budget

## Quick Start

```bash
pip install -e .
```

### Run dummy baselines (no GPU needed)

```bash
python -m eval.evaluate --n_episodes 20 --seed 42
```

Dummy baselines live in `baselines/dummy/` and are for budget/reward smoke tests.

### Run LLM-backed baselines (accuracy-oriented)

LLM baselines live in `baselines/llm/` and require endpoint/model env vars.

API-backed baseline:

```bash
export BASELINE_API_BASE_URL="https://your-provider.example/v1"
export BASELINE_API_KEY="your_api_key"
export BASELINE_API_MODEL="your-model-name"
python -m eval.evaluate --include_llm --baselines llm_api --n_episodes 2 --seed 42
```

Local/self-hosted baseline (OpenAI-compatible server, e.g. vLLM):

```bash
export BASELINE_LOCAL_BASE_URL="http://127.0.0.1:8001/v1"
export BASELINE_LOCAL_API_KEY="local"
export BASELINE_LOCAL_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
python -m eval.evaluate --include_llm --baselines llm_local --n_episodes 2 --seed 42
```

### GRPO training (requires GPU + vLLM)

```bash
python -m training.grpo_train \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --num_train_epochs 1 \
    --output_dir runs/grpo_train
```

### OpenEnv server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Environment Contract

**Action**: `ReasonBudgetAction(response=str)` -- the LLM's full text output (reasoning trace + answer).

**Observation**: question text, remaining budget, questions remaining, budget per remaining question, accuracy so far, episode history, done, reward.

**Reward**:
- Hard-cap mode (`hard_cap_mode=True`): correctness + efficiency bonus - cost penalty, plus episode bonus.
- Soft-budget mode (`hard_cap_mode=False`): same core reward plus an overspend penalty term that increases as total spend exceeds episode budget.

**Termination**:
- Hard-cap mode: all questions answered, or remaining budget below `min_tokens`.
- Soft-budget mode: all questions answered (no early stop on budget exhaustion).

## Datasets

| Dataset | HuggingFace ID | Notes |
|---------|---------------|--------|
| MetaMathQA | `meta-math/MetaMathQA` | Rows keyed by dataset `type` (e.g. `GSM_SV`, `MATH_FOBAR`, …) |
| NuminaMath-TIR | `AI-MO/NuminaMath-TIR` | Mixed into episodes as `NuminaMath_TIR` |

Episodes sample an **even mix** across available MetaMathQA types plus Numina (reward uses only **total token budget** per episode, not difficulty buckets).

## Budget Modes

Configure in `env/config.py`:
- `hard_cap_mode = True` (default): clip per-step spend to remaining budget and stop early when budget is effectively exhausted.
- `hard_cap_mode = False`: no clipping by remaining budget, allow overspend, and learn via explicit overspend penalties.

## Docker

```bash
docker build -f server/Dockerfile -t reasoning-economic-env .
docker run -p 8000:8000 reasoning-economic-env
```

## OpenEnv

```bash
openenv validate
openenv push
```
