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

**Action**: `ReasonBudgetAction(response=str)` -- the LLM's full text output (reasoning trace + answer). Optional `metadata` may include:
- **`tokenizer_name`** (Hugging Face model id) — when set, the server uses `AutoTokenizer.from_pretrained` for that id to count tokens on that step (and updates the session tokenizer if it changed).
- **`grading_response`** (string, optional) — if non-empty, used **only** for `\\boxed{}` extraction and answer grading. Token budgeting always uses `response` in full. Use this when the policy emits hybrid internal reasoning before the final answer (e.g. Qwen3 family) so a spurious `\\boxed{}` inside the reasoning span does not affect grading.

**Reset (OpenEnv / WebSocket)**: Besides `seed` and `episode_id`, clients may pass:
- **`tokenizer_name`** (string, Hugging Face model id) — aligns both the episode budget cap **and** per-step token counting to the policy tokenizer (see Total episode budget below).
- **`total_budget`** (int, optional) — explicit override for the episode budget cap; skips all automatic computation.

**Total episode budget**: The `total_budget` for an episode is resolved with the following priority on each `reset`:

| Priority | Condition | Formula | `budget_source` in obs metadata |
|----------|-----------|---------|--------------------------------|
| 1 — Client override | `total_budget` passed on reset | exact integer from client | `"client"` |
| 2 — Tokenizer-native *(default for training)* | `tokenizer_name` passed on reset | `budget_ratio × Σ tokenize(question_i.text)` over the **10 sampled questions** | `"tokenizer_native"` |
| 2b — Tokenizer load failure | `tokenizer_name` passed but server cannot load it | falls back to config formula (warns) | `"config"` |
| 3 — Config fallback | neither `tokenizer_name` nor `total_budget` passed (warns) | `budget_ratio × num_questions × (min_tokens + max_tokens) / 2` | `"config"` |

Under **tokenizer-native** (the normal training path), after sampling the 10 questions for the episode the server tokenizes each question’s text with `AutoTokenizer.from_pretrained(tokenizer_name)` and sets `total_budget = int(budget_ratio × total_question_tokens)`. Both the cap **and** per-step spend are then measured in the same policy-tokenizer units.

The resolved `total_budget` and `budget_source` are returned in `obs.metadata` of the reset observation so clients can verify which path was taken.

**Server requirement**: The env host (Docker, HF Space, etc.) must be able to download and cache tokenizers from the Hub for any `tokenizer_name` you send (network access on first use).

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

Build context is the **repo root** (`ReasoningEconomicsEnv/`).

```bash
docker build -f server/Dockerfile -t reasoning-economic-env .
docker run --rm -p 8000:8000 reasoning-economic-env
docker run -d --name ree-env -p 8000:8000 reasoning-economic-env # detached container to continue using shell
```

**Build logs:** The Dockerfile prints `BASE_IMAGE` and a JSON dump of default `EnvConfig` (including `tokenizer_name`, budget fields) in the **builder** and **final** stages. **Container logs:** On start, the entrypoint prints `REE_*` env vars and the **effective** `EnvConfig` after `REE_DEFAULT_TOKENIZER_NAME` / `REE_PROD` (optional).

**Remote H100 (or any GPU host) — run env beside training:** The OpenEnv server is CPU-only (tokenizer + grading); no GPU flag is required for this container.

```bash
# On the H100 machine (from this directory)
docker build -f server/Dockerfile -t ree-env:latest .

# Optional: set default Hub tokenizer if clients omit reset tokenizer_name
docker run --rm -d --name ree-env -p 8000:8000 \
  -e REE_DEFAULT_TOKENIZER_NAME="Qwen/Qwen3-4B" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  ree-env:latest
```

Point **ReasoningEconomicsPT** at `http://<host>:8000` (`--env_base_url` or `http://127.0.0.1:8000` if co-located). Ensure the host can reach Hugging Face to download tokenizers the first time (or pre-populate the cache volume).

## OpenEnv

```bash
openenv validate
openenv push
```
