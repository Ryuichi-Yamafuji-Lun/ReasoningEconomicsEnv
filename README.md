---
title: ReasoningEconomicsEnv
sdk: docker
app_port: 8000
tags:
  - openenv
  - reasoning-economic-env
---

# ReasoningEconomicsEnv

RL environment for sequential reasoning budget allocation. The project is **OpenEnv-only**: no gymnasium, no stable-baselines3. Baselines and training use `ReasonBudgetObservation` and `ReasonBudgetAction` only.

## Interface

- **Core env**: `ReasonBudgetEnvironment` (OpenEnv `Environment` subclass). Use in-process:
  - `obs = env.reset(seed=..., episode_id=...)` → `ReasonBudgetObservation`
  - `obs = env.step(ReasonBudgetAction(token_allocation=tokens))` → `ReasonBudgetObservation` (with `obs.reward`, `obs.done`)
  - `state = env.state` → `ReasonBudgetState`
- **Deployment**: OpenEnv HTTP/WebSocket server in `server.app`. For remote use, use `ReasonBudgetEnvClient` in `client.py` (typed `EnvClient` subclass for `ReasonBudgetAction` / `ReasonBudgetObservation`).

## Install

```bash
pip install -e .
```

## Quick start

```python
from env import (
    EnvConfig,
    ReasonBudgetEnvironment,
    ReasonBudgetAction,
    ReasonBudgetObservation,
)
env = ReasonBudgetEnvironment(config=EnvConfig(num_questions=10))
obs = env.reset(seed=42)
while not obs.done:
    action = ReasonBudgetAction(token_allocation=200)
    obs = env.step(action)
    print(obs.reward, obs.done)
print(env.state.total_correct, env.state.questions_answered)
```

If remote dataset/model downloads are unavailable, the environment now falls back to a small built-in question set and deterministic embeddings so reset/step still work end-to-end.

## Run server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Or with uv: `uv run server` (from project root).

## Publishing to Hugging Face (OpenEnv)

To validate and publish this environment as an OpenEnv Space on Hugging Face:

**Prerequisites:** Docker, Python 3.11+, and `pip install openenv-core` (optionally [uv](https://github.com/astral-sh/uv) for faster installs and `uv lock` for reproducible Docker builds).

1. **Validate** — From the project root (`ReasoningEconomicsEnv/`):
   ```bash
   openenv validate
   ```
   Fix any reported issues (e.g. `openenv.yaml` path, app import, health endpoint).

2. **Local container test** — Build and run the image to confirm the API:
   ```bash
   docker build -t reasoning-economic-env .
   docker run -p 8000:8000 reasoning-economic-env
   ```
   Then `GET http://localhost:8000/health` and, if desired, a quick reset/step to confirm the env API.

3. **Publish** — From the project root:
   ```bash
   openenv push
   ```
   Optionally: `openenv push --repo-id <org>/reasoning-economic-env` and use `--private` or leave public as needed. Ensure you are logged in (`huggingface-cli login` or set `HF_TOKEN`). If the CLI uses a different flow (e.g. `openenv build` then manual upload), run `openenv build --help` and `openenv push --help` for current options.

## References

- Implementation plan: see project `ReasonBudget_Gym_Implementation_Plan.md` (legacy name); ReasoningEconomicsEnv is the current package name.
- OpenEnv: https://github.com/meta-pytorch/OpenEnv
