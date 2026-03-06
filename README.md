# ReasonBudget-Gym (ReasoningEconomicsEnv)

RL environment for sequential reasoning budget allocation. The project is **OpenEnv-only**: no gymnasium, no stable-baselines3. Baselines and training use `ReasonBudgetObservation` and `ReasonBudgetAction` only.

## Interface

- **Core env**: `ReasonBudgetEnvironment` (OpenEnv `Environment` subclass). Use in-process:
  - `obs = env.reset(seed=..., episode_id=...)` → `ReasonBudgetObservation`
  - `obs = env.step(ReasonBudgetAction(budget_allocation=index))` → `ReasonBudgetObservation` (with `obs.reward`, `obs.done`)
  - `state = env.state` → `ReasonBudgetState`
- **Deployment**: OpenEnv HTTP/WebSocket server in `reasonbudget_gym.server.app`. For remote use, use OpenEnv's generic client with dict actions/observations, e.g. `step({"budget_allocation": 2})`. See `reasonbudget_gym.client` for a short example.

## Install

```bash
pip install -e .
```

## Quick start

```python
from reasonbudget_gym.env import (
    EnvConfig,
    ReasonBudgetEnvironment,
    ReasonBudgetAction,
    ReasonBudgetObservation,
)
env = ReasonBudgetEnvironment(config=EnvConfig(num_questions=10))
obs = env.reset(seed=42)
while not obs.done:
    action = ReasonBudgetAction(budget_allocation=2)  # tier index 0..4
    obs = env.step(action)
    print(obs.reward, obs.done)
print(env.state.total_correct, env.state.questions_answered)
```

## Run server

```bash
uvicorn reasonbudget_gym.server.app:app --host 0.0.0.0 --port 8000
```

## References

- Implementation plan: see project `ReasonBudget_Gym_Implementation_Plan.md`.
- OpenEnv: https://github.com/meta-pytorch/OpenEnv
