"""FastAPI app exposing ReasonBudgetEnvironment via OpenEnv create_app."""

from reasonbudget_gym.env import EnvConfig, ReasonBudgetEnvironment
from reasonbudget_gym.env.models import ReasonBudgetAction, ReasonBudgetObservation

try:
    from openenv.core.env_server import create_app
except ImportError:
    create_app = None  # type: ignore


def _env_factory():
    """Factory that returns a new ReasonBudgetEnvironment instance (for each WebSocket session)."""
    return ReasonBudgetEnvironment(config=EnvConfig())


if create_app is not None:
    app = create_app(
        _env_factory,
        ReasonBudgetAction,
        ReasonBudgetObservation,
        env_name="reasonbudget-gym",
    )
else:
    from fastapi import FastAPI
    app = FastAPI(title="reasonbudget-gym")
    app.get("/health")(lambda: {"status": "ok"})


def main():
    """Entry point for uv run server or python -m."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
