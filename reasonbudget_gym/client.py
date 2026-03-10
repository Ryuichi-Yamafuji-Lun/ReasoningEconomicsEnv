"""Remote use: connect to the ReasonBudget OpenEnv server via OpenEnv's generic client.

This package does not ship a custom typed client. For remote env interaction, use
OpenEnv's generic client with dict actions and observations. Example:

    from openenv.core.env_client import EnvClient  # or GenericEnvClient, per OpenEnv API
    client = EnvClient(base_url="http://localhost:8000")  # adjust to your server
    obs = client.reset(seed=42)
    while not obs.get("done", False):
        action = {"token_allocation": 350}  # direct token allocation
        obs = client.step(action)
        print(obs.get("reward"), obs.get("done"))

Action and observation schemas match ReasonBudgetAction and ReasonBudgetObservation
in reasonbudget_gym.env.models (e.g. observation has question_embedding, remaining_budget,
questions_remaining, step_idx, budget_per_remaining, accuracy_so_far, done, reward).
"""
