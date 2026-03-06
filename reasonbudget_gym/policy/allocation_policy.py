"""MLP allocation policy: actor (action logits) + value head for PPO."""

import torch
import torch.nn as nn
import numpy as np


def obs_to_tensor(observation: dict, device: torch.device) -> torch.Tensor:
    """Flatten observation dict to feature vector [embedding(384), remaining_budget_norm, q_rem_norm, budget_per_norm, accuracy_so_far]."""
    emb = np.asarray(observation["question_embedding"], dtype=np.float32).ravel()
    if emb.size != 384:
        emb = np.zeros(384, dtype=np.float32)
    remaining = float(observation["remaining_budget"][0])
    q_rem = float(observation["questions_remaining"])
    budget_per = float(observation["budget_per_remaining"][0])
    acc = float(observation["accuracy_so_far"][0])
    # Normalize for stability
    remaining_norm = min(1.0, remaining / 5000.0)
    q_rem_norm = min(1.0, q_rem / 20.0)
    budget_per_norm = min(1.0, budget_per / 1000.0)
    return torch.from_numpy(
        np.concatenate([
            emb[:384],
            [remaining_norm, q_rem_norm, budget_per_norm, acc],
        ]).astype(np.float32)
    ).unsqueeze(0).to(device)


class AllocationPolicy(nn.Module):
    """3-layer MLP with shared backbone and actor/value heads. Input 388 -> 256 -> 128 -> 5 (actor) / 1 (value)."""

    def __init__(self, input_dim: int = 388, n_actions: int = 5, hidden: tuple = (256, 128)):
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.feature_dim = prev
        self.actor = nn.Linear(self.feature_dim, n_actions)
        self.value = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        logits = self.actor(feat)
        v = self.value(feat)
        return logits, v

    def get_action(self, observation: dict, device: torch.device, deterministic: bool = False) -> tuple[int, torch.Tensor, torch.Tensor]:
        x = obs_to_tensor(observation, device)
        with torch.no_grad():
            logits, v = self.forward(x)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                action = dist.sample().item()
        return action, logits, v
