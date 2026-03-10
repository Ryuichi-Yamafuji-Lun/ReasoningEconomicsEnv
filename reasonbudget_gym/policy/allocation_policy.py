"""MLP allocation policy: Gaussian actor (continuous tokens) + value head."""

import torch
import torch.nn as nn
import numpy as np


def _to_scalar(value: object) -> float:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def obs_to_tensor(observation: dict, device: torch.device) -> torch.Tensor:
    """Flatten observation dict to feature vector [embedding(384), remaining_budget_norm, q_rem_norm, budget_per_norm, accuracy_so_far]."""
    emb = np.asarray(observation["question_embedding"], dtype=np.float32).ravel()
    if emb.size != 384:
        emb = np.zeros(384, dtype=np.float32)
    remaining = _to_scalar(observation["remaining_budget"])
    q_rem = _to_scalar(observation["questions_remaining"])
    budget_per = _to_scalar(observation["budget_per_remaining"])
    acc = _to_scalar(observation["accuracy_so_far"])
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
    """3-layer MLP with shared backbone and Gaussian actor/value heads."""

    def __init__(
        self,
        input_dim: int = 388,
        hidden: tuple = (256, 128),
        min_tokens: int = 10,
        max_tokens: int = 800,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.feature_dim = prev
        self.mean_head = nn.Linear(self.feature_dim, 1)
        self.log_std_head = nn.Linear(self.feature_dim, 1)
        self.value = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        mean = self.mean_head(feat)
        log_std = torch.clamp(self.log_std_head(feat), min=-5.0, max=2.0)
        v = self.value(feat)
        return mean, log_std, v

    def get_action(
        self,
        observation: dict,
        device: torch.device,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        x = obs_to_tensor(observation, device)
        with torch.no_grad():
            mean, log_std, v = self.forward(x)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            if deterministic:
                sample = mean
            else:
                sample = dist.sample()
            log_prob = dist.log_prob(sample).sum(dim=-1)
            clamped = torch.clamp(sample, min=float(self.min_tokens), max=float(self.max_tokens))
            action = int(torch.round(clamped).item())
        return action, log_prob, v
