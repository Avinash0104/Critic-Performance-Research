"""Actor/critic/value networks shared by Cal-QL and IQL."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class Actor(nn.Module):
    """Squashed Gaussian policy over a bounded continuous action space."""

    def __init__(self, obs_dim: int, action_dim: int, action_scale: float, action_bias: float, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.trunk:
            h = layer(h)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action from the distribution mean (used for eval)."""
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def sample(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        raw_action = dist.rsample()
        squashed = torch.tanh(raw_action)
        action = squashed * self.action_scale + self.action_bias

        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(self.action_scale * (1 - squashed.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def score(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        unscaled = ((action - self.action_bias) / self.action_scale).clamp(-0.999999, 0.999999)
        raw_action = torch.atanh(unscaled)

        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(self.action_scale * (1 - unscaled.pow(2)) + 1e-6)
        return log_prob.sum(-1, keepdim=True)


class QNetwork(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def features(self, obs: torch.Tensor, action: torch.Tensor, activations: list | None = None) -> torch.Tensor:
        h = torch.cat([obs, action], dim=-1)
        for layer in self.trunk:
            h = F.relu(layer(h))
            if activations is not None: activations.append(h)
        return h

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(obs, action))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim=256):
        super().__init__()
        self.q1 = QNetwork(obs_dim, act_dim, hidden_dim)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        return self.q1(obs, action), self.q2(obs, action)

    def min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs, action)
        return torch.min(q1, q2)


class VNetwork(nn.Module):
    """State-value network used only by IQL."""

    def __init__(self, obs_dim: int, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data = (1 - tau) * tp.data + tau * sp.data
