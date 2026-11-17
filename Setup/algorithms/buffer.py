"""Fixed-size and growing replay buffers, both storing Monte-Carlo
return-to-go alongside transitions (needed by Cal-QL's calibration term).
"""
from __future__ import annotations

import numpy as np
import torch

# Sentinel mc_return for online-collected transitions (their episode hasn't
# finished when they're added, so no real Monte-Carlo return exists yet).
# max(Q, NO_CALIBRATION) is a no-op, so Cal-QL's calibration term only ever
# binds on genuine offline-buffer transitions.
NO_CALIBRATION = -1e8


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.mc_returns = np.zeros((capacity, 1), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, obs, action, reward, next_obs, done, mc_return):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.mc_returns[i] = mc_return
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(self, obs, actions, rewards, next_obs, dones, gamma: float):
        """Add a full episode, computing discounted return-to-go backward."""
        n = len(rewards)
        mc_returns = np.zeros(n, dtype=np.float32)
        running = 0.0
        for t in reversed(range(n)):
            running = rewards[t] + gamma * running * (1.0 - dones[t])
            mc_returns[t] = running
        for t in range(n):
            self.add(obs[t], actions[t], rewards[t], next_obs[t], dones[t], mc_returns[t])

    def sample(self, batch_size: int, device: torch.device, generator: torch.Generator | None = None):
        idx = torch.randint(0, self.size, (batch_size,), generator=generator).numpy()
        return (
            torch.as_tensor(self.obs[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_obs[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
            torch.as_tensor(self.mc_returns[idx], device=device),
        )

    def save(self, path: str):
        np.savez(
            path,
            obs=self.obs[: self.size], actions=self.actions[: self.size],
            rewards=self.rewards[: self.size], next_obs=self.next_obs[: self.size],
            dones=self.dones[: self.size], mc_returns=self.mc_returns[: self.size],
        )

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        data = np.load(path)
        n, obs_dim = data["obs"].shape
        act_dim = data["actions"].shape[1]
        buffer = cls(n, obs_dim, act_dim)
        buffer.obs[:] = data["obs"]
        buffer.actions[:] = data["actions"]
        buffer.rewards[:] = data["rewards"]
        buffer.next_obs[:] = data["next_obs"]
        buffer.dones[:] = data["dones"]
        buffer.mc_returns[:] = data["mc_returns"]
        buffer.size = n
        buffer.ptr = 0
        return buffer


class MixedBuffer:
    """Samples a fixed fraction of each batch from a frozen offline buffer
    and the rest from a growing online buffer - standard offline-to-online
    fine-tuning practice (retain offline data during online phase)."""

    def __init__(self, offline_buffer: ReplayBuffer, online_capacity: int, offline_frac: float = 0.5):
        self.offline_buffer = offline_buffer
        self.online_buffer = ReplayBuffer(online_capacity, offline_buffer.obs.shape[1], offline_buffer.actions.shape[1])
        self.offline_frac = offline_frac

    def add(self, *args, **kwargs):
        self.online_buffer.add(*args, **kwargs)

    def sample(self, batch_size: int, device: torch.device, generator: torch.Generator | None = None):
        if self.online_buffer.size == 0:
            return self.offline_buffer.sample(batch_size, device, generator)
        n_offline = int(batch_size * self.offline_frac)
        n_online = batch_size - n_offline
        off = self.offline_buffer.sample(n_offline, device, generator)
        on = self.online_buffer.sample(n_online, device, generator)
        return tuple(torch.cat([a, b], dim=0) for a, b in zip(off, on))
