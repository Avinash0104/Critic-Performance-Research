
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from algorithms.networks import Actor, QNetwork, soft_update


def make_probe1_target(obs_dim: int, act_dim: int, hidden_dims, states: torch.Tensor,
                        actions: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Draw the frozen random target function once per run (not per
    checkpoint) - matching Probe 2's fixed random policy - so probe1_ratio's
    trend over time isn't confounded by a different random target's
    difficulty at every checkpoint."""
    target_net = QNetwork(obs_dim, act_dim, hidden_dims[0]).to(device)
    for p in target_net.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        return target_net(states, actions)


def probe_random_target(current_q: QNetwork, obs_dim: int, act_dim: int, hidden_dims, states: torch.Tensor, actions: torch.Tensor, target_y: torch.Tensor, probe_steps: int, lr: float, device: torch.device) -> dict:
    """Probe 1: can a copy of the live critic still be fit to an arbitrary
    new function, compared to a fresh network given the identical task?"""
    y = target_y

    def _fit(net: QNetwork) -> float:
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(probe_steps):
            loss = F.mse_loss(net(states, actions), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return F.mse_loss(net(states, actions), y).item()

    trained_copy = copy.deepcopy(current_q).to(device)
    # Reinitialize only the output head before probing. The question is
    # whether the TRUNK's representation can still support fitting an
    # arbitrary new function - without this, the probe is dominated by how
    # many optimizer steps it takes to unwind the live critic's existing
    # output scale (e.g. ~-500 for a critic matching dataset returns, vs
    # ~0 for a fresh net), which is an optimization artifact, not plasticity.
    trained_copy.head = type(trained_copy.head)(
        trained_copy.head.in_features, trained_copy.head.out_features
    ).to(device)
    trained_error = _fit(trained_copy)

    fresh_net = QNetwork(obs_dim, act_dim, hidden_dims[0]).to(device)
    fresh_error = _fit(fresh_net)

    return {
        "trained_error": trained_error,
        "fresh_error": fresh_error,
        "ratio": fresh_error / max(trained_error, 1e-12),
    }


def probe_td_other_policy(current_q: QNetwork, obs_dim: int, act_dim: int, hidden_dims, random_policy: Actor, minibatches: list, lr: float, gamma: float, tau: float, device: torch.device) -> dict:
    """Probe 2: can a copy of the live critic still be fit via TD to a
    different policy's values, compared to a fresh
    network given the identical transitions and bootstrap policy?"""

    def _run(net: QNetwork, target_net: QNetwork) -> float:
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        last_loss = None
        for s, a, r, s2, d in minibatches:
            with torch.no_grad():
                a2, _ = random_policy.sample(s2)
                y = r + gamma * (1 - d) * target_net(s2, a2)
            loss = F.mse_loss(net(s, a), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            soft_update(target_net, net, tau)
            last_loss = loss.item()
        return last_loss

    trained_copy = copy.deepcopy(current_q).to(device)
    trained_target = copy.deepcopy(trained_copy).to(device)
    trained_error = _run(trained_copy, trained_target)

    fresh_net = QNetwork(obs_dim, act_dim, hidden_dims[0]).to(device)
    fresh_target = copy.deepcopy(fresh_net).to(device)
    fresh_error = _run(fresh_net, fresh_target)

    return {
        "trained_error": trained_error,
        "fresh_error": fresh_error,
        "ratio": fresh_error / max(trained_error, 1e-12),
    }


def sample_td_minibatches(buffer, n: int, batch_size: int, device: torch.device, generator: torch.Generator) -> list:
    """Pre-sample a fixed sequence of (s,a,r,s2,d) minibatches once, so both
    probe runs within a single diagnostic call train on identical data."""
    minibatches = []
    for _ in range(n):
        s, a, r, s2, d, _ = buffer.sample(batch_size, device, generator)
        minibatches.append((s, a, r, s2, d))
    return minibatches
