
from __future__ import annotations

import torch

from algorithms.networks import QNetwork


@torch.no_grad()
def _layer_activation_stats(activations: torch.Tensor, tau: float) -> dict:
    """`activations`: (batch, units) post-ReLU for one hidden layer."""
    fau = (activations > 0).float().mean().item()

    fire_rate = (activations > 0).float().mean(dim=0)  # per-unit, over batch
    dead_fraction = (fire_rate == 0).float().mean().item()
    linearized_fraction = (fire_rate == 1).float().mean().item()

    magnitude = activations.abs().mean(dim=0)  # per-unit average |activation|
    mean_magnitude = magnitude.mean().clamp_min(1e-12)
    normalized = magnitude / mean_magnitude
    dormant_ratio = (normalized <= tau).float().mean().item()

    return {
        "fau": fau,
        "dormant_ratio": dormant_ratio,
        "dead_fraction": dead_fraction,
        "linearized_fraction": linearized_fraction,
    }


@torch.no_grad()
def per_layer_activation_diagnostics(q_network: QNetwork, states: torch.Tensor, actions: torch.Tensor, tau: float = 0.1) -> dict:
    """FAU, dormant ratio, dead fraction, linearized fraction for every
    hidden layer, plus a mean-across-layers rollup per metric."""
    activations: list[torch.Tensor] = []
    q_network.features(states, actions, activations=activations)
    per_layer = [_layer_activation_stats(a, tau) for a in activations]
    keys = per_layer[0].keys()
    mean = {k: sum(layer[k] for layer in per_layer) / len(per_layer) for k in keys}
    return {"per_layer": per_layer, "mean": mean}


@torch.no_grad()
def effective_rank(q_network: QNetwork, states: torch.Tensor, actions: torch.Tensor, eps: float = 0.01) -> int:
    """SRank of the penultimate (last captured hidden) layer's features."""
    activations: list[torch.Tensor] = []
    q_network.features(states, actions, activations=activations)
    feats = activations[-1]
    singular_values = torch.linalg.svdvals(feats)
    total = singular_values.sum()
    if total <= 0:
        return 0
    cumulative = torch.cumsum(singular_values, dim=0)
    threshold = (1 - eps) * total
    k = int(torch.searchsorted(cumulative, threshold).item()) + 1
    return min(k, singular_values.numel())


@torch.no_grad()
def weight_norm(q_network: QNetwork) -> float:
    total = 0.0
    for layer in q_network.trunk:
        total += layer.weight.pow(2).sum().item()
        total += layer.bias.pow(2).sum().item()
    total += q_network.head.weight.pow(2).sum().item()
    total += q_network.head.bias.pow(2).sum().item()
    return total ** 0.5
