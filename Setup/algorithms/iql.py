"""IQL: Implicit Q-Learning.

V(s) is fit by expectile regression against a target (lagged) twin-Q
estimate on *dataset* actions only - this is what lets IQL avoid ever
querying Q at out-of-distribution actions, so no explicit conservative
penalty is needed (unlike Cal-QL/CQL). Q(s,a) is a plain TD regression
using V(s') as the next-state value. The actor is extracted afterwards by
advantage-weighted regression (AWR): states/actions with positive
advantage (Q - V) under the learned critic get up-weighted in a
log-likelihood objective.

Online fine-tuning reuses the exact same losses as offline (standard IQL
practice) - the only difference between phases is where the data comes
from.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from algorithms.networks import Actor, Critic, VNetwork, soft_update


class IQL:
    def __init__(self, obs_dim, act_dim, action_scale, action_bias, hidden_dims, device, lr=3e-4, gamma=0.99, tau=0.005, expectile=0.7, awr_beta=3.0, awr_clip=100.0, seed=0):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.awr_beta = awr_beta
        self.awr_clip = awr_clip

        self.actor = Actor(obs_dim, act_dim, action_scale, action_bias, hidden_dims[0]).to(device)
        self.critic = Critic(obs_dim, act_dim, hidden_dims[0]).to(device)
        self.target_critic = Critic(obs_dim, act_dim, hidden_dims[0]).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = VNetwork(obs_dim, hidden_dims[0]).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

    @staticmethod
    def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return (weight * diff.pow(2)).mean()

    def update(self, batch) -> dict:
        s, a, r, s2, d, _ = batch  # mc_returns unused - only Cal-QL's calibration needs it

        with torch.no_grad():
            target_q = self.target_critic.min(s, a)
            # Pre-update V(s'), so the Q-loss bootstrap target uses the same
            # V network snapshot the rest of this step is based on, rather
            # than one that's already been nudged by this step's V update.
            next_v = self.value(s2)

        v = self.value(s)
        v_for_advantage = v.detach()  # pre-update V(s), reused below for the actor
        v_loss = self._expectile_loss(target_q - v, self.expectile)
        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()

        y = r + self.gamma * (1 - d) * next_v
        q1, q2 = self.critic(s, a)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        advantage = target_q - v_for_advantage
        weight = torch.exp(self.awr_beta * advantage).clamp(max=self.awr_clip)
        log_prob = self.actor.score(s, a)
        actor_loss = -(weight * log_prob).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.target_critic, self.critic, self.tau)

        return {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_advantage": advantage.mean().item(),
        }
