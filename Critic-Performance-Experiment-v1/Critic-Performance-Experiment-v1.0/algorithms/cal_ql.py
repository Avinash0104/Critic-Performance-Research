"""Cal-QL: CQL-style conservative offline pretraining, calibrated against a
Monte-Carlo return-to-go lower bound, with online fine-tuning that reuses
the exact same loss (Cal-QL's point is that nothing needs to change between
offline and online).

Simplification vs. the original paper: alpha_cql (the CQL penalty weight)
is a fixed hyperparameter here rather than Lagrange-auto-tuned - adequate
for a 2-layer MLP on a 3-D observation, and keeps the implementation
legible.

Sentinel: online-collected transitions don't have a real Monte-Carlo return
yet (their episode hasn't finished when they're added), so they're stored
with mc_return = NO_CALIBRATION, which makes the calibration max() a no-op
for them while still applying normally to offline data.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from algorithms.networks import Actor, Critic, soft_update
from algorithms.buffer import NO_CALIBRATION  # re-exported for callers


class CalQL:
    def __init__(self, obs_dim, act_dim, action_scale, action_bias, hidden_dims, device, lr=3e-4, gamma=0.99, tau=0.005, alpha_cql=5.0, num_cql_samples=4, seed=0):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha_cql = alpha_cql
        self.num_cql_samples = num_cql_samples
        self.generator = torch.Generator().manual_seed(seed)

        self.actor = Actor(obs_dim, act_dim, action_scale, action_bias, hidden_dims[0]).to(device)
        self.critic = Critic(obs_dim, act_dim, hidden_dims[0]).to(device)
        self.target_critic = Critic(obs_dim, act_dim, hidden_dims[0]).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -act_dim

    def _cql_penalty_term(self, s: torch.Tensor, mc_returns: torch.Tensor):
        batch, n = s.shape[0], self.num_cql_samples
        s_rep = s.unsqueeze(1).expand(batch, n, -1).reshape(batch * n, -1)

        with torch.no_grad():
            policy_actions, _ = self.actor.sample(s_rep)
        low = self.actor.action_bias - self.actor.action_scale
        high = self.actor.action_bias + self.actor.action_scale
        random_actions = torch.rand(batch * n, policy_actions.shape[-1], device=s.device,
                                     generator=self.generator) * (high - low) + low

        q1_pi, q2_pi = self.critic(s_rep, policy_actions)
        q1_rand, q2_rand = self.critic(s_rep, random_actions)

        mc_rep = mc_returns.unsqueeze(1).expand(batch, n, 1).reshape(batch * n, 1)
        q1_pi = torch.maximum(q1_pi, mc_rep).view(batch, n)
        q2_pi = torch.maximum(q2_pi, mc_rep).view(batch, n)
        q1_rand = torch.maximum(q1_rand, mc_rep).view(batch, n)
        q2_rand = torch.maximum(q2_rand, mc_rep).view(batch, n)

        logsumexp_q1 = torch.logsumexp(torch.cat([q1_pi, q1_rand], dim=1), dim=1, keepdim=True)
        logsumexp_q2 = torch.logsumexp(torch.cat([q2_pi, q2_rand], dim=1), dim=1, keepdim=True)
        return logsumexp_q1, logsumexp_q2

    def update(self, batch) -> dict:
        s, a, r, s2, d, mc_returns = batch
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(s2)
            target_q = self.target_critic.min(s2, next_action) - alpha * next_log_prob
            y = r + self.gamma * (1 - d) * target_q

        q1, q2 = self.critic(s, a)
        td_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        logsumexp_q1, logsumexp_q2 = self._cql_penalty_term(s, mc_returns)
        cql_loss = (logsumexp_q1 - q1).mean() + (logsumexp_q2 - q2).mean()

        critic_loss = td_loss + self.alpha_cql * cql_loss
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_action, log_prob = self.actor.sample(s)
        q_new = self.critic.min(s, new_action)
        actor_loss = (alpha * log_prob - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        soft_update(self.target_critic, self.critic, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item(),
        }
