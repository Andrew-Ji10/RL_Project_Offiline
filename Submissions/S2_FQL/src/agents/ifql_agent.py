from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class IFQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor_flow,
        make_actor_flow_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        online_training: bool = False,
        num_samples: int = 32,
        expectile: float = 0.9,
        rho: float = 0.5,
    ):
        super().__init__()

        self.action_dim = action_dim
        # TODO(student): Create flow actor

        self.actor_flow = make_actor_flow(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = make_value(observation_shape)
        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions), and value function

        # TODO(student): Create optimizers for all the above models
        self.actor_flow_optimizer = make_actor_flow_optimizer(self.actor_flow.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.num_samples = num_samples
        self.expectile = expectile
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def expectile_loss(adv: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Compute the expectile loss for IFQL
        """
        # TODO(student): Implement the expectile loss
        weight = torch.where(adv >= 0, expectile, 1.0 - expectile)
        return (weight * adv.pow(2)).mean()

    @torch.compile
    def update_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """
        Update value function
        """
        # TODO(student): Implement the value function update
        
        # TODO(student): Update value function
        with torch.no_grad():
            actions_clamped = torch.clamp(actions, -1, 1)
            q = self.target_critic(observations, actions_clamped).min(dim=0).values

        v = self.value(observations)
        adv = q - v
        loss = self.expectile_loss(adv, self.expectile)

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        return {
            "value_loss": loss,
            "v_mean": v.mean(),
            "adv_mean": adv.mean(),
            "adv_max": adv.max(),
            "adv_min": adv.min(),
        }

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Rejection / best-of-n sampling using the flow policy and critic.

        We:
          1. Sample multiple candidate actions via the BC flow.
          2. Evaluate them with the critic.
          3. Pick the action with the highest Q-value.
        """
        # TODO(student): Implement the rejection sampling
        B = observations.shape[0]
        N = self.num_samples
        D = self.action_dim

        obs_rep = observations.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        noise = torch.randn(B * N, D, device=observations.device)
        candidates = self.get_flow_action(obs_rep, noise)

        q = self.critic(obs_rep, candidates).mean(dim=0).view(B, N)
        best_idx = q.argmax(dim=1)

        candidates = candidates.view(B, N, D)
        chosen = candidates[torch.arange(B, device=observations.device), best_idx]
        return chosen

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation. Returns the best-of-N rejection-sampled action.
        """
        # TODO(student): Implement get action
        if isinstance(observation, tuple):
            observation = observation[0]
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.sample_actions(observation)
        return ptu.to_numpy(action[0])

    @torch.compile
    def get_flow_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Compute the flow action using Euler integration for `self.flow_steps` steps.
        """
        # TODO(student): Implement euler integration to get flow action
        action = noise
        dt = 1.0 / self.flow_steps
        for k in range(self.flow_steps):
            t = torch.full((action.shape[0], 1), k * dt, device=action.device)
            vel = self.actor_flow(observation, action, t)
            action = action + dt * vel
        return torch.clamp(action, -1, 1)

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a) using the learned value function for bootstrapping,
        as in IFQL / IQL-style critic training.
        """
        # TODO(student): Implement Q-function update
        
        # TODO(student): Update Q-function
        with torch.no_grad():
            v_next = self.value(next_observations)
            target_q = rewards + self.discount * (1.0 - dones) * v_next

        actions_clamped = torch.clamp(actions, -1, 1)
        q_pred = self.critic(observations, actions_clamped)
        loss = self.loss_fn(q_pred, target_q.unsqueeze(0).expand_as(q_pred))

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q_pred.mean(),
            "q_max": q_pred.max(),
            "q_min": q_pred.min(),
            "target_q_mean": target_q.mean(),
        }


    @torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the flow actor using the velocity matching loss.
        """
        # TODO(student): Implement flow actor update
        
        # TODO(student): Update flow actor
        noise = torch.randn_like(actions)
        t = torch.rand((actions.shape[0], 1), device=actions.device)
        x_t = (1.0 - t) * noise + t * actions
        v_target = actions - noise
        v_pred = self.actor_flow(observations, x_t, t)
        loss = self.loss_fn(v_pred, v_target)

        self.actor_flow_optimizer.zero_grad()
        loss.backward()
        self.actor_flow_optimizer.step()

        return {
            "loss": loss,
        }


    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_v = self.update_value(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                p_targ.data.mul_(1.0 - self.target_update_rate)
                p_targ.data.add_(self.target_update_rate * p.data)
