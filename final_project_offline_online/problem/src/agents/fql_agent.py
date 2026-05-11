from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


def weighted_mean(values: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if sample_weights is None:
        return values.mean()
    if values.shape[0] == sample_weights.shape[0]:
        weights = sample_weights.view(-1, *([1] * (values.dim() - 1)))
    else:
        weights = sample_weights.view(1, -1, *([1] * (values.dim() - 2)))
    return (values * weights).sum() / (weights.sum().clamp_min(1e-8) * values.numel() / sample_weights.numel())


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return weighted_mean((prediction - target).pow(2), sample_weights)


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha
        self.loss_fn = nn.MSELoss()

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # TODO(student): Compute the action for evaluation
        # Hint: Unlike SAC+BC and IQL, the evaluation action is *sampled* (i.e., not the mode or mean) from the policy
        noise = torch.randn(observation.shape[0], self.action_dim, device=observation.device)
        t0 = torch.zeros((observation.shape[0], 1), device=observation.device)
        action = noise + self.onestep_actor(observation, noise, t0)
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        action = noise
        dt = 1.0 / self.flow_steps
        for k in range(self.flow_steps):
            t = torch.full((action.shape[0], 1), k * dt, device=action.device)
            vel = self.bc_actor(observation, action, t)
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
        sample_weights: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Compute the Q loss
        # Hint: Use the one-step actor to compute next actions
        # Hint: Remember to clamp the actions to be in [-1, 1] when feeding them to the critic!
        with torch.no_grad():
            noise = torch.randn(next_observations.shape[0], self.action_dim, device=next_observations.device)
            t = torch.zeros((next_observations.shape[0], 1), device=next_observations.device)
            next_action = torch.clamp(noise + self.onestep_actor(next_observations, noise, t), -1, 1)

            q_next = self.target_critic(next_observations, next_action).min(dim=0).values
            target_q = rewards + self.discount * (1 - dones) * q_next
        
        actions = torch.clamp(actions, -1, 1)
        q = self.critic(observations, actions)
        loss = weighted_mse(q, target_q.unsqueeze(0).expand_as(q), sample_weights)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ):
        """
        Update the BC actor
        """
        # TODO(student): Compute the BC flow loss
        noise = torch.randn_like(actions)
        t = torch.rand((actions.shape[0], 1), device=actions.device)
        x_t = (1.0 - t) * noise + t * actions
        v_target = actions - noise
        v_pred = self.bc_actor(observations, x_t, t)
        loss = weighted_mse(v_pred, v_target, sample_weights)

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ):
        """
        Update the one-step actor
        """
        # TODO(student): Compute the one-step actor loss
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss
        noise = torch.randn_like(actions)
        with torch.no_grad():
            bc_action = self.get_bc_action(observations, noise)
        t0 = torch.zeros((actions.shape[0], 1), device=actions.device)
        pred = noise + self.onestep_actor(observations, noise, t0)  # unclipped for distill
        distill_loss = self.alpha * weighted_mse(pred, bc_action, sample_weights)

        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        clipped = torch.clamp(pred, -1, 1)
        q_val = self.critic(observations, clipped)
        q_loss = -weighted_mean(q_val, sample_weights)

        # Total loss.
        loss = distill_loss + q_loss

        # Additional metrics for logging.
        mse = weighted_mse(pred, actions, sample_weights)

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
        sample_weights: Optional[torch.Tensor] = None,
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones, sample_weights)
        metrics_bc_actor = self.update_bc_actor(observations, actions, sample_weights)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions, sample_weights)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                p_targ.mul_(1.0 - self.target_update_rate)
                p_targ.add_(self.target_update_rate * p)
