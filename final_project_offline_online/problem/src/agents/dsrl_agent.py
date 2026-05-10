from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Sequence


class DSRLAgent(nn.Module):
    """DSRL agent - https://arxiv.org/abs/2506.15799"""

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_flow_actor,
        make_bc_flow_actor_optimizer,
        make_noise_actor,
        make_noise_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_z_critic=None,
        make_z_critic_optimizer=None,

        discount: float = 0.99,
        target_update_rate: float = 0.005,
        flow_steps: int = 10,
        noise_scale: float = 1.0,

        online_training: bool = False,
        **kwargs
    ):
        super().__init__()

        self.action_dim = action_dim
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.noise_scale = noise_scale
        self.target_entropy = -action_dim

        if make_z_critic is None:
            make_z_critic = kwargs.get('make_noise_critic', make_critic)
        if make_z_critic_optimizer is None:
            make_z_critic_optimizer = kwargs.get('make_noise_critic_optimizer', make_critic_optimizer)

        # Create BC flow actor and target BC flow actor
        self.bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor.load_state_dict(self.bc_flow_actor.state_dict())

        # Create noise policy
        self.noise_actor = make_noise_actor(observation_shape, action_dim)

        # Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions), and z critic (for noise policy)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.z_critic = make_z_critic(observation_shape, action_dim)

        # Create learnable entropy coefficient
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))

        # Create optimizers for all the above models
        self.bc_flow_actor_optimizer = make_bc_flow_actor_optimizer(self.bc_flow_actor.parameters())
        self.noise_actor_optimizer = make_noise_actor_optimizer(self.noise_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.z_critic_optimizer = make_z_critic_optimizer(self.z_critic.parameters())
        
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.to(ptu.device)

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    @torch.compiler.disable
    def sample_flow_actions(self, observations: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """Euler integration of BC flow from t=0 to t=1."""
        dt = 1.0 / self.flow_steps
        a = noises.clone()
        for i in range(self.flow_steps):
            # t must be [B, 1] for torch.cat in the VectorFieldPolicy
            t = torch.ones((observations.shape[0], 1), device=ptu.device) * (i * dt)
            v = self.target_bc_flow_actor(observations, a, t)
            a = a + v * dt
        return a

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample actions using noise policy for noise input to BC flow policy."""
        dist = self.noise_actor(observations)
        z = dist.sample()
        return self.sample_flow_actions(observations, z * self.noise_scale)
    
    def get_action(self, observation: np.ndarray):
        """Used for evaluation."""
        obs = torch.tensor(observation, dtype=torch.float32, device=ptu.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.noise_actor(obs)

            z = dist.mean 
            action = self.sample_flow_actions(obs, z * self.noise_scale)
        return action.squeeze(0).cpu().numpy()

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """Update critic"""
        with torch.no_grad():
            z = torch.randn((next_observations.shape[0], self.action_dim), device=ptu.device)
            next_a = self.sample_flow_actions(next_observations, z)
            q1_target, q2_target = self.target_critic(next_observations, next_a)
            next_q = (q1_target + q2_target) / 2.0
            
            target_q = rewards.view(-1) + self.discount * (1.0 - dones.float().view(-1)) * next_q.view(-1)

        q1, q2 = self.critic(observations, actions)
        loss = ((q1.view(-1) - target_q)**2).mean() + ((q2.view(-1) - target_q)**2).mean()
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        return {"q_loss": loss.item(), "q1": q1.mean().item()}
    
    def update_qz(self, 
        observations: torch.Tensor,
        **kwargs
    ) -> dict:
        """Update z_critic."""
        with torch.no_grad():
            z = torch.randn((observations.shape[0], self.action_dim), device=ptu.device)
            a_bc = self.sample_flow_actions(observations, z)
            q1, q2 = self.critic(observations, a_bc)
            target_qz = (q1 + q2) / 2.0

        qz1, qz2 = self.z_critic(observations, z)
        loss = ((qz1.view(-1) - target_qz.view(-1))**2).mean() + ((qz2.view(-1) - target_qz.view(-1))**2).mean()

        self.z_critic_optimizer.zero_grad()
        loss.backward()
        self.z_critic_optimizer.step()
        
        return {"qz_loss": loss.item()}

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """Update BC flow actor"""
        z = torch.randn_like(actions)
        # t must be [B, 1] for broadcasting and VectorFieldPolicy
        t = torch.rand((actions.shape[0], 1), device=ptu.device)
        a_tilde = (1 - t) * z + t * actions
        v_pred = self.bc_flow_actor(observations, a_tilde, t)
        loss = ((v_pred - (actions - z))**2).mean()
        
        self.bc_flow_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_flow_actor_optimizer.step()
        
        return {"bc_actor_loss": loss.item()}
    
    def update_noise_actor(self,
        observations: torch.Tensor,
    ) -> dict:
        """Update noise actor."""
        dist = self.noise_actor(observations)
        z = dist.rsample()
        log_pi = dist.log_prob(z)
        if len(log_pi.shape) > 1:
            log_pi = log_pi.sum(dim=-1)
            
        qz1, qz2 = self.z_critic(observations, z * self.noise_scale)
        qz = torch.min(qz1, qz2).view(-1)
        
        loss = (self.alpha.detach() * log_pi.view(-1) - qz).mean()
        
        self.noise_actor_optimizer.zero_grad()
        loss.backward()
        self.noise_actor_optimizer.step()
        
        self.last_log_pi = log_pi.detach().view(-1)
        return {"noise_actor_loss": loss.item()}

    def update_alpha(self) -> dict:
        """Update alpha."""
        if not hasattr(self, 'last_log_pi'):
            return {"alpha_loss": 0.0, "alpha": self.alpha.item()}

        loss = -(self.alpha * (self.last_log_pi + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()
        
        return {"alpha_loss": loss.item(), "alpha": self.alpha.item()}

    def update_target_critic(self) -> None:
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.target_update_rate * param.data + (1.0 - self.target_update_rate) * target_param.data)

    def update_target_bc_flow_actor(self) -> None:
        for param, target_param in zip(self.bc_flow_actor.parameters(), self.target_bc_flow_actor.parameters()):
            target_param.data.copy_(self.target_update_rate * param.data + (1.0 - self.target_update_rate) * target_param.data)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_qz = self.update_qz(observations)
        metrics_actor = self.update_actor(observations, actions)
        metrics_noise_actor = self.update_noise_actor(observations)
        metrics_alpha = self.update_alpha()
        
        metrics = {
            **{f"critic/{k}": v for k, v in metrics_q.items()},
            **{f"z_critic/{k}": v for k, v in metrics_qz.items()},
            **{f"actor/{k}": v for k, v in metrics_actor.items()},
            **{f"noise_actor/{k}": v for k, v in metrics_noise_actor.items()},
            **{f"alpha/{k}": v for k, v in metrics_alpha.items()},
        }

        self.update_target_critic()
        self.update_target_bc_flow_actor()

        return metrics