import math
from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List

class QSMAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        inv_temp: float,
        flow_steps: int,
    ):
        super().__init__()

        self.action_dim = action_dim
        # TODO(student): Create actor
        
        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions)
        
        # TODO(student): Create optimizers for all the above models

        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.inv_temp = inv_temp
        self.flow_steps = flow_steps
        self.loss_fn = nn.MSELoss()

        betas = self.cosine_beta_schedule(flow_steps)
        alphas = 1.0 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)
        # TODO(student): Implement betas, alphas, and alpha_hats
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hats", alpha_hats)

        self.to(ptu.device)
    
    def cosine_beta_schedule(self, timesteps: int, s: float = 0.08) -> torch.Tensor:
        """
        Cosine annealing beta schedule
        """
        # TODO(student): Implement cosine annealing beta schedule
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alpha_bar = torch.cos(((t / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clamp(betas, min=1e-4, max=0.999)
    
    @torch.compiler.disable
    def ddpm_sampler(self, observations: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        DDPM sampling with x0-prediction clipping.

        At every reverse step we recover the implied clean action x0 from
        eps_pred, clamp it into the action box [-1, 1], and then reconstruct
        the posterior mean q(x_{t-1} | x_t, x0) from that clamped x0 plus the
        current noisy x_t. This is mathematically equivalent to the standard
        eps-form reverse step but stays numerically stable when flow_steps is
        small (here T=10), because the eps-form coefficient 1/sqrt(alpha_t)
        blows up at the last reverse step where alpha_t ~ 0.
        """
        # TODO(student): Implement DDPM sampling
        T = self.flow_steps
        x = noise
        B = x.shape[0]
        for ti in range(T - 1, -1, -1):
            t_norm = torch.full((B, 1), (ti + 1) / T, device=x.device, dtype=x.dtype)
            eps_pred = self.actor(observations, x, t_norm)

            beta_t = self.betas[ti]
            alpha_t = self.alphas[ti]
            alpha_hat_t = self.alpha_hats[ti]
            # coef = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_hat_t)
            # mean = (x - coef * eps_pred) / torch.sqrt(alpha_t)
            # we use the actor to predict the clean action and clip it 
            x0_pred = (x - torch.sqrt(1.0 - alpha_hat_t) * eps_pred) / torch.sqrt(alpha_hat_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if ti > 0:
                #here we create new posterior using the predicted clipped x0 and noise
                alpha_hat_prev = self.alpha_hats[ti - 1]
                coef_x0 = torch.sqrt(alpha_hat_prev) * beta_t / (1.0 - alpha_hat_t)
                coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_hat_prev) / (1.0 - alpha_hat_t)
                mean = coef_x0 * x0_pred + coef_xt * x
                z = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * z
            else:
                x = x0_pred
        return torch.clamp(x, -1.0, 1.0)

    def get_action(self, observation):
        """
        Used for evaluation.
        """
        # TODO(student): Implement get_action
        if isinstance(observation, tuple):
            observation = observation[0]
        observation = ptu.from_numpy(np.asarray(observation))[None]
        noise = torch.randn(observation.shape[0], self.action_dim, device=observation.device)
        with torch.no_grad():
            action = self.ddpm_sampler(observation, noise)
        return ptu.to_numpy(action)[0]

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
        Update Critic
        """
        # TODO(student): Implement critic update
        
        # TODO(student): Update critic
        with torch.no_grad():
            noise = torch.randn(
                next_observations.shape[0], self.action_dim, device=next_observations.device
            )
            next_action = self.ddpm_sampler(next_observations, noise)  # already clamped
            q_next = self.target_critic(next_observations, next_action).min(dim=0).values
            target_q = rewards + self.discount * (1.0 - dones) * q_next

        actions_clamped = torch.clamp(actions, -1.0, 1.0)
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
        
    @torch.compiler.disable
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        QSM actor loss:

            L(pi) = E[ ||-eps_phi(s, a_t, t) - eta *grad_a Q(s, a_t) ||^2 ]
                  + alpha * E[ || z - eps_phi(s, a_t, t) ||^2 ]

        .with t ~ Unif({0,...,T-1}) and a_t = sqrt(alpha_hat_t) * a + sqrt(1 - alpha_hat_t) * z
        """
        B = actions.shape[0]
        T = self.flow_steps
        # TODO(student): Implement actor update
        
        # TODO(student): Update actor
        t_idx = torch.randint(0, T, (B,), device=actions.device)
        alpha_hat_t = self.alpha_hats[t_idx].unsqueeze(-1)            # (B, 1)
        sqrt_ah = torch.sqrt(alpha_hat_t)
        sqrt_one_minus_ah = torch.sqrt(1.0 - alpha_hat_t)
        t_norm = ((t_idx.float() + 1.0) / T).unsqueeze(-1)            # (B, 1), matches sampler

        z = torch.randn_like(actions)
        a_t = sqrt_ah * actions + sqrt_one_minus_ah * z               # forward diffusion

        a_t_grad = a_t.detach().clone().requires_grad_(True)
        q_for_grad = self.critic(observations, a_t_grad).min(dim=0).values  # (B,)
        grad_q = torch.autograd.grad(q_for_grad.sum(), a_t_grad, create_graph=False)[0]
        grad_q = grad_q.detach()

        eps_pred = self.actor(observations, a_t, t_norm)

        qsm_loss = self.loss_fn(-eps_pred, self.inv_temp * grad_q)
        bc_loss = self.loss_fn(eps_pred, z)
        loss = qsm_loss + self.alpha * bc_loss

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "loss": loss,
            "qsm_loss": qsm_loss,
            "bc_loss": bc_loss,
            "grad_q_norm": grad_q.norm(dim=-1).mean(),
            "eps_norm": eps_pred.detach().norm(dim=-1).mean(),
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
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                p_targ.mul_(1.0 - self.target_update_rate)
                p_targ.add_(self.target_update_rate * p)