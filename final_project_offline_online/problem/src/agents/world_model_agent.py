from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

import infrastructure.pytorch_util as ptu
from agents.fql_agent import FQLAgent
from agents.ifql_agent import IFQLAgent
from agents.sacbc_agent import SACBCAgent


LOWER_AGENT_CLASSES = {
    "fql": FQLAgent,
    "ifql": IFQLAgent,
    "sacbc": SACBCAgent,
}


class EnsembleWorldModel(nn.Module):
    """Predicts state deltas and rewards with an ensemble for epistemic uncertainty."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        hidden_size: int,
        num_layers: int,
        ensemble_size: int,
    ):
        super().__init__()
        self.observation_dim = int(np.prod(observation_shape))
        self.action_dim = action_dim
        self.output_dim = self.observation_dim + 1

        self.models = nn.ModuleList(
            [
                ptu.build_mlp(
                    input_size=self.observation_dim + action_dim,
                    output_size=self.output_dim,
                    n_layers=num_layers,
                    size=hidden_size,
                )
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([observations, torch.clamp(actions, -1.0, 1.0)], dim=-1)
        return torch.stack([model(inputs) for model in self.models], dim=0)


class WorldModelAgent(nn.Module):
    """
    Inspired by Wombet, uses existing lower level agents to do the decision making.
    """

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        lower_agent: str,
        lower_agent_kwargs: dict,
        make_world_model_optimizer,
        world_model_hidden_size: int = 512,
        world_model_num_layers: int = 3,
        ensemble_size: int = 5,
        model_updates_per_step: int = 1,
        world_model_warmup_steps: int = 0,
        synthetic_start_uncertainty_threshold: float = 0.1,
        synthetic_rollout_horizon: int = 1,
        synthetic_ratio: float = 1.0,
        initial_synthetic_ratio: float = 0.1,
        synthetic_ratio_ramp_rate: float = 0.01,
        synthetic_uncertainty_weight_coef: float = 1.0,
        uncertainty_penalty: float = 1.0,
        uncertainty_threshold: float = 1.0,
        return_threshold: float = -float("inf"),
        synthetic_discount: float = 0.99,
        utd_ratio: int = 1,
        td_error_threshold: float = 1.0,
        td_error_ema_decay: float = 0.99,
    ):
        super().__init__()

        if lower_agent not in LOWER_AGENT_CLASSES:
            raise ValueError(
                f"Unsupported lower_agent={lower_agent!r}. "
                f"Expected one of {sorted(LOWER_AGENT_CLASSES)}."
            )

        self.observation_shape = tuple(observation_shape)
        self.observation_dim = int(np.prod(observation_shape))
        self.action_dim = action_dim
        self.lower_agent_name = lower_agent
        self.lower_agent = LOWER_AGENT_CLASSES[lower_agent](
            observation_shape,
            action_dim,
            **lower_agent_kwargs,
        )

        self.world_model = EnsembleWorldModel(
            observation_shape=observation_shape,
            action_dim=action_dim,
            hidden_size=world_model_hidden_size,
            num_layers=world_model_num_layers,
            ensemble_size=ensemble_size,
        )
        self.world_model_optimizer = make_world_model_optimizer(self.world_model.parameters())
        self.model_loss_fn = nn.MSELoss()

        self.chunk_size = getattr(self.lower_agent, 'chunk_size', 1)
        self.model_updates_per_step = model_updates_per_step
        self.world_model_warmup_steps = world_model_warmup_steps
        self.synthetic_start_uncertainty_threshold = synthetic_start_uncertainty_threshold
        self.synthetic_rollout_horizon = synthetic_rollout_horizon
        self.max_synthetic_ratio = synthetic_ratio
        self.initial_synthetic_ratio = initial_synthetic_ratio
        self.synthetic_ratio_ramp_rate = synthetic_ratio_ramp_rate
        self.current_synthetic_ratio = min(initial_synthetic_ratio, synthetic_ratio)
        self.synthetic_uncertainty_weight_coef = synthetic_uncertainty_weight_coef
        self.uncertainty_penalty = uncertainty_penalty
        self.uncertainty_threshold = uncertainty_threshold
        self.return_threshold = return_threshold
        self.synthetic_discount = synthetic_discount
        self.utd_ratio = utd_ratio
        self.td_error_threshold = td_error_threshold
        self.td_error_ema_decay = td_error_ema_decay
        self.td_error_ema = None

    def set_alpha(self, alpha: float) -> None:
        if hasattr(self.lower_agent, "set_alpha"):
            self.lower_agent.set_alpha(alpha)

    def set_synthetic_threshold(self, threshold: float) -> None:
        self.synthetic_start_uncertainty_threshold = threshold

    def get_action(self, observation: np.ndarray):
        return self.lower_agent.get_action(observation)

    def update_world_model(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
    ) -> dict:
        target_delta = next_observations - observations
        target = torch.cat([target_delta, rewards.unsqueeze(-1)], dim=-1)

        last_loss = None
        for _ in range(self.model_updates_per_step):
            predictions = self.world_model(observations, actions)
            target_expanded = target.unsqueeze(0).expand_as(predictions)
            loss = self.model_loss_fn(predictions, target_expanded)

            self.world_model_optimizer.zero_grad()
            loss.backward()
            self.world_model_optimizer.step()
            last_loss = loss

        return {"loss": last_loss.detach()}

    @torch.no_grad()
    def _sample_lower_actions(self, observations: torch.Tensor) -> torch.Tensor:
        if self.lower_agent_name == "ifql":
            return self.lower_agent.sample_actions(observations)

        if self.lower_agent_name == "fql":
            # sample_actions returns full chunk (B, K*ac_dim); world model needs single actions
            chunk = self.lower_agent.sample_actions(observations)
            return chunk[:, :self.action_dim]

        action_dist = self.lower_agent.actor(observations)
        return torch.clamp(action_dist.sample(), -1.0, 1.0)

    @torch.no_grad()
    def generate_synthetic_batch(
        self,
        observations: torch.Tensor,
    ) -> dict:
        if self.chunk_size > 1:
            return self._generate_synthetic_batch_chunked(observations)
        return self._generate_synthetic_batch_single(observations)

    @torch.no_grad()
    def _generate_synthetic_batch_single(
        self,
        observations: torch.Tensor,
    ) -> dict:
        obs = observations
        rollout_observations = []
        rollout_actions = []
        rollout_rewards = []
        rollout_next_observations = []
        rollout_uncertainties = []
        cumulative_returns = torch.zeros(observations.shape[0], device=observations.device)

        for horizon_idx in range(self.synthetic_rollout_horizon):
            actions = self._sample_lower_actions(obs)
            predictions = self.world_model(obs, actions)
            mean_prediction = predictions.mean(dim=0)

            uncertainty = predictions.var(dim=0, unbiased=False).sum(dim=-1).sqrt()
            delta = mean_prediction[:, : self.observation_dim]
            reward = mean_prediction[:, self.observation_dim] - self.uncertainty_penalty * uncertainty
            next_obs = obs + delta

            rollout_observations.append(obs)
            rollout_actions.append(actions)
            rollout_rewards.append(reward)
            rollout_next_observations.append(next_obs)
            rollout_uncertainties.append(uncertainty)
            cumulative_returns = cumulative_returns + (self.synthetic_discount**horizon_idx) * reward
            obs = next_obs

        uncertainties = torch.stack(rollout_uncertainties, dim=0)
        candidate_mean_uncertainty = uncertainties.mean()
        accepted = (
            (uncertainties.mean(dim=0) <= self.uncertainty_threshold)
            & (cumulative_returns >= self.return_threshold)
        )

        if accepted.sum() == 0:
            return {
                "candidate_mean_uncertainty": candidate_mean_uncertainty,
                "acceptance_rate": accepted.float().mean(),
            }

        synthetic = {
            "observations": torch.stack(rollout_observations, dim=0)[:, accepted].reshape(-1, self.observation_dim),
            "actions": torch.stack(rollout_actions, dim=0)[:, accepted].reshape(-1, self.action_dim),
            "rewards": torch.stack(rollout_rewards, dim=0)[:, accepted].reshape(-1),
            "next_observations": torch.stack(rollout_next_observations, dim=0)[:, accepted].reshape(-1, self.observation_dim),
            "uncertainties": uncertainties[:, accepted].reshape(-1),
            "dones": torch.zeros(
                int(accepted.sum().item()) * self.synthetic_rollout_horizon,
                device=observations.device,
                dtype=observations.dtype,
            ),
            "candidate_mean_uncertainty": candidate_mean_uncertainty,
            "accepted_mean_uncertainty": uncertainties[:, accepted].mean(),
            "acceptance_rate": accepted.float().mean(),
        }
        return synthetic

    @torch.no_grad()
    def _generate_synthetic_batch_chunked(
        self,
        observations: torch.Tensor,
    ) -> dict:
        """Produce K-step chunk transitions so actions match the chunked real batch format."""
        B = observations.shape[0]
        chunk_action_dim = self.action_dim * self.chunk_size
        device = observations.device

        # Each outer horizon step is one full K-step chunk transition
        rollout_observations = []
        rollout_chunk_actions = []
        rollout_rewards = []
        rollout_next_observations = []
        rollout_uncertainties = []
        cumulative_returns = torch.zeros(B, device=device)

        obs = observations
        for horizon_idx in range(self.synthetic_rollout_horizon):
            # Roll out chunk_size 1-step predictions, accumulate
            chunk_start_obs = obs
            chunk_actions = []
            chunk_reward = torch.zeros(B, device=device)
            chunk_uncertainty = torch.zeros(B, device=device)
            alive = torch.ones(B, device=device)

            for k in range(self.chunk_size):
                action = self._sample_lower_actions(obs)
                predictions = self.world_model(obs, action)
                mean_prediction = predictions.mean(dim=0)

                uncertainty = predictions.var(dim=0, unbiased=False).sum(dim=-1).sqrt()
                delta = mean_prediction[:, : self.observation_dim]
                step_reward = mean_prediction[:, self.observation_dim] - self.uncertainty_penalty * uncertainty
                next_obs = obs + delta

                chunk_actions.append(action)
                chunk_reward = chunk_reward + alive * (self.synthetic_discount ** k) * step_reward
                chunk_uncertainty = chunk_uncertainty + uncertainty
                obs = next_obs

            chunk_uncertainty = chunk_uncertainty / self.chunk_size
            # actions: (B, chunk_size * action_dim)
            chunk_action_tensor = torch.cat(chunk_actions, dim=-1)

            rollout_observations.append(chunk_start_obs)
            rollout_chunk_actions.append(chunk_action_tensor)
            rollout_rewards.append(chunk_reward)
            rollout_next_observations.append(obs)
            rollout_uncertainties.append(chunk_uncertainty)
            cumulative_returns = cumulative_returns + (self.synthetic_discount ** (horizon_idx * self.chunk_size)) * chunk_reward

        uncertainties = torch.stack(rollout_uncertainties, dim=0)  # (H, B)
        candidate_mean_uncertainty = uncertainties.mean()
        accepted = (
            (uncertainties.mean(dim=0) <= self.uncertainty_threshold)
            & (cumulative_returns >= self.return_threshold)
        )

        if accepted.sum() == 0:
            return {
                "candidate_mean_uncertainty": candidate_mean_uncertainty,
                "acceptance_rate": accepted.float().mean(),
            }

        n_accepted = int(accepted.sum().item())
        synthetic = {
            "observations": torch.stack(rollout_observations, dim=0)[:, accepted].reshape(-1, self.observation_dim),
            "actions": torch.stack(rollout_chunk_actions, dim=0)[:, accepted].reshape(-1, chunk_action_dim),
            "rewards": torch.stack(rollout_rewards, dim=0)[:, accepted].reshape(-1),
            "next_observations": torch.stack(rollout_next_observations, dim=0)[:, accepted].reshape(-1, self.observation_dim),
            "uncertainties": uncertainties[:, accepted].reshape(-1),
            "dones": torch.zeros(
                n_accepted * self.synthetic_rollout_horizon,
                device=device,
                dtype=observations.dtype,
            ),
            "candidate_mean_uncertainty": candidate_mean_uncertainty,
            "accepted_mean_uncertainty": uncertainties[:, accepted].mean(),
            "acceptance_rate": accepted.float().mean(),
        }
        return synthetic

    def _mix_real_and_synthetic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        synthetic: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
        if not synthetic or self.current_synthetic_ratio <= 0:
            sample_weights = torch.ones_like(rewards)
            return observations, actions, rewards, next_observations, dones, sample_weights, 0, 0.0

        max_synthetic = int(observations.shape[0] * self.current_synthetic_ratio)
        num_synthetic = min(max_synthetic, synthetic["observations"].shape[0])
        if num_synthetic <= 0:
            sample_weights = torch.ones_like(rewards)
            return observations, actions, rewards, next_observations, dones, sample_weights, 0, 0.0

        idx = torch.randperm(synthetic["observations"].shape[0], device=observations.device)[:num_synthetic]
        mixed_observations = torch.cat([observations, synthetic["observations"][idx]], dim=0)
        mixed_actions = torch.cat([actions, synthetic["actions"][idx]], dim=0)
        mixed_rewards = torch.cat([rewards, synthetic["rewards"][idx]], dim=0)
        mixed_next_observations = torch.cat(
            [next_observations, synthetic["next_observations"][idx]], dim=0
        )
        mixed_dones = torch.cat([dones, synthetic["dones"][idx]], dim=0)
        synthetic_weights = torch.clamp(
            torch.exp(-self.synthetic_uncertainty_weight_coef * synthetic["uncertainties"][idx]),
            min=0.05,
            max=0.5,
        )
        sample_weights = torch.cat([torch.ones_like(rewards), synthetic_weights], dim=0)
        return (
            mixed_observations,
            mixed_actions,
            mixed_rewards,
            mixed_next_observations,
            mixed_dones,
            sample_weights,
            num_synthetic,
            synthetic_weights.mean().item(),
        )

    def _update_synthetic_ratio(self, uncertainty_gate_open: bool, td_error: float = None) -> None:
        if not uncertainty_gate_open:
            self.current_synthetic_ratio = self.initial_synthetic_ratio
            return

        if td_error is not None:
            if self.td_error_ema is None:
                self.td_error_ema = td_error
            else:
                self.td_error_ema = (
                    self.td_error_ema_decay * self.td_error_ema
                    + (1.0 - self.td_error_ema_decay) * td_error
                )

        # High TD error means critic is destabilized — pull back toward real data
        if self.td_error_ema is not None and self.td_error_ema > self.td_error_threshold:
            self.current_synthetic_ratio = max(
                self.current_synthetic_ratio - self.synthetic_ratio_ramp_rate,
                self.initial_synthetic_ratio,
            )
        else:
            self.current_synthetic_ratio = min(
                self.current_synthetic_ratio + self.synthetic_ratio_ramp_rate,
                self.max_synthetic_ratio,
            )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
        real_batch: Optional[dict] = None,
    ):
        # When action chunking is active, the main batch contains K-step aggregated transitions
        # (next_obs is K steps ahead, reward is cumulative). Use real_batch of true single-step
        # transitions for world model training so the dynamics model sees consistent 1-step targets.
        if real_batch is not None:
            wm_obs = real_batch["observations"]
            wm_actions = real_batch["actions"]
            wm_rewards = real_batch["rewards"]
            wm_next_obs = real_batch["next_observations"]
        else:
            single_actions = actions[:, :self.action_dim] if actions.shape[-1] > self.action_dim else actions
            wm_obs = observations
            wm_actions = single_actions
            wm_rewards = rewards
            wm_next_obs = next_observations
        model_metrics = self.update_world_model(wm_obs, wm_actions, wm_rewards, wm_next_obs)

        warmup_gate_open = step >= self.world_model_warmup_steps

        can_use_synthetic = (
            warmup_gate_open
            and self.synthetic_start_uncertainty_threshold > 0
            and self.current_synthetic_ratio > 0
        )
        if can_use_synthetic:
            synthetic_candidate = self.generate_synthetic_batch(observations)
            uncertainty_gate_open = (
                bool(synthetic_candidate)
                and synthetic_candidate["candidate_mean_uncertainty"].item()
                <= self.synthetic_start_uncertainty_threshold
            )
            synthetic = synthetic_candidate if uncertainty_gate_open and "observations" in synthetic_candidate else {}
        else:
            synthetic_candidate = {}
            uncertainty_gate_open = False
            synthetic = {}

        synthetic_ratio_used = self.current_synthetic_ratio

        # UTD loop: mix once per iteration (randperm gives different synthetic subsets),
        # update lower agent utd_ratio times, average the metrics.
        all_lower_metrics = []
        num_synthetic = 0
        mean_synthetic_weight = 0.0
        for _ in range(self.utd_ratio):
            (
                mixed_observations,
                mixed_actions,
                mixed_rewards,
                mixed_next_observations,
                mixed_dones,
                sample_weights,
                num_synthetic,
                mean_synthetic_weight,
            ) = self._mix_real_and_synthetic(
                observations,
                actions,
                rewards,
                next_observations,
                dones,
                synthetic,
            )
            all_lower_metrics.append(self.lower_agent.update(
                mixed_observations,
                mixed_actions,
                mixed_rewards,
                mixed_next_observations,
                mixed_dones,
                step,
                sample_weights,
            ))

        lower_metrics = {
            k: sum(m[k] for m in all_lower_metrics) / len(all_lower_metrics)
            for k in all_lower_metrics[-1]
        }

        td_error = lower_metrics.get("critic/q_loss", None)
        self._update_synthetic_ratio(uncertainty_gate_open, td_error)

        metrics = {
            **{f"lower/{k}": v for k, v in lower_metrics.items()},
            "world_model/loss": model_metrics["loss"].item(),
            "world_model/synthetic_batch_size": num_synthetic,
            "world_model/mean_synthetic_weight": mean_synthetic_weight,
            "world_model/synthetic_uncertainty_weight_coef": self.synthetic_uncertainty_weight_coef,
            "world_model/using_synthetic": float(num_synthetic > 0),
            "world_model/synthetic_ratio_used": synthetic_ratio_used,
            "world_model/current_synthetic_ratio": self.current_synthetic_ratio,
            "world_model/max_synthetic_ratio": self.max_synthetic_ratio,
            "world_model/warmup_gate_open": float(warmup_gate_open),
            "world_model/uncertainty_gate_open": float(uncertainty_gate_open),
            "world_model/warmup_steps_remaining": max(self.world_model_warmup_steps - step, 0),
            "world_model/synthetic_start_uncertainty_threshold": self.synthetic_start_uncertainty_threshold,
            "world_model/td_error_ema": self.td_error_ema if self.td_error_ema is not None else 0.0,
            "world_model/acceptance_rate": (
                synthetic_candidate["acceptance_rate"].item() if synthetic_candidate else 0.0
            ),
            "world_model/mean_uncertainty": (
                synthetic_candidate["candidate_mean_uncertainty"].item() if synthetic_candidate else 0.0
            ),
            "world_model/accepted_mean_uncertainty": (
                synthetic_candidate["accepted_mean_uncertainty"].item()
                if synthetic_candidate and "accepted_mean_uncertainty" in synthetic_candidate
                else 0.0
            ),
        }
        return metrics
