from typing import Optional

import torch

from .fql_config import fql_config
from .ifql_config import ifql_config
from .sacbc_config import sacbc_config


LOWER_CONFIGS = {
    "fql": fql_config,
    "ifql": ifql_config,
    "sacbc": sacbc_config,
}


def world_model_config(
    env_name: str,
    exp_name: Optional[str] = None,
    lower_agent: str = "sacbc",
    hidden_size: int = 512,
    num_layers: int = 4,
    learning_rate: float = 3e-4,
    discount: float = 0.99,
    target_update_rate: float = 0.005,
    alpha: float = 1.0,
    flow_steps: int = 10,
    expectile: float = 0.9,
    num_samples: int = 32,
    n_critics: int = 2,
    q_pessimism_rho: Optional[float] = None,
    num_action_samples: int = 1,
    world_model_hidden_size: int = 512,
    world_model_num_layers: int = 3,
    world_model_learning_rate: float = 3e-4,
    ensemble_size: int = 5,
    model_updates_per_step: int = 1,
    world_model_warmup_steps: int = 0,
    synthetic_start_uncertainty_threshold: float = 0.1,
    synthetic_rollout_horizon: int = 1,
    synthetic_ratio: float = 0.5,
    initial_synthetic_ratio: float = 0.1,
    synthetic_ratio_ramp_rate: float = 0.001,
    synthetic_uncertainty_weight_coef: float = 1.0,
    uncertainty_penalty: float = 1.0,
    uncertainty_threshold: float = 1.0,
    return_threshold: float = -float("inf"),
    synthetic_discount: float = 0.99,
    total_steps: int = 1000000,
    batch_size: int = 256,
    **kwargs,
):
    if lower_agent not in LOWER_CONFIGS:
        raise ValueError(
            f"Unsupported lower_agent={lower_agent!r}. "
            f"Expected one of {sorted(LOWER_CONFIGS)}."
        )

    lower_kwargs = {
        "exp_name": exp_name,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "discount": discount,
        "target_update_rate": target_update_rate,
        "total_steps": total_steps,
        "batch_size": batch_size,
    }
    if lower_agent in ("fql", "ifql"):
        lower_kwargs["flow_steps"] = flow_steps
    if lower_agent in ("fql", "sacbc"):
        lower_kwargs["alpha"] = alpha
    if lower_agent == "fql":
        lower_kwargs["n_critics"] = n_critics
        lower_kwargs["q_pessimism_rho"] = q_pessimism_rho
        lower_kwargs["num_action_samples"] = num_action_samples
    if lower_agent == "ifql":
        lower_kwargs["expectile"] = expectile
        lower_kwargs["num_samples"] = num_samples

    lower_config = LOWER_CONFIGS[lower_agent](env_name, **lower_kwargs)

    def make_world_model_optimizer(params) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=world_model_learning_rate)

    log_string = f"{exp_name or 'world_model'}_{lower_agent}_{env_name}"

    config = {
        "agent_kwargs": {
            "lower_agent": lower_agent,
            "lower_agent_kwargs": lower_config["agent_kwargs"],
            "make_world_model_optimizer": make_world_model_optimizer,
            "world_model_hidden_size": world_model_hidden_size,
            "world_model_num_layers": world_model_num_layers,
            "ensemble_size": ensemble_size,
            "model_updates_per_step": model_updates_per_step,
            "world_model_warmup_steps": world_model_warmup_steps,
            "synthetic_start_uncertainty_threshold": synthetic_start_uncertainty_threshold,
            "synthetic_rollout_horizon": synthetic_rollout_horizon,
            "synthetic_ratio": synthetic_ratio,
            "initial_synthetic_ratio": initial_synthetic_ratio,
            "synthetic_ratio_ramp_rate": synthetic_ratio_ramp_rate,
            "synthetic_uncertainty_weight_coef": synthetic_uncertainty_weight_coef,
            "uncertainty_penalty": uncertainty_penalty,
            "uncertainty_threshold": uncertainty_threshold,
            "return_threshold": return_threshold,
            "synthetic_discount": synthetic_discount,
        },
        "agent": "world_model",
        "lower_agent": lower_agent,
        "log_name": log_string,
        "make_env_and_dataset": lower_config["make_env_and_dataset"],
        "total_steps": total_steps,
        "env_name": env_name,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "n_critics": n_critics,
        "q_pessimism_rho": q_pessimism_rho,
        "num_action_samples": num_action_samples,
        "world_model_hidden_size": world_model_hidden_size,
        "world_model_num_layers": world_model_num_layers,
        "world_model_learning_rate": world_model_learning_rate,
        "ensemble_size": ensemble_size,
        "world_model_warmup_steps": world_model_warmup_steps,
        "synthetic_start_uncertainty_threshold": synthetic_start_uncertainty_threshold,
        "synthetic_rollout_horizon": synthetic_rollout_horizon,
        "synthetic_ratio": synthetic_ratio,
        "initial_synthetic_ratio": initial_synthetic_ratio,
        "synthetic_ratio_ramp_rate": synthetic_ratio_ramp_rate,
        "synthetic_uncertainty_weight_coef": synthetic_uncertainty_weight_coef,
        "uncertainty_penalty": uncertainty_penalty,
        "uncertainty_threshold": uncertainty_threshold,
        "return_threshold": return_threshold,
        **kwargs,
    }

    return config
