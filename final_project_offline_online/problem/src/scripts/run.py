import argparse
import os
from datetime import datetime

import numpy as np
import torch
import tqdm

import configs
from agents import agents
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log


def run_training_loop(config: dict, train_logger, eval_logger, args: argparse.Namespace):
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # Make the gymnasium environment
    env, dataset = config["make_env_and_dataset"]()

    example_batch = dataset.sample(1)
    agent_cls = agents[config["agent"]]
    agent = agent_cls(
        example_batch['observations'].shape[1:],
        example_batch['actions'].shape[-1],
        **config["agent_kwargs"],
    )

    ep_len = env.spec.max_episode_steps or env.max_episode_steps

    for step in tqdm.trange(config["training_steps"] + 1, dynamic_ncols=True):
        # Train with offline RL
        batch = dataset.sample(config["batch_size"])

        batch = {
            k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
        }

        metrics = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            step,
        )

        if step % args.log_interval == 0:
            train_logger.log(metrics, step=step)

        if step % args.eval_interval == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            
            successes = [t["episode_statistics"]["s"] for t in trajectories]

            eval_logger.log(
                {
                    "eval/success_rate": float(np.mean(successes)),
                },
                step=step,
            )

    dump_log(agent, train_logger, eval_logger, config, args.save_dir)


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default='sacbc')
    parser.add_argument("--env_name", type=str, default='cube-single-play-singletask-task1-v0')
    parser.add_argument("--exp_name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_group", type=str, default='Debug')
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", default=0)
    parser.add_argument("--training_steps", type=int, default=1000000)  # Should be less than or equal to 1M to pass autograder
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=100000)
    parser.add_argument("--num_eval_trajectories", type=int, default=25)  # Should be greater than or equal to 20 to pass autograder

    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--target_update_rate", type=float, default=None)
    parser.add_argument("--expectile", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--lower_agent", type=str, default=None, choices=["fql", "ifql", "sacbc"])
    parser.add_argument("--n_critics", type=int, default=None)
    parser.add_argument("--q_pessimism_rho", type=float, default=None)
    parser.add_argument("--num_action_samples", type=int, default=None)
    parser.add_argument("--compile_fql", action="store_true")
    parser.add_argument("--world_model_warmup_steps", type=int, default=None)
    parser.add_argument("--synthetic_start_uncertainty_threshold", type=float, default=None)
    parser.add_argument("--initial_synthetic_ratio", type=float, default=None)
    parser.add_argument("--synthetic_ratio", type=float, default=None)
    parser.add_argument("--synthetic_ratio_ramp_rate", type=float, default=None)
    parser.add_argument("--synthetic_uncertainty_weight_coef", type=float, default=None)
    parser.add_argument("--uncertainty_penalty", type=float, default=None)
    parser.add_argument("--uncertainty_threshold", type=float, default=None)

    # For njobs mode (optional)
    parser.add_argument("--njobs", type=int, default=None)
    parser.add_argument("job_specs", nargs="*")

    args = parser.parse_args(args=args)

    return args


def main(args):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder

    config_kwargs = {}
    if args.lower_agent is not None:
        config_kwargs["lower_agent"] = args.lower_agent
    if args.n_critics is not None:
        config_kwargs["n_critics"] = args.n_critics
    if args.q_pessimism_rho is not None:
        config_kwargs["q_pessimism_rho"] = args.q_pessimism_rho
    if args.num_action_samples is not None:
        config_kwargs["num_action_samples"] = args.num_action_samples
    if args.compile_fql:
        config_kwargs["compile_fql"] = True
    if args.learning_rate is not None:
        config_kwargs["learning_rate"] = args.learning_rate
    if args.target_update_rate is not None:
        config_kwargs["target_update_rate"] = args.target_update_rate
    config = configs.configs[args.base_config](args.env_name, **config_kwargs)

    # Set common config values from args for autograder
    config['seed'] = args.seed
    config['run_group'] = args.run_group
    config['training_steps'] = args.training_steps
    config['log_interval'] = args.log_interval
    config['eval_interval'] = args.eval_interval
    config['num_eval_trajectories'] = args.num_eval_trajectories

    exp_name = f"sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['log_name']}"
    if args.lower_agent is not None:
        exp_name = f"{exp_name}_lower{args.lower_agent}"
    if args.n_critics is not None:
        exp_name = f"{exp_name}_nc{args.n_critics}"
    if args.q_pessimism_rho is not None:
        exp_name = f"{exp_name}_qrho{args.q_pessimism_rho}"
    if args.num_action_samples is not None:
        exp_name = f"{exp_name}_nas{args.num_action_samples}"
    if args.compile_fql:
        exp_name = f"{exp_name}_compilefql"
    if args.learning_rate is not None:
        exp_name = f"{exp_name}_lr{args.learning_rate}"
    if args.target_update_rate is not None:
        exp_name = f"{exp_name}_tur{args.target_update_rate}"

    # Override agent hyperparameters if specified
    if args.expectile is not None:
        if "expectile" in config["agent_kwargs"]:
            config['agent_kwargs']['expectile'] = args.expectile
        elif "expectile" in config["agent_kwargs"].get("lower_agent_kwargs", {}):
            config["agent_kwargs"]["lower_agent_kwargs"]["expectile"] = args.expectile
        exp_name = f"{exp_name}_e{args.expectile}"
    if args.alpha is not None:
        if "alpha" in config["agent_kwargs"]:
            config['agent_kwargs']['alpha'] = args.alpha
        elif "alpha" in config["agent_kwargs"].get("lower_agent_kwargs", {}):
            config["agent_kwargs"]["lower_agent_kwargs"]["alpha"] = args.alpha
        exp_name = f"{exp_name}_a{args.alpha}"
    if args.world_model_warmup_steps is not None and "world_model_warmup_steps" in config["agent_kwargs"]:
        config["agent_kwargs"]["world_model_warmup_steps"] = args.world_model_warmup_steps
        config["world_model_warmup_steps"] = args.world_model_warmup_steps
        exp_name = f"{exp_name}_wmw{args.world_model_warmup_steps}"
    if (
        args.synthetic_start_uncertainty_threshold is not None
        and "synthetic_start_uncertainty_threshold" in config["agent_kwargs"]
    ):
        config["agent_kwargs"]["synthetic_start_uncertainty_threshold"] = args.synthetic_start_uncertainty_threshold
        config["synthetic_start_uncertainty_threshold"] = args.synthetic_start_uncertainty_threshold
        exp_name = f"{exp_name}_suth{args.synthetic_start_uncertainty_threshold}"
    if args.initial_synthetic_ratio is not None and "initial_synthetic_ratio" in config["agent_kwargs"]:
        config["agent_kwargs"]["initial_synthetic_ratio"] = args.initial_synthetic_ratio
        config["initial_synthetic_ratio"] = args.initial_synthetic_ratio
        exp_name = f"{exp_name}_isr{args.initial_synthetic_ratio}"
    if args.synthetic_ratio is not None and "synthetic_ratio" in config["agent_kwargs"]:
        config["agent_kwargs"]["synthetic_ratio"] = args.synthetic_ratio
        config["synthetic_ratio"] = args.synthetic_ratio
        exp_name = f"{exp_name}_sr{args.synthetic_ratio}"
    if args.synthetic_ratio_ramp_rate is not None and "synthetic_ratio_ramp_rate" in config["agent_kwargs"]:
        config["agent_kwargs"]["synthetic_ratio_ramp_rate"] = args.synthetic_ratio_ramp_rate
        config["synthetic_ratio_ramp_rate"] = args.synthetic_ratio_ramp_rate
        exp_name = f"{exp_name}_srr{args.synthetic_ratio_ramp_rate}"
    if (
        args.synthetic_uncertainty_weight_coef is not None
        and "synthetic_uncertainty_weight_coef" in config["agent_kwargs"]
    ):
        config["agent_kwargs"]["synthetic_uncertainty_weight_coef"] = args.synthetic_uncertainty_weight_coef
        config["synthetic_uncertainty_weight_coef"] = args.synthetic_uncertainty_weight_coef
        exp_name = f"{exp_name}_suw{args.synthetic_uncertainty_weight_coef}"
    if args.uncertainty_penalty is not None and "uncertainty_penalty" in config["agent_kwargs"]:
        config["agent_kwargs"]["uncertainty_penalty"] = args.uncertainty_penalty
        config["uncertainty_penalty"] = args.uncertainty_penalty
        exp_name = f"{exp_name}_up{args.uncertainty_penalty}"
    if args.uncertainty_threshold is not None and "uncertainty_threshold" in config["agent_kwargs"]:
        config["agent_kwargs"]["uncertainty_threshold"] = args.uncertainty_threshold
        config["uncertainty_threshold"] = args.uncertainty_threshold
        exp_name = f"{exp_name}_uth{args.uncertainty_threshold}"

    setup_wandb(project='cs285_offline_online_proj', name=exp_name, group=args.run_group, config=config)
    args.save_dir = os.path.join(logdir_prefix, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    train_logger = Logger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = Logger(os.path.join(args.save_dir, 'eval.csv'))

    run_training_loop(config, train_logger, eval_logger, args)


if __name__ == "__main__":
    args = setup_arguments()
    if args.njobs is not None and len(args.job_specs) > 0:
        # Run n jobs in parallel
        from scripts.run_njobs import main_njobs
        main_njobs(job_specs=args.job_specs, njobs=args.njobs)
    else:
        # Run a single job
        main(args)
