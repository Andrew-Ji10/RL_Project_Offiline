import argparse
import os
from datetime import datetime

import numpy as np
import torch
import tqdm
import wandb

import configs
from agents import agents
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log
from infrastructure.replay_buffer import ReplayBuffer


def run_offline_training_loop(config: dict, train_logger, eval_logger, args: argparse.Namespace, start_step: int = 0):
    """
    Run offline training loop
    """
    # TODO(student): Implement offline training loop
    
    #inspired by HW 5
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

    if "alpha_offline" in config and hasattr(agent, "set_alpha"):
        agent.set_alpha(config["alpha_offline"])
    if "synthetic_threshold_offline" in config and hasattr(agent, "set_synthetic_threshold"):
        agent.set_synthetic_threshold(config["synthetic_threshold_offline"])

    ep_len = env.spec.max_episode_steps or env.max_episode_steps

    best_eval_success = -float("inf")
    best_agent_path = os.path.join(args.save_dir, "agent_offline_best.pt")

    for step in tqdm.trange(config["offline_training_steps"] + 1, dynamic_ncols=True):
        # Train with offline RL
        chunk_size = config.get("action_chunk_size", 1)
        chunk_discount = config.get("discount", 0.99)
        batch = dataset.sample_chunk(config["batch_size"], chunk_size, chunk_discount)

        batch = {
            k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
        }

        real_batch = None
        if chunk_size > 1 and hasattr(agent, "update_world_model"):
            wm_np = dataset.sample(config["batch_size"])
            real_batch = {k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in wm_np.items()}

        update_kwargs = {"real_batch": real_batch} if real_batch is not None else {}
        metrics = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            step,
            **update_kwargs,
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
            eval_success = float(np.mean(successes))
            best_eval_success = max(best_eval_success, eval_success)

            eval_logger.log(
                {
                    "eval/success_rate": eval_success,
                    "eval/best_success_rate": best_eval_success,
                },
                step=step,
            )
            if eval_success >= best_eval_success:
                torch.save(agent.state_dict(), best_agent_path)

    
    agent_pt_path = dump_log(agent, train_logger, eval_logger, config, args.save_dir)
    torch.save(agent.state_dict(), os.path.join(args.save_dir, "agent_offline.pt"))
    return agent_pt_path

def run_online_training_loop(config: dict, train_logger, eval_logger, args: argparse.Namespace, agent_path: str, start_step: int = 0):
    """
    Run online training loop
    """
    # TODO(student): Implement online training loop


    #inspired by HW 3
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # make the gym environment
    env, _ = config["make_env_and_dataset"]()
    eval_env, _ = config["make_env_and_dataset"]()

    #load agent (new)
    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]
    
    agent_cls = agents[config["agent"]]
    agent = agent_cls(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    if agent_path is not None:
        agent.load_state_dict(torch.load(agent_path))
    if "alpha_online" in config and hasattr(agent, "set_alpha"):
        agent.set_alpha(config["alpha_online"])
    if "synthetic_threshold_online" in config and hasattr(agent, "set_synthetic_threshold"):
        agent.set_synthetic_threshold(config["synthetic_threshold_online"])
    if config.get("disable_world_model_online", False) and hasattr(agent, "set_world_model_enabled"):
        agent.set_world_model_enabled(False)
        print("[online] World model disabled: skipping WM updates and synthetic data online.")
    # load agent (end)


    # render_env = config["make_env"](eval=True, render=True)

    ep_len = env.spec.max_episode_steps or env.max_episode_steps

    # discrete = isinstance(env.action_space, gym.spaces.Discrete)
    # assert (
    #     not discrete
    # ), "SAC only supports continuous action spaces."

    # ob_shape = env.observation_space.shape
    # ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    # if "model" in dir(env):
    #     fps = 1 / env.model.opt.timestep
    # elif "render_fps" in env.env.metadata:
    #     fps = env.env.metadata["render_fps"]
    # else:
    #     fps = 10

    # initialize agent
    # agent = SoftActorCritic(
    #     ob_shape,
    #     ac_dim,
    #     **config["agent_kwargs"],
    # )
    #done

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])
    _online_chunk_size = config.get("action_chunk_size", 1)
    _wm_online_active = (
        hasattr(agent, "update_world_model")
        and not config.get("disable_world_model_online", False)
    )
    single_step_buffer = (
        ReplayBuffer(config["replay_buffer_capacity"])
        if _online_chunk_size > 1 and _wm_online_active
        else None
    )

    #4.1- offline data pre-filling the replay buffer
    n_offline = int(config.get("offline_data", 0))
    if n_offline > 0:
        _, offline_dataset = config["make_env_and_dataset"]()
        n = min(n_offline, offline_dataset.size)
        idx = np.random.choice(offline_dataset.size, size=n, replace=False)
        for i in idx:
            r = offline_dataset.rewards[i]
            d = offline_dataset.dones[i]
            replay_buffer.insert(
                observation=offline_dataset.observations[i],
                action=offline_dataset.actions[i],
                reward=float(r) if np.ndim(r) == 0 else r,
                next_observation=offline_dataset.next_observations[i],
                done=bool(d),
            )
        print(f"[s2_offline] Seeded replay buffer with {n} offline transitions.")

    observation, _ = env.reset()

    best_eval_success = -float("inf")
    best_agent_path = os.path.join(args.save_dir, "agent_best.pt")

    for step in tqdm.trange(start_step, start_step + config['online_training_steps'] + 1, dynamic_ncols=True):
        
        #TODO - Personal, could bring this back if wanted 
        # if step < config["random_steps"]:
        #     action = env.action_space.sample()
        # else:
        #     # TODO(Section 3.1): Select an action
        #     action = agent.get_action(observation)
        #     # ENDTODO
        chunk_size = config.get("action_chunk_size", 1)
        chunk_discount = config.get("discount", 0.99)
        ac_dim = env.action_space.shape[0]

        if chunk_size > 1 and hasattr(agent, "get_action_chunk"):
            action_chunk = agent.get_action_chunk(observation)
            cum_reward = 0.0
            done = False
            truncated = False
            final_obs = observation
            current_obs = observation
            for k in range(chunk_size):
                ac = action_chunk[k * ac_dim:(k + 1) * ac_dim]
                next_observation, reward, terminated, truncated, info = env.step(ac)
                done = terminated or truncated
                cum_reward += (chunk_discount ** k) * reward
                if single_step_buffer is not None:
                    single_step_buffer.insert(
                        observation=current_obs,
                        action=ac,
                        reward=reward,
                        next_observation=next_observation,
                        done=done and not truncated,
                    )
                current_obs = next_observation
                final_obs = next_observation
                if done:
                    break
            replay_buffer.insert(
                observation=observation,
                action=action_chunk,
                reward=cum_reward,
                next_observation=final_obs,
                done=done and not truncated,
            )
            next_observation = final_obs
        else:
            action = agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done and not truncated,
            )

        if done:
            episode_info = info.get("episode", {})
            episode_return = episode_info.get("r", episode_info.get("return"))
            episode_len = episode_info.get("l", episode_info.get("length"))
            train_logger.log({
                "Train_EpisodeReturn": episode_return,
                "Train_EpisodeLen": episode_len,
            }, step)
            observation, _ = env.reset()
        else:
            observation = next_observation

        #4.2- WSRL for N warmup online steps
        #NOT gradient-updated during wsrl steps.
        online_step = step - start_step
        in_warmup = online_step < int(config.get("wsrl_steps", 0))
        #buffer is done filling up?
        buffer_warm = step >= config["training_starts"] + start_step

        # Train the agent
        # if step >= config["training_starts"]  + start_step:
        #     # TODO(Section 3.1): Sample a batch of config["batch_size"] transitions from the replay buffer
        # change for warmstart compatibility
        if buffer_warm and not in_warmup:
            update_infos = []
            _chunk_size = config.get("action_chunk_size", 1)
            _chunk_discount = config.get("discount", 0.99)
            for utd_idx in range(int(config.get("update_to_data_ratio", 1))):
                batch = replay_buffer.sample_chunk(config['batch_size'], _chunk_size, _chunk_discount)
                batch = {
                    k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
                }

                real_batch = None
                if single_step_buffer is not None and single_step_buffer.size >= config['batch_size']:
                    wm_np = single_step_buffer.sample(config['batch_size'])
                    real_batch = {k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in wm_np.items()}

                online_update_kwargs = {"real_batch": real_batch} if real_batch is not None else {}
                update_infos.append(agent.update(
                    observations = batch["observations"],
                    actions = batch["actions"],
                    rewards = batch['rewards'],
                    next_observations = batch['next_observations'],
                    dones = batch['dones'],
                    step = step,
                    **online_update_kwargs))
            update_info = {
                k: float(np.mean([info[k] for info in update_infos]))
                for k in update_infos[-1]
            }
            # ENDTODO

            # Logging
            if step % args.log_interval == 0:
                if step % args.eval_interval != 0 and buffer_warm and not in_warmup:
                    train_logger.log(update_info, step)

        # Run evaluation
        if step % args.eval_interval == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]
            successes = [t["episode_statistics"]["s"] for t in trajectories]
            eval_success = float(np.mean(successes))
            best_eval_success = max(best_eval_success, eval_success)

            eval_metrics = {
                "Eval_AverageReturn": float(np.mean(returns)),
                "Eval_StdReturn": float(np.std(returns)),
                "Eval_MaxReturn": float(np.max(returns)),
                "Eval_MinReturn": float(np.min(returns)),
                "Eval_AverageEpLen": float(np.mean(ep_lens)),
                "eval/success_rate": eval_success,
                "eval/best_success_rate": best_eval_success,
            }
            if eval_success >= best_eval_success:
                torch.save(agent.state_dict(), best_agent_path)

            # Merge training metrics if available (skipped during WSRL warmup,
            # since `update_info` is only defined when the agent has been updated).
            # if step >= start_step + config["training_starts"]:
            if buffer_warm and not in_warmup:
                eval_metrics.update(update_info)
            eval_logger.log(eval_metrics, step)
            # if args.num_render_trajectories > 0:
            #     video_trajectories = utils.sample_n_trajectories(
            #         render_env,
            #         agent,
            #         args.num_render_trajectories,
            #         ep_len,
            #         render=True,
            #     )

            #     eval_logger.log_paths_as_videos(
            #         video_trajectories,
            #         step,
            #         fps=fps,
            #         max_videos_to_save=args.num_render_trajectories,
            #         video_title="eval_rollouts",
            #     )

            # Save checkpoint periodically
            dump_log(agent, train_logger, eval_logger, config, args.save_dir)




    agent_pt_path = dump_log(agent, train_logger, eval_logger, config, args.save_dir)
    torch.save(agent.state_dict(), os.path.join(args.save_dir, "agent_online.pt"))
    return agent_pt_path



def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default='sacbc')
    parser.add_argument("--env_name", type=str, default='cube-single-play-singletask-task1-v0')
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_group", type=str, default='Debug')
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", default=0)
    parser.add_argument("--offline_training_steps", type=int, default=500000)  # Should be 500k to pass the autograder
    parser.add_argument("--online_training_steps", type=int, default=100000)  # Should be 100k to pass the autograder
    parser.add_argument("--replay_buffer_capacity", type=int, default=1000000)
    parser.add_argument("--log_interval", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", type=int, default=25)  # Should be greater than or equal to 20 to pass autograder
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--target_update_rate", type=float, default=None)
    

    # Online retention of offline data
    # TODO(student): If desired, add arguments for online retention of offline data
    parser.add_argument(
        "--offline_data", type=int, default=0,
        help="Number of offline transitions to pre-fill into the online replay buffer.",
    )

    # WSRL
    # TODO (student): If desired, add arguments for WSRL
    parser.add_argument(
        "--wsrl_steps", type=int, default=0,
        help="Number of warm-up env steps at the start of online training during which the agent is NOT updated.",
    )
    

    # IFQL
    parser.add_argument("--expectile", type=float, default=None)

    # FQL / QSM
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--alpha_offline", type=float, default=None,
                        help="Alpha for offline phase only. Overrides --alpha for offline.")
    parser.add_argument("--alpha_online", type=float, default=None,
                        help="Alpha for online phase only. Overrides --alpha for online.")
    parser.add_argument("--lower_agent", type=str, default=None, choices=["fql", "ifql", "sacbc"])
    parser.add_argument("--n_critics", type=int, default=None)
    parser.add_argument("--q_pessimism_rho", type=float, default=None)
    parser.add_argument("--num_action_samples", type=int, default=None)
    parser.add_argument("--compile_fql", action="store_true")
    parser.add_argument("--update_to_data_ratio", type=int, default=None)
    parser.add_argument("--world_model_warmup_steps", type=int, default=None)
    parser.add_argument("--synthetic_start_uncertainty_threshold", type=float, default=None)
    parser.add_argument("--synthetic_threshold_offline", type=float, default=None,
                        help="Uncertainty threshold for world model synthetic data during offline training.")
    parser.add_argument("--synthetic_threshold_online", type=float, default=None,
                        help="Uncertainty threshold for world model synthetic data during online training.")
    parser.add_argument("--initial_synthetic_ratio", type=float, default=None)
    parser.add_argument("--synthetic_ratio", type=float, default=None)
    parser.add_argument("--synthetic_ratio_ramp_rate", type=float, default=None)
    parser.add_argument("--synthetic_uncertainty_weight_coef", type=float, default=None)
    parser.add_argument("--uncertainty_penalty", type=float, default=None)
    parser.add_argument("--uncertainty_threshold", type=float, default=None)
    parser.add_argument("--utd_ratio", type=int, default=None)
    parser.add_argument("--td_error_threshold", type=float, default=None)
    parser.add_argument("--action_chunk_size", type=int, default=None)

    # QSM
    parser.add_argument("--inv_temp", type=float, default=None)

    # DSRL
    parser.add_argument("--noise_scale", type=float, default=None)

    # Pretrained checkpoint: skip offline training and load this agent for online
    parser.add_argument(
        "--pretrained_agent_path", type=str, default=None,
        help="Path to a saved agent checkpoint (e.g. agent_offline.pt). "
             "When set, offline training is skipped and this checkpoint is loaded for online training. "
             "Use with --offline_training_steps=0 so logs start at step 0 (splice_online.py will shift them).",
    )
    parser.add_argument(
        "--disable_world_model_online", action="store_true",
        help="If set, skip world model training and synthetic data generation during online "
             "training. Only the lower agent is updated on real replay data. Faster online step "
             "time when you don't intend to use the world model online.",
    )

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
    config['offline_training_steps'] = args.offline_training_steps
    config['online_training_steps'] = args.online_training_steps
    config['log_interval'] = args.log_interval
    config['eval_interval'] = args.eval_interval
    config['num_eval_trajectories'] = args.num_eval_trajectories
    config['replay_buffer_capacity'] = args.replay_buffer_capacity
    
    # TODO(student): If necessary, add additional config values
    config["training_starts"] = 10000 # HW 3 sac_config.py
    config["offline_data"] = args.offline_data #4.1 offline data
    config["wsrl_steps"] = args.wsrl_steps #4.2 WSRL
    config["update_to_data_ratio"] = args.update_to_data_ratio or config.get("update_to_data_ratio", 1)
    config["disable_world_model_online"] = args.disable_world_model_online

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
    if args.update_to_data_ratio is not None:
        exp_name = f"{exp_name}_utd{args.update_to_data_ratio}"
    #add expname debug to differentiate between runs
    if args.offline_data > 0:
        exp_name = f"{exp_name}_od{args.offline_data}"
    if args.wsrl_steps > 0:
        exp_name = f"{exp_name}_w{args.wsrl_steps}"

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
    if args.alpha_offline is not None:
        config["alpha_offline"] = args.alpha_offline
        exp_name = f"{exp_name}_aoff{args.alpha_offline}"
    if args.alpha_online is not None:
        config["alpha_online"] = args.alpha_online
        exp_name = f"{exp_name}_aon{args.alpha_online}"
    if args.synthetic_threshold_offline is not None:
        config["synthetic_threshold_offline"] = args.synthetic_threshold_offline
        exp_name = f"{exp_name}_stoff{args.synthetic_threshold_offline}"
    if args.synthetic_threshold_online is not None:
        config["synthetic_threshold_online"] = args.synthetic_threshold_online
        exp_name = f"{exp_name}_ston{args.synthetic_threshold_online}"
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
    if args.utd_ratio is not None and "utd_ratio" in config["agent_kwargs"]:
        config["agent_kwargs"]["utd_ratio"] = args.utd_ratio
        config["utd_ratio"] = args.utd_ratio
        exp_name = f"{exp_name}_utd{args.utd_ratio}"
    if args.td_error_threshold is not None and "td_error_threshold" in config["agent_kwargs"]:
        config["agent_kwargs"]["td_error_threshold"] = args.td_error_threshold
        config["td_error_threshold"] = args.td_error_threshold
        exp_name = f"{exp_name}_tdet{args.td_error_threshold}"
    if args.action_chunk_size is not None:
        config["action_chunk_size"] = args.action_chunk_size
        # push into agent_kwargs (direct FQL) or lower_agent_kwargs (world model+FQL)
        if "action_chunk_size" in config["agent_kwargs"]:
            config["agent_kwargs"]["action_chunk_size"] = args.action_chunk_size
        elif "action_chunk_size" in config["agent_kwargs"].get("lower_agent_kwargs", {}):
            config["agent_kwargs"]["lower_agent_kwargs"]["action_chunk_size"] = args.action_chunk_size
        exp_name = f"{exp_name}_ck{args.action_chunk_size}"
    if args.inv_temp is not None:
        config['agent_kwargs']['inv_temp'] = args.inv_temp
        exp_name = f"{exp_name}_i{args.inv_temp}"
    if args.noise_scale is not None:
        config['agent_kwargs']['noise_scale'] = args.noise_scale
        exp_name = f"{exp_name}_n{args.noise_scale}"
    if args.disable_world_model_online:
        exp_name = f"{exp_name}_nowmon"
    if args.online_training_steps > 0:
        exp_name = f"{exp_name}_online"
    if args.offline_training_steps > 0:
        exp_name = f"{exp_name}_offline"

    setup_wandb(project='cs185_default_project', name=exp_name, group=args.run_group, config=config)
    args.save_dir = os.path.join(logdir_prefix, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    train_logger = Logger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = Logger(os.path.join(args.save_dir, 'eval.csv'))

    start_step = 0
    agent_path_offline = None
    if args.pretrained_agent_path is not None:
        # Skip offline training entirely; use the provided checkpoint for online training.
        # Set start_step=0 so logs are relative to online phase start (use splice_online.py
        # to prepend the offline portion from the original full run).
        print(f"[pretrained] Skipping offline training. Loading checkpoint: {args.pretrained_agent_path}")
        agent_path_offline = args.pretrained_agent_path
        start_step = 0
    elif args.offline_training_steps > 0:
        print(f"Running offline training loop with {args.offline_training_steps} steps")
        agent_path_offline = run_offline_training_loop(config, train_logger, eval_logger, args, start_step=0)
        start_step = args.offline_training_steps

    if args.online_training_steps > 0:
        print(f"Running online training loop with {args.online_training_steps} steps")
        agent_path_online = run_online_training_loop(config, train_logger, eval_logger, args, agent_path_offline, start_step=start_step)
        
    wandb.finish()


if __name__ == "__main__":
    args = setup_arguments()
    if args.njobs is not None and len(args.job_specs) > 0:
        # Run n jobs in parallel
        from scripts.run_njobs import main_njobs
        main_njobs(job_specs=args.job_specs, njobs=args.njobs)
    else:
        # Run a single job
        main(args)
