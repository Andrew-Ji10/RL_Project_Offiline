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

    ep_len = env.spec.max_episode_steps or env.max_episode_steps

    for step in tqdm.trange(config["offline_training_steps"] + 1, dynamic_ncols=True):
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

    
    return dump_log(agent, train_logger, eval_logger, config, args.save_dir)

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

    agent.load_state_dict(torch.load(agent_path))
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

    observation, _ = env.reset()

    for step in tqdm.trange(start_step, start_step + config['online_training_steps'], dynamic_ncols=True):
        
        #TODO - Personal, could bring this back if wanted 
        # if step < config["random_steps"]:
        #     action = env.action_space.sample()
        # else:
        #     # TODO(Section 3.1): Select an action
        #     action = agent.get_action(observation)
        #     # ENDTODO
        action = agent.get_action(observation)

        # Step the environment and add the data to the replay buffer
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

        # Train the agent
        if step >= config["training_starts"]  + start_step:
            # TODO(Section 3.1): Sample a batch of config["batch_size"] transitions from the replay buffer
            batch = replay_buffer.sample(config['batch_size'])
            batch = {
                k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
            }


            update_info = agent.update(
                observations = batch["observations"], 
                actions = batch["actions"], 
                rewards = batch['rewards'], 
                next_observations = batch['next_observations'], 
                dones = batch['dones'], 
                step = step)
            # ENDTODO

            # Logging
            if step % args.log_interval == 0:
                if step % args.eval_interval != 0:
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

            eval_metrics = {
                "Eval_AverageReturn": np.mean(returns),
                "Eval_StdReturn": np.std(returns),
                "Eval_MaxReturn": np.max(returns),
                "Eval_MinReturn": np.min(returns),
                "Eval_AverageEpLen": np.mean(ep_lens),
                "eval/success_rate": float(np.mean(successes)),
            }

            # Merge training metrics if available
            if step >= start_step + config["training_starts"]:
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




    return dump_log(agent, train_logger, eval_logger, config, args.save_dir)



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
    parser.add_argument("--log_interval", type=int, default=20000)
    parser.add_argument("--eval_interval", type=int, default=20000)
    parser.add_argument("--num_eval_trajectories", type=int, default=25)  # Should be greater than or equal to 20 to pass autograder
    

    # Online retention of offline data
    # TODO(student): If desired, add arguments for online retention of offline data
    
    # WSRL
    # TODO (student): If desired, add arguments for WSRL
    

    # IFQL
    parser.add_argument("--expectile", type=float, default=None)

    # FQL / QSM
    parser.add_argument("--alpha", type=float, default=None)

    # QSM
    parser.add_argument("--inv_temp", type=float, default=None)

    # DSRL
    parser.add_argument("--noise_scale", type=float, default=None)

    # For njobs mode (optional)
    parser.add_argument("--njobs", type=int, default=None)
    parser.add_argument("job_specs", nargs="*")

    args = parser.parse_args(args=args)

    return args


def main(args):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder

    config = configs.configs[args.base_config](args.env_name)

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

    exp_name = f"sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['log_name']}"

    # Override agent hyperparameters if specified
    if args.expectile is not None:
        config['agent_kwargs']['expectile'] = args.expectile
        exp_name = f"{exp_name}_e{args.expectile}"
    if args.alpha is not None:
        config['agent_kwargs']['alpha'] = args.alpha
        exp_name = f"{exp_name}_a{args.alpha}"
    if args.inv_temp is not None:
        config['agent_kwargs']['inv_temp'] = args.inv_temp
        exp_name = f"{exp_name}_i{args.inv_temp}"
    if args.noise_scale is not None:
        config['agent_kwargs']['noise_scale'] = args.noise_scale
        exp_name = f"{exp_name}_n{args.noise_scale}"
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
    if args.offline_training_steps > 0:
        print(f"Running offline training loop with {args.offline_training_steps} steps")
        # TODO(student): Implement offline training loop
        # Hint: You might consider passing the agent's path to the online training loop
        agent_path_offline = run_offline_training_loop(config, train_logger, eval_logger, args, start_step=0)
        start_step = args.offline_training_steps
        
    
    if args.online_training_steps > 0:
        print(f"Running online training loop with {args.online_training_steps} steps")
        # TODO(student): Implement online training loop
        agent_path_online = run_online_training_loop(config, train_logger, eval_logger, args, agent_path_offline, start_step=start_step)


if __name__ == "__main__":
    args = setup_arguments()
    if args.njobs is not None and len(args.job_specs) > 0:
        # Run n jobs in parallel
        from scripts.run_njobs import main_njobs
        main_njobs(job_specs=args.job_specs, njobs=args.njobs)
    else:
        # Run a single job
        main(args)
