import os
import argparse
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from wrappers import make_env
from model import SACActor, SACQNetwork, SACDiscreteActor, SACDiscreteQNetwork
from agent import SACAgent
from replay import ReplayBuffer
from config import SACConfig

def train(cfg: SACConfig, mode: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    is_continuous = (mode == "continuous")
    env_fns = [make_env(cfg.env_id, continuous=is_continuous, seed=cfg.seed + i, max_episode_steps=cfg.max_t) for i in range(cfg.num_envs)]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    
    if is_continuous:
        actor = SACActor(n_stack=4, n_actions=3).to(device)
        q1 = SACQNetwork(n_stack=4, n_actions=3).to(device)
        q2 = SACQNetwork(n_stack=4, n_actions=3).to(device)
        q1_target = SACQNetwork(n_stack=4, n_actions=3).to(device)
        q2_target = SACQNetwork(n_stack=4, n_actions=3).to(device)
    else:
        actor = SACDiscreteActor(n_stack=4, n_actions=5).to(device)
        q1 = SACDiscreteQNetwork(n_stack=4, n_actions=5).to(device)
        q2 = SACDiscreteQNetwork(n_stack=4, n_actions=5).to(device)
        q1_target = SACDiscreteQNetwork(n_stack=4, n_actions=5).to(device)
        q2_target = SACDiscreteQNetwork(n_stack=4, n_actions=5).to(device)

    agent = SACAgent(actor, q1, q2, q1_target, q2_target, config=cfg, device=device, is_discrete=not is_continuous)
    buffer = ReplayBuffer(capacity=cfg.buffer_size, device=device)

    checkpoint_path = f"{cfg.checkpoint_dir}/car_racing_sac_{mode}.pt"
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    if cfg.resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        agent.load(checkpoint_path)

    scores = []
    scores_window = deque(maxlen=100)
    states, _ = envs.reset()
    episode_rewards = np.zeros(cfg.num_envs)
    total_steps = 0
    finished_episodes = 0
    
    print(f"Starting training in {mode} mode...")
    
    while finished_episodes < cfg.max_episodes:
        if total_steps < cfg.start_steps and not cfg.resume:
            actions = envs.action_space.sample()
        else:
            actions = agent.select_action(states)
            if is_continuous:
                # CarRacing continuous actions are [steer, gas, break]
                # steer in [-1, 1], gas in [0, 1], break in [0, 1]
                # actor output is tanh in [-1, 1]
                actions[:, 1] = (actions[:, 1] + 1) / 2
                actions[:, 2] = (actions[:, 2] + 1) / 2
        
        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
        dones = terminateds | truncateds
        
        for i in range(cfg.num_envs):
            real_next_state = next_states[i]
            if dones[i] and "final_observation" in infos:
                real_next_state = infos["final_observation"][i]
            
            buffer.push(states[i], actions[i], rewards[i], real_next_state, terminateds[i])
            episode_rewards[i] += rewards[i]
            
            if dones[i]:
                finished_episodes += 1
                scores_window.append(episode_rewards[i])
                scores.append(episode_rewards[i])
                
                avg_rew = np.mean(scores_window)
                log_str = f"Episode {finished_episodes} | Avg Ep Reward: {avg_rew:.2f}"
                
                if finished_episodes % 50 == 0:
                    print(f"\r{log_str}")
                    agent.save(checkpoint_path)
                else:
                    print(f"\r{log_str}", end="", flush=True)
                
                episode_rewards[i] = 0

        states = next_states
        total_steps += cfg.num_envs

        if total_steps >= cfg.start_steps or cfg.resume:
            for _ in range(cfg.updates_per_step * cfg.num_envs):
                agent.update(buffer, cfg.batch_size)

    envs.close()
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["continuous", "discrete"], default="continuous")
    # Dynamically build parser from SACConfig fields
    for key, value in SACConfig().__dict__.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value) if value is not None else str, default=value)
    
    args = parser.parse_args()
    mode = args.mode
    # Remove mode from args to match SACConfig fields
    args_dict = vars(args)
    del args_dict['mode']
    config = SACConfig(**args_dict)
    
    scores = train(config, mode)

    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('rewards.png')
    print(f"\nTraining plot saved as rewards.png")
