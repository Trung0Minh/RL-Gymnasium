import os
import argparse
import torch
import gymnasium as gym
import numpy as np
from wrappers import make_env
from model import SACActor, SACQNetwork, SACDiscreteActor, SACDiscreteQNetwork
from sac import SACAgent, ReplayBuffer
import config

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["continuous", "discrete"], default="continuous")
    parser.add_argument("--episodes", type=int, default=config.EPISODES)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--gamma", type=float, default=config.GAMMA)
    parser.add_argument("--tau", type=float, default=config.TAU)
    parser.add_argument("--alpha", type=float, default=config.ALPHA)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num_envs", type=int, default=config.NUM_ENVS)
    parser.add_argument("--capacity", type=int, default=config.CAPACITY)
    parser.add_argument("--start_steps", type=int, default=config.START_STEPS)
    parser.add_argument("--updates_per_step", type=int, default=config.UPDATES_PER_STEP)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_num_threads(1)
    print(f"Training with SAC in {args.mode} mode on {device} using {args.num_envs} envs")

    is_continuous = (args.mode == "continuous")
    env_fns = [make_env("CarRacing-v3", continuous=is_continuous, seed=i) for i in range(args.num_envs)]
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

    agent = SACAgent(actor, q1, q2, q1_target, q2_target, lr=args.lr, gamma=args.gamma, tau=args.tau, alpha=args.alpha, device=device, is_discrete=not is_continuous)
    buffer = ReplayBuffer(capacity=args.capacity)

    states, _ = envs.reset()
    episode_rewards = np.zeros(args.num_envs)
    best_reward = -float('inf')
    total_steps = 0
    finished_episodes = 0
    
    while finished_episodes < args.episodes:
        if total_steps < args.start_steps:
            actions = envs.action_space.sample()
        else:
            states_t = torch.tensor(states, dtype=torch.float32).to(device)
            with torch.no_grad():
                if is_continuous:
                    actions_t, _, _ = actor.sample(states_t)
                    actions = actions_t.cpu().numpy()
                    # Remap Gas/Brake for all envs
                    actions[:, 1] = (actions[:, 1] + 1) / 2
                    actions[:, 2] = (actions[:, 2] + 1) / 2
                else:
                    actions_t, _, _ = actor.sample(states_t)
                    actions = actions_t.cpu().numpy()
        
        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
        dones = terminateds | truncateds
        
        for i in range(args.num_envs):
            # When an env is done, the next_state in the vector is actually the start of the next episode
            # We need the actual final observation from the info dict if gymnasium supports it
            real_next_state = next_states[i]
            if dones[i] and "final_observation" in infos:
                real_next_state = infos["final_observation"][i]
            
            buffer.push(states[i], actions[i], rewards[i], real_next_state, terminateds[i])
            episode_rewards[i] += rewards[i]
            
            if dones[i]:
                finished_episodes += 1
                if episode_rewards[i] > best_reward:
                    best_reward = episode_rewards[i]
                    checkpoint_path = f"best_sac_{args.mode}.pth"
                    torch.save(actor.state_dict(), checkpoint_path)
                
                print(f"\rEpisode {finished_episodes} | Reward: {episode_rewards[i]:.2f}", end='')
                if finished_episodes % 100 == 0:
                    print(f"\rEpisode {finished_episodes} | Reward: {episode_rewards[i]:.2f}")
                episode_rewards[i] = 0

        states = next_states
        total_steps += args.num_envs

        if total_steps >= args.start_steps:
            for _ in range(args.updates_per_step * args.num_envs):
                agent.update(buffer, args.batch_size)

    envs.close()

if __name__ == "__main__":
    train()
