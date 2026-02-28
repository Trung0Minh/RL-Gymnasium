import gymnasium as gym
import torch
import numpy as np
from agent import TD3Agent
from replay_buffer import ReplayBuffer
import config
from collections import deque
from utils import RunningMeanStd
import pickle
import os

def make_env():
    def _init():
        env = gym.make('BipedalWalker-v3')
        return env
    return _init

def train(args):
    # Vectorized environments
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(args.num_envs)])
    
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    
    agent = TD3Agent(state_dim, action_dim)
    # Override agent config if needed or pass args to agent
    # For now, TD3Agent uses config.py directly, but we can update it or just use args here
    
    replay_buffer = ReplayBuffer(state_dim, action_dim, args.replay_size)
    state_rms = RunningMeanStd(shape=(state_dim,))
    
    max_timesteps = 10000000 # Total environment steps across all envs
    running_reward = deque(maxlen=100)
    total_timesteps = 0
    best_reward = -np.inf

    def normalize_state(state):
        state_rms.update(state)
        return np.clip((state - state_rms.mean) / np.sqrt(state_rms.var + 1e-8), -10, 10)

    states, infos = envs.reset()
    states = normalize_state(states)
    
    episode_rewards = np.zeros(args.num_envs)
    
    while total_timesteps < max_timesteps:
        total_timesteps += args.num_envs
        
        # Select action
        if total_timesteps < args.start_steps:
            actions = envs.action_space.sample()
        else:
            actions = agent.select_action(states)
            actions = (actions + np.random.normal(0, args.expl_noise, size=actions.shape)).clip(-1, 1)
        
        # Step environments
        next_states, rewards, dones, truncateds, infos = envs.step(actions)
        next_states_norm = normalize_state(next_states)
        
        # Store transitions in replay buffer
        for i in range(args.num_envs):
            # Check if this environment finished an episode
            actual_done = dones[i] or truncateds[i]
            
            # For BipedalWalker, we need to handle terminal state specifically if truncated
            # but for simplicity we use the simple add
            replay_buffer.add(states[i], actions[i], next_states_norm[i], rewards[i], dones[i])
            
            episode_rewards[i] += rewards[i]
            
            if actual_done:
                running_reward.append(episode_rewards[i])
                episode_rewards[i] = 0
                
        states = next_states_norm
        
        # Periodic Batch Updates
        if total_timesteps >= args.update_after and (total_timesteps // args.num_envs) % args.update_every == 0:
            agent.update(replay_buffer, iterations=args.update_iters)
        
        # Logging and Saving
        if len(running_reward) > 0 and (total_timesteps // args.num_envs) % 100 == 0:
            avg_rew = np.mean(running_reward)
            print(f"\rTotal Steps: {total_timesteps} | Avg Reward: {avg_rew:.2f}", end='')
            
            if avg_rew > best_reward and len(running_reward) >= 100:
                best_reward = avg_rew
                torch.save(agent.actor.state_dict(), "td3_actor_best.pth")
                with open("state_rms_best.pkl", "wb") as f:
                    pickle.dump(state_rms, f)
                
            if avg_rew > 300:
                print("\nSolved!")
                break

    envs.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=config.NUM_ENVS)
    parser.add_argument("--replay_size", type=int, default=config.REPLAY_SIZE)
    parser.add_argument("--start_steps", type=int, default=config.START_STEPS)
    parser.add_argument("--expl_noise", type=float, default=config.EXPL_NOISE)
    parser.add_argument("--update_after", type=int, default=config.UPDATE_AFTER)
    parser.add_argument("--update_every", type=int, default=config.UPDATE_EVERY)
    parser.add_argument("--update_iters", type=int, default=config.UPDATE_ITERS)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--gamma", type=float, default=config.GAMMA)
    parser.add_argument("--tau", type=float, default=config.TAU)
    args = parser.parse_args()

    # Update config for agent if necessary (agent imports config)
    config.LR = args.lr
    config.BATCH_SIZE = args.batch_size
    config.GAMMA = args.gamma
    config.TAU = args.tau

    train(args)
