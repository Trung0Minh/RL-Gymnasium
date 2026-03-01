import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from agent import TD3Agent
from replay_buffer import ReplayBuffer
import config
from collections import deque
from utils import RunningMeanStd
import pickle
import os
import argparse

def make_env(max_episode_steps=None):
    def _init():
        env = gym.make('BipedalWalker-v3', max_episode_steps=max_episode_steps)
        return env
    return _init

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Vectorized environments
    envs = gym.vector.AsyncVectorEnv([make_env(args.max_t) for _ in range(args.num_envs)])
    
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    
    agent = TD3Agent(state_dim, action_dim)
    
    # Load checkpoint if requested
    if args.load:
        if os.path.isfile(args.load):
            print(f"Loading checkpoint from {args.load}")
            agent.actor.load_state_dict(torch.load(args.load, map_location=device))
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            if args.rms_load and os.path.isfile(args.rms_load):
                print(f"Loading RMS from {args.rms_load}")
                with open(args.rms_load, "rb") as f:
                    state_rms = pickle.load(f)
            else:
                state_rms = RunningMeanStd(shape=(state_dim,))
        else:
            print(f"Checkpoint {args.load} not found. Starting from scratch.")
            state_rms = RunningMeanStd(shape=(state_dim,))
    else:
        state_rms = RunningMeanStd(shape=(state_dim,))
    
    replay_buffer = ReplayBuffer(state_dim, action_dim, args.replay_size)
    
    scores = []
    scores_window = deque(maxlen=100)
    best_reward = -np.inf
    total_timesteps = 0
    finished_episodes = 0

    def normalize_state(state):
        state_rms.update(state)
        return np.clip((state - state_rms.mean) / np.sqrt(state_rms.var + 1e-8), -10, 10)

    states, infos = envs.reset()
    states = normalize_state(states)
    
    episode_rewards = np.zeros(args.num_envs)
    
    while finished_episodes < args.episodes:
        total_timesteps += args.num_envs
        
        # Select action
        if total_timesteps < args.start_steps and not args.load:
            actions = envs.action_space.sample()
        else:
            actions = agent.select_action(states)
            actions = (actions + np.random.normal(0, args.expl_noise, size=actions.shape)).clip(-1, 1)
        
        # Step environments
        next_states, rewards, dones, truncateds, infos = envs.step(actions)
        next_states_norm = normalize_state(next_states)
        
        # Store transitions in replay buffer
        for i in range(args.num_envs):
            actual_done = dones[i] or truncateds[i]
            replay_buffer.add(states[i], actions[i], next_states_norm[i], rewards[i], dones[i])
            episode_rewards[i] += rewards[i]
            
            if actual_done:
                finished_episodes += 1
                scores_window.append(episode_rewards[i])
                scores.append(episode_rewards[i])
                
                avg_rew = np.mean(scores_window)
                print(f"\rEpisode {finished_episodes} | Reward: {episode_rewards[i]:.2f} | Running Avg: {avg_rew:.2f}", end="")
                
                if finished_episodes % 100 == 0:
                    print(f"\rEpisode {finished_episodes} | Reward: {episode_rewards[i]:.2f} | Running Avg: {avg_rew:.2f}")

                if avg_rew > best_reward and len(scores_window) >= 100:
                    best_reward = avg_rew
                    os.makedirs('weights', exist_ok=True)
                    torch.save(agent.actor.state_dict(), "weights/td3_actor_best.pth")
                    with open("weights/state_rms_best.pkl", "wb") as f:
                        pickle.dump(state_rms, f)
                
                episode_rewards[i] = 0
                
        states = next_states_norm
        
        # Periodic Batch Updates
        if (total_timesteps >= args.update_after or args.load) and (total_timesteps // args.num_envs) % args.update_every == 0:
            agent.update(replay_buffer, iterations=args.update_iters)
        
        if len(scores_window) > 0 and np.mean(scores_window) > 300:
            print("\nSolved!")

    envs.close()

    # Plot results
    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores.png')
    print(f"\nTraining complete. Scores plot saved as 'scores.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=config.NUM_ENVS)
    parser.add_argument("--episodes", type=int, default=config.EPISODES)
    parser.add_argument("--max_t", type=int, default=config.MAX_T)
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
    parser.add_argument("--hidden_dim", type=int, default=config.HIDDEN_DIM)
    parser.add_argument("--load", type=str, default=None, help="Path to actor checkpoint to continue training")
    parser.add_argument("--rms_load", type=str, default=None, help="Path to state_rms checkpoint")
    args = parser.parse_args()

    # Update config for agent
    config.LR = args.lr
    config.BATCH_SIZE = args.batch_size
    config.GAMMA = args.gamma
    config.TAU = args.tau
    config.HIDDEN_DIM = args.hidden_dim

    train(args)
