import gymnasium as gym
import torch
import numpy as np
import argparse
import os
from agent import DPGAgent
import config

def test_dpg(n_episodes=5, checkpoint_path='best_actor_checkpoint.pth', seed=1):
    """Evaluate the trained DPG agent."""
    
    try:
        env = gym.make('MountainCarContinuous-v0', render_mode='human')
    except:
        print("Could not initialize human render mode. Running without visualization.")
        env = gym.make('MountainCarContinuous-v0')
        
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    agent = DPGAgent(state_size=state_size, action_size=action_size, seed=seed)
    
    # Load the trained weights
    try:
        agent.actor_local.load_checkpoint(checkpoint_path)
        print(f"Successfully loaded weights from '{checkpoint_path}'")
    except FileNotFoundError:
        print(f"Error: '{checkpoint_path}' not found. Please run 'train.py' first.")
        return

    print(f"Starting evaluation for {n_episodes} episodes...")

    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        while True:
            # We don't add noise during testing to evaluate the pure policy
            action = agent.act(state, add_noise=False)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
            
            if done:
                break
        print(f"Episode {i_episode}: Score: {score:.2f}")
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DPG Mountain Car Continuous Testing')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to test')
    parser.add_argument('--checkpoint', type=str, default='best_actor_checkpoint.pth', help='Path to actor checkpoint')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed')
    
    args = parser.parse_args()
    test_dpg(n_episodes=args.episodes, checkpoint_path=args.checkpoint, seed=args.seed)
