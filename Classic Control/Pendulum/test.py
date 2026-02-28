import gymnasium as gym
import torch
import numpy as np
import argparse
import os
from ddpg_agent import Agent
import config

def test_ddpg(n_episodes=5, checkpoint_actor='model_weight/best_actor_checkpoint.pth', seed=2):
    """Evaluate the trained DDPG agent."""
    
    try:
        env = gym.make('Pendulum-v1', render_mode='human')
    except:
        print("Could not initialize human render mode. Running without visualization.")
        env = gym.make('Pendulum-v1')
        
    agent = Agent(state_size=3, action_size=1, random_seed=seed)
    
    # Load the trained weights
    try:
        agent.actor_local.load_state_dict(torch.load(checkpoint_actor, map_location=lambda storage, loc: storage))
        print(f"Successfully loaded weights from '{checkpoint_actor}'")
    except FileNotFoundError:
        print(f"Error: '{checkpoint_actor}' not found. Please run 'train.py' first.")
        return

    print(f"Starting evaluation for {n_episodes} episodes...")

    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        while True:
            # We don't add noise during testing to evaluate the pure policy
            action = agent.act(state, add_noise=False)
            
            # Rescale action from [-1, 1] to [-2, 2]
            rescaled_action = action * 2.0
            
            next_state, reward, terminated, truncated, _ = env.step(rescaled_action)
            done = terminated or truncated
            state = next_state
            score += reward
            
            if done:
                break
        print(f"Episode {i_episode}: Score: {score:.2f}")
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG Pendulum Testing')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to test')
    parser.add_argument('--checkpoint', type=str, default='model_weight/best_actor_checkpoint.pth', help='Path to actor checkpoint')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed')
    
    args = parser.parse_args()
    test_ddpg(n_episodes=args.episodes, checkpoint_actor=args.checkpoint, seed=args.seed)
