import gymnasium as gym
import torch
import numpy as np
from agent import Agent
import argparse

def test(n_episodes=5, checkpoint_path='checkpoint.pth'):
    """Test the trained DQN agent."""
    # Create environment with human rendering
    env = gym.make('LunarLander-v3', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agent and load saved weights
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
    
    for i in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        done = False
        while not done:
            # Select action greedily (eps=0)
            action = agent.act(state, eps=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
            
        print(f"Episode {i}: Score = {score:.2f}")
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    
    try:
        test(n_episodes=args.episodes)
    except FileNotFoundError:
        print("Error: 'checkpoint.pth' not found. Please train the agent first using 'python train.py'.")