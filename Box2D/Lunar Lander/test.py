import gymnasium as gym
import torch
import numpy as np
from agent import Agent
import argparse

def test(n_episodes=5, checkpoint_path='weights/checkpoint.pth'):
    """Test the trained DQN agent."""
    # Create environment with human rendering
    env = gym.make('LunarLander-v3', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agent and load saved weights
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Successfully loaded model from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_path} not found.")
        return

    agent.qnetwork_local.eval()
    
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
            
        print(f"Test Episode {i} | Reward: {score:.2f}")
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default="weights/checkpoint.pth")
    args = parser.parse_args()
    
    test(n_episodes=args.episodes, checkpoint_path=args.checkpoint)
