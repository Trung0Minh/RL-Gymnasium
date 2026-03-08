import gymnasium as gym
import torch
import numpy as np
import argparse
import os
from agent import DQNAgent
from config import DQNConfig

def test(cfg: DQNConfig, n_episodes=5):
    """Test the trained DQN agent."""
    # Create environment with human rendering
    env = gym.make(cfg.env_id, render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize agent and load saved weights
    agent = DQNAgent(state_size=state_size, action_size=action_size, config=cfg, device=device)
    
    checkpoint_path = f"{cfg.checkpoint_dir}/acrobot_dqn.pt"
    try:
        agent.load(checkpoint_path)
        print(f"Successfully loaded model from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_path} not found. Running with random weights.")

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
    # Dynamically build parser from DQNConfig fields
    for key, value in DQNConfig().__dict__.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value) if value is not None else str, default=value)
    parser.add_argument("--test_episodes", type=int, default=5)
    
    args = parser.parse_args()
    config = DQNConfig(**{k: v for k, v in vars(args).items() if k in DQNConfig().__dict__})
    
    test(config, n_episodes=args.test_episodes)
