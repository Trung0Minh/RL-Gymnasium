import gymnasium as gym
import torch
import numpy as np
from agent import TD3Agent
from config import TD3Config
import argparse
import os

def test(cfg: TD3Config, n_episodes=5):
    """Test the trained TD3 agent."""
    # Create environment with human rendering
    env = gym.make(cfg.env_id, render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize agent
    agent = TD3Agent(state_dim=state_dim, action_dim=action_dim, config=cfg, device=device)
    
    checkpoint_path = f"{cfg.checkpoint_dir}/bipedal_walker_td3.pt"
    stats_path = f"{cfg.checkpoint_dir}/bipedal_walker_obs_rms.pt"
    
    try:
        agent.load(checkpoint_path)
        print(f"Successfully loaded model from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_path} not found. Running with random weights.")

    obs_mean, obs_var = None, None
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location=device, weights_only=False)
        obs_mean = stats["mean"]
        obs_var = stats["var"]
        print(f"Loaded normalization stats from {stats_path}")

    def normalize_state(state):
        if obs_mean is not None:
            return np.clip((state - obs_mean) / np.sqrt(obs_var + 1e-8), -10, 10)
        return state

    for i in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        done = False
        while not done:
            state_norm = normalize_state(state)
            action = agent.select_action(state_norm)
            state, reward, terminated, truncated, _ = env.step(action[0])
            score += reward
            done = terminated or truncated
            
        print(f"Test Episode {i} | Reward: {score:.2f}")
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dynamically build parser from TD3Config fields
    for key, value in TD3Config().__dict__.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value) if value is not None else str, default=value)
    parser.add_argument("--test_episodes", type=int, default=5)
    
    args = parser.parse_args()
    config = TD3Config(**{k: v for k, v in vars(args).items() if k in TD3Config().__dict__})
    
    test(config, n_episodes=args.test_episodes)
