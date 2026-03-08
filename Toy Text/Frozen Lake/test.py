import gymnasium as gym
import argparse
import os
from agent import QLearningAgent
from config import QConfig

def test(cfg: QConfig, n_episodes=5):
    """Test the trained Q-Learning agent."""
    # Create environment with human rendering
    env = gym.make(cfg.env_id, is_slippery=cfg.is_slippery, map_name=cfg.map_name, render_mode='human')
    agent = QLearningAgent(n_states=env.observation_space.n, n_actions=env.action_space.n, config=cfg)
    
    checkpoint_path = f"{cfg.checkpoint_dir}/frozen_lake_q.pkl"
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
            # Select action greedily (no training/exploration)
            action = agent.act(state, is_training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
            
        print(f"Test Episode {i} | Reward: {score:.2f}")
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dynamically build parser from QConfig fields
    for key, value in QConfig().__dict__.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value) if value is not None else str, default=value)
    parser.add_argument("--test_episodes", type=int, default=5)
    
    args = parser.parse_args()
    config = QConfig(**{k: v for k, v in vars(args).items() if k in QConfig().__dict__})
    
    test(config, n_episodes=args.test_episodes)
