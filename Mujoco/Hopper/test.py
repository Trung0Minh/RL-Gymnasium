import gymnasium as gym
import torch
import numpy as np
import pickle
from actor_critic import Actor
from utils import RunningMeanStd

def test(env_name='Hopper-v5', model_path='ppo_actor.pth', stats_path='obs_rms.pkl', num_episodes=5):
    # Initialize environment with rendering
    env = gym.make(env_name, render_mode='human')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Load the trained actor
    actor = Actor(obs_dim, act_dim)
    try:
        actor.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please run train.py first.")
        return

    # Load normalization stats
    try:
        with open(stats_path, 'rb') as f:
            obs_rms = pickle.load(f)
        print(f"Successfully loaded stats from {stats_path}")
    except FileNotFoundError:
        print(f"Warning: {stats_path} not found. Using identity normalization.")
        obs_rms = None

    def normalize_obs(obs):
        if obs_rms is None:
            return obs
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)

    actor.eval()

    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0
        ep_len = 0

        while not done:
            n_obs = normalize_obs(obs)
            obs_tensor = torch.as_tensor(n_obs, dtype=torch.float32)
            with torch.no_grad():
                dist = actor(obs_tensor)
                act = dist.mean  # Deterministic action
            
            obs, rew, terminated, truncated, _ = env.step(act.numpy())
            done = terminated or truncated
            ep_ret += rew
            ep_len += 1
            
        print(f"Episode {i+1}: Return = {ep_ret:.2f}, Length = {ep_len}")

    env.close()

if __name__ == '__main__':
    test()
