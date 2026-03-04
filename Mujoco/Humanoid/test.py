import gymnasium as gym
import torch
import os
import numpy as np
from agent import PPOAgent

def test():
    env_id = "Humanoid-v5"
    checkpoint_path = "humanoid_final.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Run train.py first.")
        return

    # Must use same wrappers as training for consistent normalization
    env = gym.make(env_id, render_mode="human")
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(
        env, 
        lambda obs: np.clip(obs, -10, 10),
        observation_space=env.observation_space
    )
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(obs_dim, action_dim)
    agent.network.load_state_dict(torch.load(checkpoint_path))
    agent.network.eval()
    
    print(f"Loaded {checkpoint_path} ({env_id}), starting evaluation...")
    
    obs, _ = env.reset()
    for episode in range(1, 6):
        terminated = truncated = False
        ep_reward = 0
        while not (terminated or truncated):
            # Select action (we need to wrap obs in a batch of 1)
            action, _, _ = agent.select_action(np.expand_dims(obs, axis=0))
            obs, reward, terminated, truncated, _ = env.step(action[0])
            ep_reward += reward
        
        print(f"Episode {episode}, Reward: {ep_reward:.2f}")
        obs, _ = env.reset()

    env.close()

if __name__ == "__main__":
    test()
