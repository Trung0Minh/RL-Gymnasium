import gymnasium as gym
import torch
import numpy as np
import pickle
from models import ActorCritic

def make_env(env_id, render_mode=None):
    """
    Creates the environment. Note: We do NOT use NormalizeObservation here
    because we will apply the saved statistics manually for precision.
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    return env

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_id = "HalfCheetah-v5"
    checkpoint_path = "half_cheetah_ppo.pt"
    rms_path = "half_cheetah_obs_rms.pkl"
    
    # Initialize environment
    env = make_env(env_id, render_mode="human")
    
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    
    # Load the model
    model = ActorCritic(num_inputs, num_actions).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"Successfully loaded model: {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: {checkpoint_path} not found.")
        return

    # Load Normalization Stats
    try:
        with open(rms_path, "rb") as f:
            rms_stats = pickle.load(f)
        mean = rms_stats["mean"]
        var = rms_stats["var"]
        epsilon = 1e-8
        print(f"Successfully loaded normalization stats: {rms_path}")
    except FileNotFoundError:
        print(f"Warning: {rms_path} not found. Testing without normalization (may fail).")
        mean = np.zeros(num_inputs)
        var = np.ones(num_inputs)
        epsilon = 0

    num_episodes = 5
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episodic_reward = 0
        
        while not done:
            # Manually apply normalization: (obs - mean) / sqrt(var + eps)
            norm_obs = (obs - mean) / np.sqrt(var + epsilon)
            norm_obs = np.clip(norm_obs, -10, 10) # Match training clip
            
            obs_tensor = torch.Tensor(norm_obs).to(device).unsqueeze(0)
            
            with torch.no_grad():
                action_mean = model.actor_mean(obs_tensor)
                action = action_mean.cpu().numpy().flatten()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = np.logical_or(terminated, truncated)
            episodic_reward += reward
            
        print(f"Episode {episode + 1}: Total Reward = {episodic_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
