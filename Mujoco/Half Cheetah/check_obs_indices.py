import gymnasium as gym
import numpy as np

def check_obs():
    env = gym.make("HalfCheetah-v5")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    # In HalfCheetah, obs[0] is typically torso z (height)
    # and obs[1] is torso angle (pitch).
    # Let's see how they change with a random action.
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Torso Height (obs[0])={obs[0]:.4f}, Torso Angle (obs[1])={obs[1]:.4f}")

if __name__ == "__main__":
    check_obs()
