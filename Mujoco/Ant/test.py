import gymnasium as gym
import torch
import numpy as np
from ppo import PPO

class ObservationNormalizer:
    def __init__(self, state_dim):
        self.mean = np.zeros(state_dim)
        self.std = np.ones(state_dim)
        self.n = 1e-4

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']
        self.n = checkpoint['n']

    def __call__(self, state):
        return (state - self.mean) / (self.std + 1e-8)

def test():
    env_name = "Ant-v5"
    checkpoint_path = f"ppo_{env_name}_final.pth"
    normalizer_path = f"ppo_{env_name}_normalizer.pth"
    max_ep_len = 1000
    num_episodes = 5  # Test over multiple episodes
    
    env = gym.make(env_name, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    normalizer = ObservationNormalizer(state_dim)
    try:
        normalizer.load(normalizer_path)
        print("Loaded normalizer parameters.")
    except FileNotFoundError:
        print("Normalizer parameters not found, using default (none).")

    ppo_agent = PPO(state_dim, action_dim, 0, 0, 0, 0, 0)
    ppo_agent.load(checkpoint_path)

    all_episode_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        state = normalizer(state)
        total_reward = 0
        
        for t in range(max_ep_len):
            # Use deterministic=True for evaluation
            action, _, _, _ = ppo_agent.select_action(state[None, :], deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action[0])
            state = normalizer(state)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        all_episode_rewards.append(total_reward)
        print(f"Episode {ep+1} Reward: {total_reward:.2f}")
            
    print("-" * 25)
    print(f"Average Reward: {np.mean(all_episode_rewards):.2f} (+/- {np.std(all_episode_rewards):.2f})")
    env.close()

if __name__ == '__main__':
    test()
