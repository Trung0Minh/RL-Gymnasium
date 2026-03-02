import os
import torch
import gymnasium as gym
import numpy as np
from ppo import PPO
from memory import RolloutBuffer
import argparse

class ObservationNormalizer:
    def __init__(self, state_dim):
        self.mean = np.zeros(state_dim)
        self.std = np.ones(state_dim)
        self.n = 1e-4

    def __call__(self, state):
        if state.ndim == 1:
            self.n += 1
            old_mean = self.mean.copy()
            self.mean += (state - self.mean) / self.n
            self.std = np.sqrt(((self.n - 1) * self.std**2 + (state - old_mean) * (state - self.mean)) / self.n)
        else:
            batch_size = state.shape[0]
            self.n += batch_size
            old_mean = self.mean.copy()
            batch_mean = np.mean(state, axis=0)
            self.mean += (batch_mean - self.mean) * (batch_size / self.n)
            self.std = np.sqrt(((self.n - batch_size) * self.std**2 + np.sum((state - old_mean) * (state - self.mean), axis=0)) / self.n)
            
        return (state - self.mean) / (self.std + 1e-8)

def make_env(env_name):
    def thunk():
        return gym.make(env_name)
    return thunk

def train():
    ####### Hyperparameters #######
    env_name = "Ant-v5"
    num_envs = 8                
    max_training_timesteps = int(10e6) # Increased to 10M
    update_timestep = 4096      # Larger batch for more stable gradients
    
    K_epochs = 10               
    eps_clip = 0.2
    gamma = 0.99
    lam = 0.95                  
    
    lr_actor = 0.0003
    lr_critic = 0.0003
    ###############################

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='resume training from existing checkpoint')
    args = parser.parse_args()

    envs = gym.vector.SyncVectorEnv([make_env(env_name) for _ in range(num_envs)])
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    normalizer = ObservationNormalizer(state_dim)
    memory = RolloutBuffer()
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, lam)
    
    # Load existing checkpoint if resume flag is set
    checkpoint_path = f"ppo_{env_name}_final.pth"
    normalizer_path = f"ppo_{env_name}_normalizer.pth"
    
    if args.resume:
        if os.path.exists(checkpoint_path):
            print(f"Resuming training: Loading existing model from {checkpoint_path}")
            ppo_agent.load(checkpoint_path)
        else:
            print(f"Warning: --resume set but {checkpoint_path} not found. Starting from scratch.")
        
        if os.path.exists(normalizer_path):
            print(f"Loading existing normalizer from {normalizer_path}")
            checkpoint = torch.load(normalizer_path, weights_only=False)
            normalizer.mean = checkpoint['mean']
            normalizer.std = checkpoint['std']
            normalizer.n = checkpoint['n']
    else:
        print("Starting training from scratch.")
    
    i_update = 0
    time_step = 0
    
    # Tracking episode rewards
    episode_rewards = np.zeros(num_envs)
    final_rewards = []

    states, _ = envs.reset()
    states = normalizer(states)

    while time_step <= max_training_timesteps:
        for t in range(update_timestep // num_envs):
            action_clipped, action_raw, logprob, state_val = ppo_agent.select_action(states)
            next_states, rewards, terminated, truncated, infos = envs.step(action_clipped)
            next_states = normalizer(next_states)
            
            memory.add(states, action_raw, logprob, state_val.flatten(), rewards, (terminated | truncated))
            
            # Track episode rewards
            episode_rewards += rewards
            for i in range(num_envs):
                if terminated[i] or truncated[i]:
                    final_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0
            
            states = next_states
            time_step += num_envs

        # Update
        with torch.no_grad():
            _, _, _, next_state_values = ppo_agent.select_action(states)
            
        ppo_agent.update(memory, next_state_values.flatten())
        memory.clear()
        i_update += 1
        
        avg_reward = np.mean(final_rewards[-20:]) if len(final_rewards) > 0 else 0
        print(f"\rUpdate: {i_update} \t Timestep: {time_step} \t Avg Ep Reward: {avg_reward:.2f}", end='')
        # Logging
        if i_update % 100 == 0:
            print(f"\rUpdate: {i_update} \t Timestep: {time_step} \t Avg Ep Reward: {avg_reward:.2f}")

        # Save checkpoint periodically
        if i_update % 100 == 0:
            ppo_agent.save(f"ppo_{env_name}_final.pth")
            torch.save({'mean': normalizer.mean, 'std': normalizer.std, 'n': normalizer.n}, 
                       f"ppo_{env_name}_normalizer.pth")

    ppo_agent.save(f"ppo_{env_name}_final.pth")
    torch.save({'mean': normalizer.mean, 'std': normalizer.std, 'n': normalizer.n}, 
               f"ppo_{env_name}_normalizer.pth")
    envs.close()

if __name__ == '__main__':
    train()
