import gymnasium as gym
import torch
import numpy as np
from agent import PPOAgent
from collections import deque

def train():
    num_envs = 16
    num_steps = 2048
    num_updates = 1000
    env_id = "Humanoid-v5"
    
    def make_env():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, 
            lambda obs: np.clip(obs, -10, 10),
            observation_space=env.observation_space
        )
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(
            env, 
            lambda reward: np.clip(reward, -10, 10)
        )
        return env

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    
    agent = PPOAgent(obs_dim, action_dim)
    
    obs_b = torch.zeros((num_steps, num_envs, obs_dim))
    action_b = torch.zeros((num_steps, num_envs, action_dim))
    logprob_b = torch.zeros((num_steps, num_envs))
    reward_b = torch.zeros((num_steps, num_envs))
    done_b = torch.zeros((num_steps, num_envs))
    value_b = torch.zeros((num_steps, num_envs))
    
    # Track rolling rewards
    reward_window = deque(maxlen=50)
    
    obs, _ = envs.reset()
    print(f"Starting training on {num_envs} environments ({env_id})...")

    for update in range(1, num_updates + 1):
        for step in range(num_steps):
            obs_b[step] = torch.from_numpy(obs)
            
            action, logprob, value = agent.select_action(obs)
            action_b[step] = torch.from_numpy(action)
            logprob_b[step] = torch.from_numpy(logprob)
            value_b[step] = torch.from_numpy(value)
            
            next_obs, reward, terminated, truncated, infos = envs.step(action)
            reward_b[step] = torch.from_numpy(reward)
            done = terminated | truncated
            done_b[step] = torch.from_numpy(done)
            
            obs = next_obs
            
            if "_episode" in infos:
                for i in range(num_envs):
                    if infos["_episode"][i]:
                        reward_window.append(infos["episode"]["r"][i])
            
            # Constant update on the same line
            if step % 100 == 0:
                avg_r = np.mean(reward_window) if reward_window else 0.0
                print(f"\rUpdate {update}/{num_updates} | Step {step}/{num_steps} | Avg Reward: {avg_r:.2f}", end="")

        _, _, next_value = agent.select_action(obs)
        agent.update(obs_b, action_b, logprob_b, reward_b, done_b, value_b, next_value, done)
        
        # Print end of update status
        avg_r = np.mean(reward_window) if reward_window else 0.0
        print(f"\rUpdate {update}/{num_updates} | Avg Reward: {avg_r:.2f}                      ")
        
        if update % 10 == 0:
            torch.save(agent.network.state_dict(), "humanoid_final.pt")

    torch.save(agent.network.state_dict(), "humanoid_final.pt")
    print("\nTraining finished. Model saved to humanoid_final.pt")
    envs.close()

if __name__ == "__main__":
    train()
