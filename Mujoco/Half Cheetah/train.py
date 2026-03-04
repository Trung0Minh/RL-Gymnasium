import gymnasium as gym
import torch
import numpy as np
import pickle
from ppo import PPOAgent

class PosturePenaltyWrapper(gym.Wrapper):
    """
    Penalizes the cheetah for being upside down or extreme torso angles.
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        torso_angle = obs[1]
        if abs(torso_angle) > 1.0:
            reward -= 0.5 
        return obs, reward, terminated, truncated, info

def make_env(env_id):
    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = PosturePenaltyWrapper(env)
    return env

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env_id = "HalfCheetah-v5"
    num_envs = 8
    
    # Create the parallel environments
    envs = gym.vector.AsyncVectorEnv([lambda: make_env(env_id) for _ in range(num_envs)])
    
    # Vector-level wrappers: These maintain unified statistics across all 8 envs
    # We keep a reference to the normalization wrapper to save stats later
    envs = gym.wrappers.vector.NormalizeObservation(envs)
    obs_rms_wrapper = envs 
    
    envs = gym.wrappers.vector.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))
    envs = gym.wrappers.vector.NormalizeReward(envs)
    envs = gym.wrappers.vector.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    
    num_inputs = envs.single_observation_space.shape[0]
    num_actions = envs.single_action_space.shape[0]
    
    agent = PPOAgent(num_inputs, num_actions, device)
    
    num_steps = 2048
    total_timesteps = 2_000_000
    batch_size = int(num_envs * num_steps)
    num_updates = total_timesteps // batch_size
    
    obs = torch.zeros((num_steps, num_envs, num_inputs)).to(device)
    actions = torch.zeros((num_steps, num_envs, num_actions)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    
    global_step = 0
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    
    print(f"Training started with {num_envs} environments...")
    last_return = 0.0
    
    def get_lr(update):
        frac = 1.0 - (update - 1.0) / num_updates
        return frac * 3e-4

    for update in range(1, num_updates + 1):
        lr = get_lr(update)
        agent.optimizer.param_groups[0]["lr"] = lr

        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.network.get_action_and_value(next_obs)
                values[step] = value.flatten()
                
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs_np, reward_np, terminated_np, truncated_np, info = envs.step(action.cpu().numpy())
            done_np = np.logical_or(terminated_np, truncated_np)
            
            rewards[step] = torch.tensor(reward_np).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done_np).to(device)

            if "episode" in info:
                if "_r" in info["episode"]:
                    for i in range(len(info["episode"]["_r"])):
                        if info["episode"]["_r"][i]:
                            last_return = float(info["episode"]["r"][i])
        
        with torch.no_grad():
            next_value = agent.network.get_value(next_obs).reshape(1, -1)
            advantages, returns = agent.compute_gae(rewards, values, dones, next_value, next_done)
        
        b_obs = obs.reshape((-1, num_inputs))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, num_actions))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        agent.update(b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values)

        mean_batch_reward = rewards.mean().item()
        log_msg = f"\rUpdate: {update}/{num_updates} | Global Step: {global_step} | Mean Batch Reward: {mean_batch_reward:.4f} | Last Return: {last_return:.2f}"
        
        if update % 5 == 0 or update == num_updates:
            print(log_msg)
        else:
            print(log_msg, end="", flush=True)

    # Save the model
    torch.save(agent.network.state_dict(), "half_cheetah_ppo.pt")
    
    # CRITICAL: Save the normalization statistics!
    # Use the specifically saved reference to the NormalizeObservation wrapper
    obs_rms = {
        "mean": obs_rms_wrapper.obs_rms.mean,
        "var": obs_rms_wrapper.obs_rms.var
    }
    with open("half_cheetah_obs_rms.pkl", "wb") as f:
        pickle.dump(obs_rms, f)
        
    print("Model and normalization stats saved.")
    envs.close()

if __name__ == "__main__":
    main()
