import gymnasium as gym
import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from collections import deque
from agent import TD3Agent
from config import TD3Config
from replay import ReplayBuffer

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk

def train(cfg: TD3Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    envs = gym.vector.SyncVectorEnv([make_env(cfg.env_id, cfg.seed + i) for i in range(cfg.num_envs)])
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.vector.NormalizeObservation(envs)
    obs_rms_wrapper = envs
    envs = gym.wrappers.vector.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))
    # Not using NormalizeReward as it's often more tricky for off-policy.
    
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    
    agent = TD3Agent(state_dim=state_dim, action_dim=action_dim, config=cfg, device=device)
    replay_buffer = ReplayBuffer(state_dim, action_dim, device, max_size=cfg.buffer_size)
    
    checkpoint_path = f"{cfg.checkpoint_dir}/bipedal_walker_td3.pt"
    stats_path = f"{cfg.checkpoint_dir}/bipedal_walker_obs_rms.pt"
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    if cfg.resume:
        if os.path.exists(checkpoint_path):
            agent.load(checkpoint_path)
            print(f"Resumed model from {checkpoint_path}")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location=device, weights_only=False)
            obs_rms_wrapper.obs_rms.mean = stats["mean"]
            obs_rms_wrapper.obs_rms.var = stats["var"]
            print(f"Resumed normalization stats from {stats_path}")

    scores = []
    scores_window = deque(maxlen=100)
    
    state, _ = envs.reset(seed=cfg.seed)
    total_timesteps = 0
    
    print(f"Starting training on {device}...")
    
    for i_episode in range(1, cfg.max_episodes + 1):
        state, _ = envs.reset()
        score = 0
        
        for t in range(cfg.max_t):
            total_timesteps += 1
            
            if total_timesteps < cfg.start_timesteps:
                action = envs.action_space.sample()
            else:
                action = agent.select_action(state)
                # Add exploration noise
                action = (action + np.random.normal(0, 1.0 * cfg.policy_noise, size=action_dim)).clip(-1.0, 1.0)

            next_state, reward, terminated, truncated, info = envs.step(action)
            
            for i in range(cfg.num_envs):
                done = terminated[i] or truncated[i]
                # Note: NormalizeObservation already normalized `state` and `next_state`.
                replay_buffer.add(state[i], action[i], next_state[i], reward[i], done)
                score += reward[i]
            
            state = next_state
            
            if total_timesteps >= cfg.start_timesteps:
                agent.update(replay_buffer)
                
            if np.any(terminated) or np.any(truncated):
                break
                
        scores_window.append(score)
        scores.append(score)
        
        avg_score = np.mean(scores_window)
        log_str = f"Episode {i_episode}\tTotal Steps: {total_timesteps}\tAverage Score: {avg_score:.2f}"
        
        if i_episode % 50 == 0:
            print(f"\r{log_str}")
            agent.save(checkpoint_path)
            torch.save({
                "mean": obs_rms_wrapper.obs_rms.mean,
                "var": obs_rms_wrapper.obs_rms.var
            }, stats_path)
        else:
            print(f"\r{log_str}", end="", flush=True)

    print()
    envs.close()
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key, value in TD3Config().__dict__.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value) if value is not None else str, default=value)
    
    args = parser.parse_args()
    config = TD3Config(**vars(args))
    
    scores = train(config)

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title("TD3 Training Scores for Bipedal Walker")
    plt.savefig('rewards.png')
    print(f"\nTraining plot saved as rewards.png")
