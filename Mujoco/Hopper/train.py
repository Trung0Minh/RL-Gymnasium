import gymnasium as gym
import torch
import numpy as np
import argparse
import os
from collections import deque
from agent import PPOAgent
from config import PPOConfig

def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk

def train(cfg: PPOConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    envs = gym.vector.SyncVectorEnv([make_env(cfg.env_id) for _ in range(cfg.num_envs)])
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.vector.NormalizeObservation(envs)
    obs_rms_wrapper = envs
    envs = gym.wrappers.vector.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))
    envs = gym.wrappers.vector.NormalizeReward(envs)
    envs = gym.wrappers.vector.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    
    agent = PPOAgent(obs_dim, action_dim, device, hidden_dim=cfg.hidden_dim)
    
    if cfg.resume:
        model_path = f"{cfg.checkpoint_dir}/hopper_ppo.pt"
        stats_path = f"{cfg.checkpoint_dir}/hopper_obs_rms.pt"
        if os.path.exists(model_path):
            agent.network.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            print(f"Resumed model from {model_path}")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location=device, weights_only=False)
            obs_rms_wrapper.obs_rms.mean = stats["mean"]
            obs_rms_wrapper.obs_rms.var = stats["var"]
            print(f"Resumed normalization stats from {stats_path}")

    obs = torch.zeros((cfg.num_steps, cfg.num_envs, obs_dim)).to(device)
    actions = torch.zeros((cfg.num_steps, cfg.num_envs, action_dim)).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

    returns_deque = deque(maxlen=50)
    global_step = 0
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)

    for update in range(1, cfg.num_updates + 1):
        for step in range(0, cfg.num_steps):
            global_step += cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(terminated | truncated).to(device)

            if "episode" in infos:
                for r in infos["episode"]["r"][infos["_episode"]]:
                    returns_deque.append(r)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + agent.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + agent.gamma * agent.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        agent.update(b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, cfg.update_epochs, cfg.mini_batch_size)
        
        avg_reward = np.mean(returns_deque) if returns_deque else 0
        log_str = f"Step: {global_step} | Update: {update}/{cfg.num_updates} | Avg Ep Reward: {avg_reward:.2f}"
        
        if update % 50 == 0:
            print(f"\r{log_str}")
        else:
            print(f"\r{log_str}", end="", flush=True)

        if update % 10 == 0:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            torch.save(agent.network.state_dict(), f"{cfg.checkpoint_dir}/hopper_ppo.pt")
            torch.save({
                "mean": obs_rms_wrapper.obs_rms.mean,
                "var": obs_rms_wrapper.obs_rms.var
            }, f"{cfg.checkpoint_dir}/hopper_obs_rms.pt")

    print()
    envs.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key, value in PPOConfig().__dict__.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    config = PPOConfig(**vars(args))
    train(config)
