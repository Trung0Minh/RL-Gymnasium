import gymnasium as gym
import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from collections import deque
from agent import DQNAgent
from config import DQNConfig

def make_env(env_id, seed, max_t):
    def thunk():
        env = gym.make(env_id, max_episode_steps=max_t)
        env.action_space.seed(seed)
        return env
    return thunk

def train(cfg: DQNConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    envs = gym.vector.SyncVectorEnv([make_env(cfg.env_id, cfg.seed + i, cfg.max_t) for i in range(cfg.num_envs)])
    
    state_size = envs.single_observation_space.shape[0]
    action_size = envs.single_action_space.n
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, config=cfg, device=device)
    
    checkpoint_path = f"{cfg.checkpoint_dir}/mountain_car_dqn.pt"
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    if cfg.resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        agent.load(checkpoint_path)

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = cfg.eps_start                # initialize epsilon

    print(f"Starting training on {device}...")
    
    for i_episode in range(1, cfg.num_episodes + 1):
        state, info = envs.reset()
        score = np.zeros(cfg.num_envs)
        
        for t in range(cfg.max_t):
            # Act for each env
            if cfg.num_envs == 1:
                action = [agent.act(state[0], eps)]
            else:
                actions = []
                for s in state:
                    actions.append(agent.act(s, eps))
                action = actions
                
            next_state, reward, terminated, truncated, info = envs.step(action)
            
            # Step for each environment
            for i in range(cfg.num_envs):
                done = terminated[i] or truncated[i]
                
                # --- Reward Shaping (Mountain Car Specific) ---
                # Position is next_state[i][0], Velocity is next_state[i][1]
                pos = next_state[i][0]
                custom_reward = reward[i]
                if pos >= 0.5:
                    custom_reward += 10
                custom_reward += abs(next_state[i][1]) * 10
                
                agent.step(state[i], action[i], custom_reward, next_state[i], done)
                score[i] += reward[i]
            
            state = next_state
            if np.any(terminated) or np.any(truncated):
                break 
                
        avg_ep_reward = np.mean(score)
        scores_window.append(avg_ep_reward)
        scores.append(avg_ep_reward)
        eps = max(cfg.eps_end, cfg.eps_decay * eps)
        
        avg_score = np.mean(scores_window)
        log_str = f"Episode {i_episode}\tAverage Score: {avg_score:.2f}"
        
        if i_episode % 100 == 0:
            print(f"\r{log_str}")
            agent.save(checkpoint_path)
        else:
            print(f"\r{log_str}", end="", flush=True)

    print()
    envs.close()
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key, value in DQNConfig().__dict__.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value) if value is not None else str, default=value)
    
    args = parser.parse_args()
    config = DQNConfig(**vars(args))
    
    scores = train(config)

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title("DQN Training Scores for Mountain Car")
    plt.savefig('rewards.png')
    print(f"\nTraining plot saved as rewards.png")
