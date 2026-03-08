import gymnasium as gym
import numpy as np
import argparse
import os
from collections import deque
import matplotlib.pyplot as plt
from agent import QLearningAgent
from config import QConfig

def train(cfg: QConfig):
    env = gym.make(cfg.env_id)
    agent = QLearningAgent(n_states=env.observation_space.n, n_actions=env.action_space.n, config=cfg)
    
    checkpoint_path = f"{cfg.checkpoint_dir}/taxi_q.pkl"
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    if cfg.resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        agent.load(checkpoint_path)

    rewards = []
    rewards_window = deque(maxlen=100)
    
    print(f"Starting training on CPU...")
    
    for i_episode in range(1, cfg.num_episodes + 1):
        state, info = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            score += reward
        
        agent.step() # epsilon decay
        rewards.append(score)
        rewards_window.append(score)
        
        avg_reward = np.mean(rewards_window)
        log_str = f"Episode {i_episode}\tAverage Reward: {avg_reward:.2f}\tEpsilon: {agent.epsilon:.4f}"
        
        if i_episode % 1000 == 0:
            print(f"\r{log_str}")
            agent.save(checkpoint_path)
        else:
            print(f"\r{log_str}", end="", flush=True)

    print()
    env.close()
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key, value in QConfig().__dict__.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value) if value is not None else str, default=value)
    
    args = parser.parse_args()
    config = QConfig(**vars(args))
    
    rewards = train(config)

    # Plotting
    plt.figure(figsize=(10, 5))
    window = 100
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg)
    plt.xlabel(f'Episodes (window={window})')
    plt.ylabel('Average Rewards')
    plt.title('DQN Training Rewards for Taxi')
    plt.savefig('rewards.png')
    print(f"\nTraining rewards plot saved as rewards.png")
