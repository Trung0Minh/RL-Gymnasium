import gymnasium as gym
import torch
import numpy as np
import argparse
from agent import Agent
from env_utils import BalancingAcrobotWrapper
import config

def test(env, agent, n_episodes=5):
    """
    Standardized Test Loop.
    """
    print(f"Starting test for {n_episodes} episodes. Press Ctrl+C to stop.")
    
    try:
        for i_episode in range(1, n_episodes + 1):
            state, info = env.reset()
            score = 0
            while True:
                action = agent.act(state, eps=0.0)
                state, reward, terminated, truncated, info = env.step(action)
                score += reward
                
                if terminated or truncated:
                    print(f"Episode {i_episode}: Final Score: {score}")
                    break
                    
                env.render()
            
    except KeyboardInterrupt:
        print(f"\nStopped by user.")
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description='DQN Acrobot Testing')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'single'], help='Training objective mode')
    parser.add_argument('--checkpoint', type=str, default='model_weight/checkpoint.pth', help='Path to checkpoint')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to test')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed')
    
    args = parser.parse_args()

    env = gym.make('Acrobot-v1', render_mode='human')
    env = BalancingAcrobotWrapper(env, mode=args.mode)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = Agent(state_size=state_size, action_size=action_size, seed=args.seed)
    
    print(f"Loading weights from {args.checkpoint}...")
    try:
        agent.load(args.checkpoint)
    except FileNotFoundError:
        print(f"Error: '{args.checkpoint}' not found. Please train the agent first.")
        return

    test(env, agent, n_episodes=args.episodes)

if __name__ == "__main__":
    main()
