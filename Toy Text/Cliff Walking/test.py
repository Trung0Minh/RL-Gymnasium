import gymnasium as gym
import numpy as np
import argparse
import os
from agent import QLearningAgent
import config

def test():
    parser = argparse.ArgumentParser(description='Test trained Q-Learning agent on Cliff Walking.')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to test.')
    parser.add_argument('--model', type=str, default=None, help='Path to the saved Q-table. If None, uses default best model.')
    parser.add_argument('--render', type=str, default="human", choices=['human', 'rgb_array', 'ansi'], help='Render mode (e.g., human, rgb_array, ansi). Default: human.')
    
    args = parser.parse_args()

    # Initialize the environment
    env = gym.make(config.ENV_NAME, render_mode=args.render)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Setup model path
    if args.model is None:
        model_path = os.path.join(config.MODELS_DIR, "best_q_table.pkl")
    else:
        model_path = args.model

    # Initialize agent and load Q-table
    agent = QLearningAgent(n_states, n_actions)
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: {model_path} not found. Please train the agent first.")
        return

    total_rewards = []
    
    print(f"Testing the agent for {args.episodes} episodes...")

    for episode in range(args.episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.get_action(state, is_training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
        
    print(f"Average Reward over {args.episodes} episodes: {np.mean(total_rewards):.4f}")
    env.close()

if __name__ == "__main__":
    test()
