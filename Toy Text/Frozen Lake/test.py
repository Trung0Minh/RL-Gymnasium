import gymnasium as gym
import numpy as np
import argparse
import os
from agent import QLearningAgent
import config

def test():
    parser = argparse.ArgumentParser(description='Test trained Q-Learning agent on Frozen Lake.')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to test.')
    parser.add_argument('--slippery', type=str, default='false', choices=['true', 'false'], help='Slippery mode (true/false).')
    parser.add_argument('--model', type=str, default=None, help='Path to the saved Q-table. If None, uses default path based on slippery mode.')
    parser.add_argument('--render', type=str, default="human", choices=['human', 'ansi', 'rgb_array'], help='Render mode (e.g., human, ansi, rgb_array). Default: human.')
    
    args = parser.parse_args()

    # Initialize the environment
    is_slippery = args.slippery == 'true'
    env = gym.make(config.ENV_NAME, is_slippery=is_slippery, map_name=config.MAP_NAME, render_mode=args.render)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Setup model path
    if args.model is None:
        mode_str = "slippery" if is_slippery else "non_slippery"
        model_path = os.path.join(config.MODELS_DIR, f"best_q_table_{mode_str}.pkl")
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
    
    print(f"Testing the agent for {args.episodes} episodes (Slippery: {is_slippery})...")

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
