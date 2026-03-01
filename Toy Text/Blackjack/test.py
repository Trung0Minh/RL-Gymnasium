import gymnasium as gym
import numpy as np
import argparse
import os
from agent import BlackjackAgent
import config

def test():
    parser = argparse.ArgumentParser(description='Test trained Q-Learning agent on Blackjack.')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to test.')
    parser.add_argument('--model', type=str, default=None, help='Path to the saved Q-table. If None, uses default best model.')
    parser.add_argument('--render', type=str, default="human", choices=['human', 'rgb_array'], help='Render mode (e.g., human, rgb_array). Default: human.')
    
    args = parser.parse_args()

    # Initialize the environment
    render_mode = args.render
    env = gym.make(config.ENV_NAME, sab=config.SAB, render_mode=render_mode)
    
    # Setup model path
    if args.model is None:
        model_path = os.path.join(config.MODELS_DIR, "best_q_table.pkl")
    else:
        model_path = args.model

    # Initialize agent and load Q-table
    agent = BlackjackAgent()
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: {model_path} not found. Please train the agent first.")
        return

    wins = 0
    losses = 0
    draws = 0
    
    print(f"Testing the agent for {args.episodes} episodes...")

    for episode in range(args.episodes):
        state, info = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state, is_training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
            
    print(f"Results over {args.episodes} episodes:")
    print(f"Wins: {wins} ({(wins/args.episodes)*100:.2f}%)")
    print(f"Losses: {losses} ({(losses/args.episodes)*100:.2f}%)")
    print(f"Draws: {draws} ({(draws/args.episodes)*100:.2f}%)")
    
    env.close()

if __name__ == "__main__":
    test()
