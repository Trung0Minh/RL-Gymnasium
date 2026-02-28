import gymnasium as gym
import pickle
import numpy as np
import argparse
import config

def evaluate(episodes, model_path):
    env = gym.make("Blackjack-v1", render_mode="human", sab=config.SAB)
    
    # Load Q-table
    try:
        with open(model_path, "rb") as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please train the agent first.")
        return

    wins = 0
    losses = 0
    draws = 0

    print(f"Testing the agent for {episodes} episodes...")

    for i in range(episodes):
        state, info = env.reset()
        done = False
        
        while not done:
            # Optimal action from Q-table (greedy selection)
            # If state not in table, default to Stick (0)
            if state in q_table:
                action = int(np.argmax(q_table[state]))
            else:
                action = 0
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    print("Results:")
    print(f"Wins: {wins} ({(wins/episodes)*100:.2f}%)")
    print(f"Losses: {losses} ({(losses/episodes)*100:.2f}%)")
    print(f"Draws: {draws} ({(draws/episodes)*100:.2f}%)")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained Blackjack agent.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to test")
    parser.add_argument("--model", type=str, default=config.SAVE_PATH, help="Path to the saved Q-table")
    
    args = parser.parse_args()
    evaluate(args.episodes, args.model)
