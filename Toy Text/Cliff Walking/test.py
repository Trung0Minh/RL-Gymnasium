import gymnasium as gym
import numpy as np
import config
import time

def test():
    # Load the trained Q-table
    try:
        q_table = np.load(config.MODEL_FILENAME)
    except FileNotFoundError:
        print(f"Error: {config.MODEL_FILENAME} not found. Please train the agent first.")
        return

    # Initialize environment with render_mode="ansi" to see output in terminal
    env = gym.make(config.ENV_NAME, render_mode="human")
    
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    print("\nStarting Test...")
    print(env.render())
    
    while not (done or truncated):
        # Always take the best action (exploit)
        action = np.argmax(q_table[state, :])
        
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        time.sleep(0.3)  # Slow down the visualization
        
    print(f"\nTest Complete. Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    test()
