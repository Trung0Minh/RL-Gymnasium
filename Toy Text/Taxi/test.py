import gymnasium as gym
import numpy as np
import pickle
import config
import time

def test():
    # Create the environment with render_mode="human" to see the agent play
    # Note: If running in a headless environment, use "ansi" or omit rendering
    try:
        env = gym.make(config.ENV_NAME, render_mode="human")
    except Exception:
        print("Human render mode failed, falling back to ansi")
        env = gym.make(config.ENV_NAME, render_mode="ansi")
    
    # Load the trained Q-table
    try:
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        print("Q-table not found! Please run train.py first.")
        return

    print("Starting testing...")
    
    total_rewards = []
    num_test_episodes = 5
    
    for episode in range(num_test_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        print(f"Episode {episode + 1}")
        
        while not done:
            # Always exploit the Q-table during testing
            action = np.argmax(q_table[state, :])
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Print environment state if using ansi
            if env.render_mode == "ansi":
                print(env.render())
                time.sleep(0.1)
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} finished with reward: {episode_reward}")
    
    print(f"Average Reward over {num_test_episodes} episodes: {np.mean(total_rewards)}")
    env.close()

if __name__ == "__main__":
    test()
