import gymnasium as gym
import numpy as np
import argparse
from agent import QLearningAgent
import config

def test(args):
    # Load the trained Q-table
    try:
        q_table = np.load(args.q_table)
    except FileNotFoundError:
        print(f"Error: {args.q_table} not found. Please train the agent first.")
        return

    # Initialize the environment
    env = gym.make(config.ENV_NAME, is_slippery=args.slippery, map_name=args.map, render_mode=args.render)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Initialize agent and set the loaded Q-table
    agent = QLearningAgent(n_states, n_actions)
    agent.q_table = q_table
    
    for episode in range(args.episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        print(f"Episode: {episode + 1}")
        while not done:
            action = agent.get_action(state, is_training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            
            if args.render == "ansi":
                print(env.render())
            else:
                env.render()
        
        print(f"Total Reward: {total_reward}")
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained Q-Learning agent on Frozen Lake.')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to test.')
    parser.add_argument('--q_table', type=str, default="q_table.npy", help='Path to the saved Q-table.')
    parser.add_argument('--render', type=str, default=config.RENDER_MODE_TEST, help='Render mode (human or ansi).')
    parser.add_argument('--slippery', action='store_true', default=config.IS_SLIPPERY, help='Whether the environment is slippery.')
    parser.add_argument('--no-slippery', action='store_false', dest='slippery', help='Make the environment non-slippery.')
    parser.add_argument('--map', type=str, default=config.MAP_NAME, help='Map size (4x4 or 8x8).')

    args = parser.parse_args()
    test(args)
