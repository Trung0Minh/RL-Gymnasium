import gymnasium as gym
import torch
import numpy as np
import argparse
from dqn_agent import Agent
import config

def test(env, agent, n_episodes=5, checkpoint_path='best_checkpoint.pth'):
    """Visualize the trained agent."""
    
    # Load the saved weights
    print(f"Loading weights from {checkpoint_path}...")
    try:
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
    except FileNotFoundError:
        print(f"Error: '{checkpoint_path}' not found. Please train the agent first.")
        return
        
    agent.qnetwork_local.eval()

    print(f"Starting test for {n_episodes} episodes. Press Ctrl+C to stop.")
    try:
        for i_episode in range(1, n_episodes + 1):
            state, info = env.reset()
            score = 0
            while True:
                # In testing, we use epsilon=0 to always take the best action
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
    parser = argparse.ArgumentParser(description='DQN Cart Pole Testing')
    parser.add_argument('--checkpoint', type=str, default='best_checkpoint.pth', help='Path to checkpoint')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to test')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed')
    
    args = parser.parse_args()

    # Create the environment with human rendering
    env = gym.make('CartPole-v1', render_mode='human', max_episode_steps=10000)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize the agent
    agent = Agent(state_size=state_size, action_size=action_size, seed=args.seed)
    
    test(env, agent, n_episodes=args.episodes, checkpoint_path=args.checkpoint)

if __name__ == "__main__":
    main()
