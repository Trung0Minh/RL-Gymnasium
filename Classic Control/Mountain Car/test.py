import gymnasium as gym
import torch
import numpy as np
import argparse
from dqn_agent import DQNAgent
import config

def test(env, agent, checkpoint_path='best_checkpoint.pth'):
    """Visualize the trained agent indefinitely."""
    
    # Load the saved weights
    print(f"Loading weights from {checkpoint_path}...")
    try:
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
    except FileNotFoundError:
        print(f"Error: '{checkpoint_path}' not found. Please train the agent first.")
        return
        
    agent.qnetwork_local.eval()

    state, _ = env.reset()
    score = 0
    
    print("Starting test. Press Ctrl+C to stop.")
    try:
        while True:
            # Scale velocity to be in a similar range to position for the NN
            scaled_state = state.copy()
            scaled_state[1] *= 15.0 # Scale ~0.07 to ~1.0
            
            # In testing, we use epsilon=0 to always take the best action
            action = agent.act(scaled_state, eps=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            state = next_state
            
            if terminated or truncated:
                print(f"Final Score: {score}")
                break
                
            env.render()
            
    except KeyboardInterrupt:
        print(f"\nStopped by user. Score reached: {score}")
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description='DQN Mountain Car Testing')
    parser.add_argument('--checkpoint', type=str, default='best_checkpoint.pth', help='Path to checkpoint')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed')
    
    args = parser.parse_args()

    # Create the environment with human rendering
    env = gym.make('MountainCar-v0', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize the agent
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=args.seed)
    
    test(env, agent, checkpoint_path=args.checkpoint)

if __name__ == "__main__":
    main()
