import argparse
import gymnasium as gym
import pickle
import os
from agent import BlackjackAgent
import config

def main():
    parser = argparse.ArgumentParser(description="Train a Blackjack agent using Q-Learning.")
    parser.add_argument("--episodes", type=int, default=config.EPISODES, help="Number of episodes to train")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=config.DISCOUNT_FACTOR, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=config.EPSILON, help="Initial epsilon")
    parser.add_argument("--decay", type=float, default=config.EPSILON_DECAY, help="Epsilon decay rate")
    parser.add_argument("--min_epsilon", type=float, default=config.MIN_EPSILON, help="Minimum epsilon")
    parser.add_argument("--save_path", type=str, default=config.SAVE_PATH, help="Path to save the Q-table")
    
    args = parser.parse_args()

    # Create the environment (Sutton & Barto rules)
    env = gym.make("Blackjack-v1", sab=config.SAB)
    
    # Initialize the agent
    agent = BlackjackAgent(
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.decay,
        min_epsilon=args.min_epsilon
    )

    print(f"Starting training for {args.episodes} episodes...")
    
    for episode in range(args.episodes):
        state, info = env.reset()
        done = False

        while not done:
            # Select action
            action = agent.get_action(state)
            
            # Execute step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update agent's Q-values
            agent.update(state, action, reward, next_state, terminated)
            
            state = next_state

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Print progress periodically
        if (episode + 1) % 100000 == 0:
            print(f"Episode {episode + 1}/{args.episodes}, Current Epsilon: {agent.epsilon:.4f}")

    print("Training finished.")
    
    # Save the Q-table as a standard dictionary
    with open(args.save_path, "wb") as f:
        pickle.dump(dict(agent.q_values), f)
    
    print(f"Q-table saved to {args.save_path}")
    env.close()

if __name__ == "__main__":
    main()
