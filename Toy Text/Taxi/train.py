import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from agent import QLearningAgent
import config

def train():
    parser = argparse.ArgumentParser(description='Train Q-Learning agent on Taxi.')
    parser.add_argument('--episodes', type=int, default=config.EPISODES, help='Number of episodes to train.')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=config.DISCOUNT_FACTOR, help='Discount factor.')
    parser.add_argument('--epsilon', type=float, default=config.EPSILON, help='Initial exploration rate.')
    parser.add_argument('--epsilon_decay', type=float, default=config.EPSILON_DECAY, help='Exploration decay rate.')
    parser.add_argument('--min_epsilon', type=float, default=config.MIN_EPSILON, help='Minimum exploration rate.')
    parser.add_argument('--continue_train', action='store_true', help='Continue training from saved checkpoint.')
    
    args = parser.parse_args()

    env = gym.make(config.ENV_NAME)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Setup save paths
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    save_path = os.path.join(config.MODELS_DIR, "q_table.pkl")
    best_save_path = os.path.join(config.MODELS_DIR, "best_q_table.pkl")

    agent = QLearningAgent(
        n_states, 
        n_actions, 
        learning_rate=args.lr, 
        discount_factor=args.gamma, 
        epsilon=args.epsilon, 
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon
    )
    
    if args.continue_train and os.path.exists(save_path):
        print(f"Loading checkpoint from {save_path}...")
        agent.load(save_path)

    rewards = []
    best_avg_reward = -float('inf')
    
    print(f"Starting training for {args.episodes} episodes...")
    
    for episode in range(args.episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        rewards.append(total_reward)
        
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards[-1000:])
            print(f"Episode: {episode + 1}, Reward: {total_reward}, Avg Reward: {avg_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(best_save_path)

    # Final check for best model
    if rewards:
        avg_reward = np.mean(rewards[-min(1000, len(rewards)):])
        if avg_reward > best_avg_reward:
            agent.save(best_save_path)
            print(f"Final model saved as best with Avg Reward: {avg_reward:.4f}")

    agent.save(save_path)
    print(f"Final Q-table saved to {save_path}")
    
    env.close()

    # Plot and save rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Training Rewards over Episodes')
    plt.savefig(config.PLOT_PATH)
    print(f"Training rewards plot saved as {config.PLOT_PATH}")
    plt.show()

if __name__ == "__main__":
    train()
