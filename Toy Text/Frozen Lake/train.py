import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from agent import QLearningAgent
import config

def train():
    parser = argparse.ArgumentParser(description='Train Q-Learning agent on Frozen Lake.')
    parser.add_argument('--episodes', type=int, default=config.EPISODES, help='Number of episodes to train.')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=config.DISCOUNT_FACTOR, help='Discount factor.')
    parser.add_argument('--epsilon', type=float, default=config.EPSILON, help='Initial exploration rate.')
    parser.add_argument('--epsilon_decay', type=float, default=config.EPSILON_DECAY, help='Exploration decay rate.')
    parser.add_argument('--min_epsilon', type=float, default=config.MIN_EPSILON, help='Minimum exploration rate.')
    parser.add_argument('--continue_train', action='store_true', help='Continue training from saved checkpoint.')
    parser.add_argument('--slippery', type=str, default='false', choices=['true', 'false'], help='Slippery mode (true/false).')
    
    args = parser.parse_args()

    is_slippery = args.slippery == 'true'
    env = gym.make(config.ENV_NAME, is_slippery=is_slippery, map_name=config.MAP_NAME)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Setup save paths
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    mode_str = "slippery" if is_slippery else "non_slippery"
    save_path = os.path.join(config.MODELS_DIR, f"q_table_{mode_str}.pkl")
    best_save_path = os.path.join(config.MODELS_DIR, f"best_q_table_{mode_str}.pkl")
    plot_path = f"training_rewards_{mode_str}.png"

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
    
    print(f"Starting training for {args.episodes} episodes (Slippery: {is_slippery})...")
    
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

    # Plot and save rewards as a bar plot (Success Rate over bins)
    bin_size = 1000
    if len(rewards) >= bin_size:
        # Group rewards into bins and calculate mean for each bin
        num_bins = len(rewards) // bin_size
        binned_rewards = np.array(rewards[:num_bins * bin_size]).reshape(num_bins, bin_size)
        success_rates = np.mean(binned_rewards, axis=1)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(success_rates)), success_rates, width=1.0, align='edge')
        plt.xlabel(f'Episode Blocks (x{bin_size})')
        plt.ylabel('Success Rate (Average Reward)')
        plt.title(f'Training Success Rate (Slippery: {is_slippery})')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(plot_path)
        print(f"Training rewards bar plot saved as {plot_path}")
        plt.show()
    else:
        # Fallback for very short training sessions
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(rewards)), rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title(f'Training Rewards (Slippery: {is_slippery})')
        plt.savefig(plot_path)
        plt.show()

if __name__ == "__main__":
    train()
