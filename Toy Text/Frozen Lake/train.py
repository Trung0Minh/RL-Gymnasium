import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
from agent import QLearningAgent
import config

def train(args):
    env = gym.make(config.ENV_NAME, is_slippery=args.slippery, map_name=args.map)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    agent = QLearningAgent(
        n_states, 
        n_actions, 
        learning_rate=args.lr, 
        discount_factor=args.gamma, 
        exploration_rate=args.epsilon, 
        exploration_decay=args.epsilon_decay
    )
    
    rewards = []
    
    for episode in range(args.episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        agent.decay_exploration()
        rewards.append(total_reward)
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode: {episode + 1}, Average Reward: {np.mean(rewards[-1000:]):.4f}, Epsilon: {agent.epsilon:.4f}")
            
    env.close()
    return agent, rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Q-Learning agent on Frozen Lake.')
    parser.add_argument('--episodes', type=int, default=config.DEFAULT_EPISODES, help='Number of episodes to train.')
    parser.add_argument('--lr', type=float, default=config.DEFAULT_LEARNING_RATE, help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=config.DEFAULT_DISCOUNT_FACTOR, help='Discount factor.')
    parser.add_argument('--epsilon', type=float, default=config.DEFAULT_EXPLORATION_RATE, help='Initial exploration rate.')
    parser.add_argument('--epsilon_decay', type=float, default=config.DEFAULT_EXPLORATION_DECAY, help='Exploration decay rate.')
    parser.add_argument('--slippery', action='store_true', default=config.IS_SLIPPERY, help='Whether the environment is slippery.')
    parser.add_argument('--no-slippery', action='store_false', dest='slippery', help='Make the environment non-slippery.')
    parser.add_argument('--map', type=str, default=config.MAP_NAME, help='Map size (4x4 or 8x8).')
    
    args = parser.parse_args()
    
    agent, rewards = train(args)
    
    # Save the Q-table
    np.save("q_table.npy", agent.q_table)
    print("Q-table saved to q_table.npy")
    
    # Plot rewards
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Training Rewards over Episodes')
    plt.savefig('training_rewards.png')
    print("Training rewards plot saved as training_rewards.png")
