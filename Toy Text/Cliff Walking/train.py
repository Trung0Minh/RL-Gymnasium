import gymnasium as gym
import numpy as np
import config

def train():
    # Initialize environment
    env = gym.make(config.ENV_NAME)
    
    # Initialize Q-table with zeros
    # Observation space: 48 states (4x12 grid)
    # Action space: 4 actions (Up, Right, Down, Left)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    
    epsilon = config.EPSILON
    
    for episode in range(config.TOTAL_EPISODES):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state, :])  # Exploit
                
            # Take action and observe results
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Q-learning update
            # Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]
            q_table[state, action] = q_table[state, action] + config.LEARNING_RATE * (
                reward + config.DISCOUNT_FACTOR * np.max(q_table[next_state, :]) - q_table[state, action]
            )
            
            state = next_state
            total_reward += reward
            
        # Epsilon decay
        epsilon = max(config.MIN_EPSILON, epsilon * config.EPSILON_DECAY)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {epsilon:.4f}")
            
    # Save the trained Q-table
    np.save(config.MODEL_FILENAME, q_table)
    print(f"Training complete. Q-table saved as {config.MODEL_FILENAME}.")
    
    env.close()

if __name__ == "__main__":
    train()
