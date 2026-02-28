import gymnasium as gym
import numpy as np
import pickle
import config

def train():
    # Create the environment
    env = gym.make(config.ENV_NAME)
    
    # Initialize Q-table: 500 states x 6 actions
    # Taxi has 500 discrete states and 6 discrete actions
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.zeros((state_size, action_size))
    
    epsilon = config.EPSILON
    
    print("Starting training...")
    
    for episode in range(config.TOTAL_EPISODES):
        state, info = env.reset()
        done = False
        
        for step in range(config.MAX_STEPS_PER_EPISODE):
            # Epsilon-greedy exploration strategy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state, :])  # Exploit
            
            # Perform the action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update Q-table using Bellman Equation
            # Q(s,a) = (1-alpha)*Q(s,a) + alpha*(reward + gamma * max(Q(s',a')))
            q_table[state, action] = (1 - config.LEARNING_RATE) * q_table[state, action] + \
                                     config.LEARNING_RATE * (reward + config.DISCOUNT_FACTOR * np.max(q_table[next_state, :]))
            
            state = next_state
            
            if done:
                break
        
        # Decay epsilon to reduce exploration over time
        epsilon = max(config.MIN_EPSILON, epsilon * config.EPSILON_DECAY)
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{config.TOTAL_EPISODES}, Epsilon: {epsilon:.4f}")
            
    print("Training finished!")
    
    # Save the trained Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Q-table saved to q_table.pkl")
    
    env.close()

if __name__ == "__main__":
    train()
