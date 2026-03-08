import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, n_states, n_actions, config, device=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config
        self.lr = config.learning_rate
        self.gamma = config.discount_factor
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.min_epsilon = config.min_epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def act(self, state, is_training=True):
        """Epsilon-greedy action selection."""
        if is_training and np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """Q-learning update rule."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (0 if done else self.gamma * self.q_table[next_state][best_next_action])
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def step(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save the Q-table."""
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        """Load the Q-table."""
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
