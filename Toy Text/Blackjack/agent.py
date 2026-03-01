import numpy as np
import random
import pickle
from collections import defaultdict

class BlackjackAgent:
    def __init__(self, learning_rate=0.01, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.999998, min_epsilon=0.05):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: (player_sum, dealer_card, usable_ace) -> [q_stick, q_hit]
        self.q_values = defaultdict(lambda: np.zeros(2))

    def get_action(self, state, is_training=True):
        """Epsilon-greedy action selection."""
        if is_training and random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            return int(np.argmax(self.q_values[state]))

    def update(self, state, action, reward, next_state, terminated):
        """Q-learning update rule."""
        future_q = 0 if terminated else np.max(self.q_values[next_state])
        
        # Temporal Difference (TD) Target
        target = reward + self.gamma * future_q
        
        # Update Q-value
        self.q_values[state][action] += self.lr * (target - self.q_values[state][action])

    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save the Q-table."""
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_values), f)

    def load(self, path):
        """Load the Q-table."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.q_values = defaultdict(lambda: np.zeros(2), data)
