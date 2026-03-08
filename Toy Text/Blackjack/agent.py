import numpy as np
import random
import pickle
from collections import defaultdict

class BlackjackAgent:
    def __init__(self, config, device=None):
        self.config = config
        self.lr = config.learning_rate
        self.gamma = config.discount_factor
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.min_epsilon = config.min_epsilon
        
        # Q-table: (player_sum, dealer_card, usable_ace) -> [q_stick, q_hit]
        self.q_values = defaultdict(lambda: np.zeros(2))

    def act(self, state, is_training=True):
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

    def step(self):
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
