import numpy as np
import random
from collections import defaultdict

class BlackjackAgent:
    def __init__(self, learning_rate, discount_factor, epsilon, epsilon_decay, min_epsilon):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: (player_sum, dealer_card, usable_ace) -> [q_stick, q_hit]
        self.q_values = defaultdict(lambda: np.zeros(2))

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
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
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
