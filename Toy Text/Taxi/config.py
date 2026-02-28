# Hyperparameters for Taxi Q-learning
import numpy as np

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01

# Training parameters
TOTAL_EPISODES = 15000
MAX_STEPS_PER_EPISODE = 99

# Environment settings
ENV_NAME = "Taxi-v3"
