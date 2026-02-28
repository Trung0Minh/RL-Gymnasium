import gymnasium as gym

# Environment configurations
ENV_NAME = "CliffWalking-v1"

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# Training parameters
TOTAL_EPISODES = 500
MAX_STEPS_PER_EPISODE = 100

# Saving configurations
MODEL_FILENAME = "q_table.npy"
