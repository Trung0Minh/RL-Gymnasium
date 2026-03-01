# Environment Settings
ENV_NAME = 'FrozenLake-v1'
IS_SLIPPERY = True
MAP_NAME = "4x4"
RENDER_MODE_TEST = "human"

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.9999
MIN_EPSILON = 0.01

# Training parameters
EPISODES = 200000
MODELS_DIR = "models"
PLOT_PATH = "training_rewards.png"
