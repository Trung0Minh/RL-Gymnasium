# Environment Settings
ENV_NAME = "Blackjack-v1"
SAB = True  # Sutton & Barto rules

# Hyperparameters
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 1
EPSILON = 1.0
EPSILON_DECAY = 0.999998
MIN_EPSILON = 0.05

# Training parameters
EPISODES = 2000000
MODELS_DIR = "models"
PLOT_PATH = "training_rewards.png"
