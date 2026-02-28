# Mountain Car Continuous/config.py

# Replay buffer size
BUFFER_SIZE = int(1e6)
# Minibatch size
BATCH_SIZE = 64
# Discount factor
GAMMA = 0.99
# Soft update parameter
TAU = 1e-3
# Learning rate for actor
LR_ACTOR = 1e-4
# Learning rate for critic
LR_CRITIC = 1e-3
# How often to update the network
UPDATE_EVERY = 4
# Noise sigma
NOISE_SIGMA = 0.5
# Noise decay
NOISE_DECAY = 0.999

# Maximum number of training episodes
N_EPISODES = 2000
# Maximum number of timesteps per episode
MAX_T = 500
# Print interval
PRINT_EVERY = 100
# Random seed
SEED = 1
