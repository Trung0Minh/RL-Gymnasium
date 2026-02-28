# Cart Pole/config.py

# Replay buffer size
BUFFER_SIZE = int(1e5)
# Minibatch size
BATCH_SIZE = 64
# Discount factor
GAMMA = 0.99
# For soft update of target parameters
TAU = 1e-3
# Learning rate
LR = 5e-4
# How often to update the network
UPDATE_EVERY = 4

# Maximum number of training episodes
N_EPISODES = 2000
# Maximum number of timesteps per episode
MAX_T = 5000
# Starting value of epsilon
EPS_START = 1.0
# Minimum value of epsilon
EPS_END = 0.01
# Multiplicative factor (per episode) for decreasing epsilon
EPS_DECAY = 0.995
# Random seed
SEED = 0
