# Pendulum/config.py

# Replay buffer size
BUFFER_SIZE = int(1e6)
# Minibatch size
BATCH_SIZE = 128
# Discount factor
GAMMA = 0.99
# Soft update parameter
TAU = 1e-3
# Actor learning rate
LR_ACTOR = 1e-4
# Critic learning rate
LR_CRITIC = 1e-3
# L2 weight decay
WEIGHT_DECAY = 0

# Maximum number of training episodes
N_EPISODES = 500
# Maximum number of timesteps per episode
MAX_T = 200
# Print interval
PRINT_EVERY = 100
# Random seed
SEED = 2
