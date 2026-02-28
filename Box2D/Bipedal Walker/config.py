# Hyperparameters
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
BATCH_SIZE = 256
HIDDEN_DIM = 256
REPLAY_SIZE = 1000000
START_STEPS = 10000
UPDATE_AFTER = 1000

# Speed Optimizations
NUM_ENVS = 8          # Number of parallel environments
UPDATE_EVERY = 50     # Update every N steps
UPDATE_ITERS = 50     # Number of gradient steps to perform

# TD3 Specific
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
EXPL_NOISE = 0.1
POLICY_FREQ = 2
