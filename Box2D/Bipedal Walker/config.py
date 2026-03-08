from dataclasses import dataclass

@dataclass
class TD3Config:
    env_id: str = "BipedalWalker-v3"
    total_timesteps: int = 1000000
    num_envs: int = 1
    buffer_size: int = 1000000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    hidden_dim: int = 256
    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    seed: int = 0
    max_episodes: int = 2000
    max_t: int = 1600
    start_timesteps: int = 25000 # Time steps initial random policy is used
