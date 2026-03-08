from dataclasses import dataclass

@dataclass
class SACConfig:
    env_id: str = "CarRacing-v3"
    total_timesteps: int = 1000000
    num_envs: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    batch_size: int = 256
    buffer_size: int = 100000
    start_steps: int = 1000
    updates_per_step: int = 1
    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    seed: int = 0
    max_episodes: int = 5000
    max_t: int = 1000
    hidden_dim: int = 256
