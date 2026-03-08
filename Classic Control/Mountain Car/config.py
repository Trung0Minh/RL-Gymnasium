from dataclasses import dataclass

@dataclass
class DQNConfig:
    env_id: str = "MountainCar-v0"
    num_episodes: int = 2000
    max_t: int = 1000
    buffer_size: int = 100000
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 1e-3
    lr: float = 5e-4
    update_every: int = 4
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995
    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    seed: int = 1
    num_envs: int = 1
