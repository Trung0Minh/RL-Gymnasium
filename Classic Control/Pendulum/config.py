from dataclasses import dataclass

@dataclass
class TD3Config:
    env_id: str = "Pendulum-v1"
    num_episodes: int = 500
    max_t: int = 200
    buffer_size: int = 100000
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 1e-3
    lr: float = 5e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    hidden_dim: int = 256
    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    seed: int = 1
    num_envs: int = 1
    start_timesteps: int = 1000
