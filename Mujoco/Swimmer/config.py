from dataclasses import dataclass

@dataclass
class PPOConfig:
    env_id: str = "Swimmer-v5"
    total_timesteps: int = 2000000
    num_envs: int = 8
    num_steps: int = 2048
    update_epochs: int = 10
    mini_batch_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    num_updates: int = 1000
