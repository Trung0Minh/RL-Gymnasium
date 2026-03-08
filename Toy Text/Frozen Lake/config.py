from dataclasses import dataclass

@dataclass
class QConfig:
    env_id: str = "FrozenLake-v1"
    num_episodes: int = 200000
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.9999
    min_epsilon: float = 0.01
    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    seed: int = 1
    is_slippery: bool = True
    map_name: str = "4x4"
