from dataclasses import dataclass

@dataclass
class QConfig:
    env_id: str = "Blackjack-v1"
    num_episodes: int = 2000000
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.999998
    min_epsilon: float = 0.05
    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    seed: int = 1
    sab: bool = True # Simple Agent Blackjack rules
