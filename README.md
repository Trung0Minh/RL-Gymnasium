# RL-Gymnasium

A collection of reinforcement learning solutions for various [Gymnasium](https://gymnasium.farama.org/) (formerly OpenAI Gym) environments, including Classic Control, Box2D, Toy Text, and MuJoCo.

> [!IMPORTANT]
> **Implementation Note**: All algorithms (DQN, DDPG, TD3, SAC, PPO, Q-Learning, etc.) in this repository are **implemented from scratch** using PyTorch or NumPy, without relying on high-level RL libraries like Stable Baselines3 or Ray Rllib.

## 📂 Repository Structure

The project is organized by environment categories:

```text
.
├── Box2D/                  # Continuous and discrete control tasks
│   ├── Bipedal Walker/     # TD3
│   ├── Car Racing/         # SAC (Continuous/Discrete)
│   └── Lunar Lander/       # DQN
├── Classic Control/        # Simple control benchmarks
│   ├── Acrobot/            # DQN
│   ├── Cart Pole/          # DQN
│   ├── Mountain Car/       # DQN
│   ├── Mountain Car Cont./ # DDPG
│   └── Pendulum/           # DDPG
├── Mujoco/                 # Physics-based robotics simulations
│   ├── Ant, Half Cheetah, Hopper, Humanoid, etc. (all PPO)
├── Toy Text/               # Discrete grid-world problems
│   ├── Blackjack/          # Q-Learning
│   ├── Cliff Walking/      # Q-Learning
│   ├── Frozen Lake/        # Q-Learning
│   └── Taxi/               # Q-Learning
└── requirements.txt        # Project dependencies
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Trung0Minh/RL-Gymnasium.git
   cd RL-Gymnasium
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

*Note: For MuJoCo environments, ensure you have a working installation of `mujoco` (included in `requirements.txt` for newer versions).*

## 📈 Usage

### Training an Agent

Navigate to the specific environment folder and run the `train.py` script. Most scripts support various command-line arguments to customize the training.

Example: Training Bipedal Walker with TD3
```bash
cd "Box2D/Bipedal Walker"
python train.py --num_envs 4 --episodes 2000

### Testing an Agent

After training, you can evaluate your agent using the `test.py` script.

Example: Testing Lunar Lander
```bash
cd "Box2D/Lunar Lander"
python test.py
```

## 📊 Results

Training progress (scores, average rewards) is typically saved in the `results/` or `model_weight/` directory of each environment as PNG plots and `.pth` or `.pkl` files for weights.

## 📜 License

This project is open-source and available under the MIT License.
