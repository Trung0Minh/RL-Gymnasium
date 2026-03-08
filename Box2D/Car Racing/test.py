import argparse
import torch
import gymnasium as gym
import numpy as np
from wrappers import ImageEnv, FrameStack
from model import SACActor, SACQNetwork, SACDiscreteActor, SACDiscreteQNetwork
from config import SACConfig
from agent import SACAgent

def test(cfg: SACConfig, mode: str, n_episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint_path = f"{cfg.checkpoint_dir}/car_racing_sac_{mode}.pt"
    print(f"Testing SAC in {mode} mode using checkpoint {checkpoint_path}")

    is_continuous = (mode == "continuous")
    env = gym.make(cfg.env_id, continuous=is_continuous, render_mode="human")
    env = ImageEnv(env)
    env = FrameStack(env, k=4)

    if is_continuous:
        actor = SACActor(n_stack=4, n_actions=3).to(device)
        q1 = SACQNetwork(n_stack=4, n_actions=3).to(device)
        q2 = SACQNetwork(n_stack=4, n_actions=3).to(device)
        q1_target = SACQNetwork(n_stack=4, n_actions=3).to(device)
        q2_target = SACQNetwork(n_stack=4, n_actions=3).to(device)
    else:
        actor = SACDiscreteActor(n_stack=4, n_actions=5).to(device)
        q1 = SACDiscreteQNetwork(n_stack=4, n_actions=5).to(device)
        q2 = SACDiscreteQNetwork(n_stack=4, n_actions=5).to(device)
        q1_target = SACDiscreteQNetwork(n_stack=4, n_actions=5).to(device)
        q2_target = SACDiscreteQNetwork(n_stack=4, n_actions=5).to(device)

    agent = SACAgent(actor, q1, q2, q1_target, q2_target, config=cfg, device=device, is_discrete=not is_continuous)

    try:
        agent.load(checkpoint_path)
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_path} not found. Running with random weights.")

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            if is_continuous:
                env_act = action[0]
                # Map [steer, gas, break] correctly
                env_act[1] = (env_act[1] + 1) / 2
                env_act[2] = (env_act[2] + 1) / 2
            else:
                env_act = action[0]
            
            state, reward, terminated, truncated, _ = env.step(env_act)
            done = terminated or truncated
            episode_reward += reward
            
        print(f"Test Episode {episode} | Reward: {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["continuous", "discrete"], default="continuous")
    # Dynamically build parser from SACConfig fields
    for key, value in SACConfig().__dict__.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value) if value is not None else str, default=value)
    parser.add_argument("--test_episodes", type=int, default=5)
    
    args = parser.parse_args()
    mode = args.mode
    args_dict = vars(args)
    test_eps = args_dict.pop('test_episodes')
    del args_dict['mode']
    config = SACConfig(**args_dict)
    
    test(config, mode, n_episodes=test_eps)
