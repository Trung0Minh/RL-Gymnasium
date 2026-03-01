import argparse
import torch
import gymnasium as gym
from wrappers import wrap_env
from model import SACActor, SACDiscreteActor

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["continuous", "discrete"], default="continuous")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = f"weights/best_sac_{args.mode}.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Testing SAC in {args.mode} mode using checkpoint {args.checkpoint}")

    is_continuous = (args.mode == "continuous")
    env = gym.make("CarRacing-v3", continuous=is_continuous, render_mode="human")
    from wrappers import ImageEnv, FrameStack
    env = ImageEnv(env)
    env = FrameStack(env, k=4)

    if is_continuous:
        model = SACActor(n_stack=4, n_actions=3).to(device)
    else:
        model = SACDiscreteActor(n_stack=4, n_actions=5).to(device)

    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        print(f"Successfully loaded checkpoint: {args.checkpoint}")
    except FileNotFoundError:
        print(f"Checkpoint {args.checkpoint} not found. Running with random weights.")

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                if is_continuous:
                    _, _, action = model.sample(state_t)
                    env_act = action.cpu().numpy()[0]
                    env_act[1] = (env_act[1] + 1) / 2
                    env_act[2] = (env_act[2] + 1) / 2
                else:
                    action, _, _ = model.sample(state_t)
                    env_act = action.item()
            
            state, reward, terminated, truncated, _ = env.step(env_act)
            done = terminated or truncated
            episode_reward += reward
            
        print(f"Test Episode {episode} | Reward: {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    test()
