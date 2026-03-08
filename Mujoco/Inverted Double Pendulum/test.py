import gymnasium as gym
import torch
import numpy as np
import argparse
from model import ActorCritic

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--render_mode", type=str, default="human")
    args = parser.parse_args()

    env_id = "InvertedDoublePendulum-v5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make(env_id, render_mode=args.render_mode)
    env = gym.wrappers.ClipAction(env)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    model = ActorCritic(obs_dim, action_dim).to(device)
    model.load_state_dict(torch.load("checkpoints/inverted_double_pendulum_ppo.pt", map_location=device, weights_only=False))
    model.eval()

    # Load normalization stats
    stats = torch.load("checkpoints/inverted_double_pendulum_obs_rms.pt", map_location=device, weights_only=False)
    mean = stats["mean"]
    var = stats["var"]

    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        done = False
        episodic_return = 0
        while not done:
            # Normalize observation
            obs = (obs - mean) / np.sqrt(var + 1e-8)
            obs = np.clip(obs, -10, 10)
            
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(torch.Tensor(obs).to(device).unsqueeze(0))
            
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            episodic_return += reward
        print(f"Episode {episode+1}: {episodic_return:.2f}")
    env.close()

if __name__ == "__main__":
    test()
