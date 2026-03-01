import gymnasium as gym
import torch
import numpy as np
from model import Actor
import pickle
import config
import argparse

def test(model_path="weights/td3_actor_best.pth", rms_path="weights/state_rms_best.pkl", num_episodes=5):
    env = gym.make('BipedalWalker-v3', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    actor = Actor(state_dim, action_dim, config.HIDDEN_DIM)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        actor.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found.")
        return
        
    actor.eval()
    actor.to(device)
    
    try:
        with open(rms_path, "rb") as f:
            state_rms = pickle.load(f)
        print(f"Successfully loaded RMS from {rms_path}")
    except FileNotFoundError:
        print(f"RMS file {rms_path} not found.")
        return

    def normalize_state(state):
        return np.clip((state - state_rms.mean) / np.sqrt(state_rms.var + 1e-8), -10, 10)

    for i in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            state_norm = normalize_state(state)
            state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor(state_t)
                action = action.cpu().squeeze(0).numpy()
            
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
        print(f"Test Episode {i} | Reward: {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="weights/td3_actor_best.pth")
    parser.add_argument("--rms", type=str, default="weights/state_rms_best.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    test(args.model, args.rms, args.episodes)
