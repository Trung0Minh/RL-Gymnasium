import torch
import mujoco
import mujoco.viewer
import time
import numpy as np
import os

from env import make_env
from model import Actor

def visualize(model_path="actor.pth", stats_path="stats.npz"):
    env = make_env()
    
    # Load stats if they exist
    if os.path.exists(stats_path):
        env.load_stats(stats_path)
        print(f"Loaded normalization stats from {stats_path}")
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    actor = Actor(obs_dim, act_dim)
    if os.path.exists(model_path):
        actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found. Running with random weights.")

    actor.eval()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            obs = env.reset()
            done = False
            
            while not done and viewer.is_running():
                step_start = time.time()

                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    mu, _ = actor(obs_tensor)
                    action = mu.numpy()[0]

                obs, reward, done, _ = env.step(action)
                
                viewer.sync()
                time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    # Prioritize 'actor_final.pth' but fallback to 'actor.pth'
    m_path = "actor_final.pth" if os.path.exists("actor_final.pth") else "actor.pth"
    s_path = "stats_final.npz" if os.path.exists("stats_final.npz") else "stats.npz"
    visualize(m_path, s_path)
