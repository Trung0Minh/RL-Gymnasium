import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import time
import torch
import os
from agent import DPGAgent

def get_mountain_car_model():
    # Define the hill for the track geoms
    x = np.linspace(-1.2, 0.6, 100)
    y = np.sin(3 * x)
    
    geoms = ""
    for i in range(len(x) - 1):
        x_mid = (x[i] + x[i+1]) / 2
        y_mid = (y[i] + y[i+1]) / 2
        angle = np.arctan2(y[i+1] - y[i], x[i+1] - x[i])
        length = np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
        geoms += f'<geom type="box" size="{length/2} 0.2 0.01" pos="{x_mid} 0 {y_mid}" euler="0 {-angle} 0" rgba="0.2 0.6 0.2 1"/>\n'

    # Load base XML
    xml_path = "mountain_car.xml"
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Could not find {xml_path}")
        
    with open(xml_path, "r") as f:
        xml_content = f.read()
    
    # Inject geoms into the track body
    # We look for the track body tag and insert geoms inside it
    track_body_tag = '<body name="track" pos="0 0 0">'
    xml_content = xml_content.replace(track_body_tag, track_body_tag + "\n" + geoms)
    
    return mujoco.MjModel.from_xml_string(xml_content)

def visualize():
    env = gym.make("MountainCarContinuous-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    agent = DPGAgent(state_size=state_size, action_size=action_size, seed=0)
    try:
        agent.actor_local.load_checkpoint('model_weight/dpg_actor_checkpoint.pth')
        print("Successfully loaded model weights.")
    except:
        try:
            agent.actor_local.load_checkpoint('dpg_actor_checkpoint.pth')
            print("Successfully loaded model weights from root.")
        except:
            print("Model weights not found. Using untrained agent.")

    model = get_mountain_car_model()
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for ep in range(5):
            state, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = agent.act(state, add_noise=False)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                # Update MuJoCo state
                pos = state[0]
                y = np.sin(3 * pos)
                angle = np.arctan2(3 * np.cos(3 * pos), 1)
                
                # Set car position
                data.qpos[0] = pos
                data.qpos[1] = 0
                data.qpos[2] = y + 0.05
                
                # Set car orientation (quaternion for rotation around Y axis)
                data.qpos[3] = np.cos(-angle/2)
                data.qpos[4] = 0
                data.qpos[5] = np.sin(-angle/2)
                data.qpos[6] = 0
                
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.02)
                
                if not viewer.is_running():
                    break
            
            print(f"Episode {ep+1} finished. Reward: {total_reward:.2f}")
            if not viewer.is_running():
                break
    
    env.close()

if __name__ == "__main__":
    visualize()
