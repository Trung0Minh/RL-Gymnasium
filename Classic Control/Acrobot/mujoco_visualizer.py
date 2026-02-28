import mujoco
import mujoco.viewer
import numpy as np
import time
import gymnasium as gym
from agent import DQNAgent
from env_utils import BalancingAcrobotWrapper
import os
import argparse

class MujocoVisualizer:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
    def _get_angles(self, state):
        # state: [cos(th1), sin(th1), cos(th2), sin(th2), th1_dot, th2_dot]
        cos_th1, sin_th1, cos_th2, sin_th2 = state[0], state[1], state[2], state[3]
        th1 = np.arctan2(sin_th1, cos_th1)
        th2 = np.arctan2(sin_th2, cos_th2)
        return th1, th2

    def display(self, state):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        th1, th2 = self._get_angles(state)
        
        # Set joint positions (Acrobot-v1 angles are relative)
        # In Mujoco XML, joint1 is at base, joint2 is at link1-link2 connection
        self.data.qpos[0] = th1
        self.data.qpos[1] = th2
        
        # Forward kinematics to update geoms
        mujoco.mj_forward(self.model, self.data)
        
        # Sync viewer
        self.viewer.sync()
        
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def run_visualization(mode='full'):
    print(f"\nVisualizing Acrobot ({mode}) using Gym Physics + MuJoCo Rendering...")
    env = gym.make("Acrobot-v1", render_mode=None)
    env = BalancingAcrobotWrapper(env, mode=mode)
    
    viz = MujocoVisualizer('acrobot.xml')
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0)

    checkpoint = f'best_{mode}.pth'
    if os.path.exists(checkpoint):
        agent.qnetwork_local.load_checkpoint(checkpoint)
    else:
        print(f"Warning: Checkpoint {checkpoint} not found. Using untrained agent.")

    try:
        while True:
            state, info = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                state, reward, terminated, truncated, info = env.step(action)
                viz.display(state)
                time.sleep(1/60)
                done = terminated or truncated
                if not viz.viewer.is_running():
                    return
    except KeyboardInterrupt:
        pass
    finally:
        viz.close()
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full', choices=['single', 'full'], help="Mục tiêu: single hoặc full")
    args = parser.parse_args()

    run_visualization(args.mode)
