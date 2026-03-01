import mujoco
import numpy as np

class Space:
    def __init__(self, shape):
        self.shape = shape

class HumanoidEnv:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu

        obs_dim = len(self._get_obs())
        self.observation_space = Space((obs_dim,))
        self.action_space = Space((self.nu,))

        # For observation normalization
        self.obs_mean = np.zeros(obs_dim)
        self.obs_std = np.ones(obs_dim)
        self.count = 1e-4

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        obs = self._get_obs()
        return self._normalize_obs(obs)
    
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        
        # Update normalization stats
        self.obs_mean = 0.999 * self.obs_mean + 0.001 * obs
        self.obs_std = 0.999 * self.obs_std + 0.001 * np.abs(obs - self.obs_mean)

        # Rewards for Standing Still
        height = self.data.qpos[2]
        target_height = 1.3
        
        # 1. Height Reward: Encourage staying at the target height (exp kernel)
        height_reward = 2.0 * np.exp(-5.0 * np.square(height - target_height))
        
        # 2. Velocity Penalty: Strongly penalize movement
        vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        vel_penalty = 0.5 * np.sum(np.square(vel))
        ang_vel_penalty = 0.1 * np.sum(np.square(ang_vel))
        
        # 3. Uprightness: Torso z-axis should be vertical
        # xmat is (nbody, 9), body 1 is torso. xmat[1, 8] is the z-component of the local z-axis.
        torso_up = self.data.xmat[1, 8]
        upright_reward = 1.0 * (torso_up > 0.9) # Binary or continuous? Let's use continuous-ish
        upright_reward = 1.0 * np.clip(torso_up, 0, 1)

        # 4. Control Cost: Encourage efficiency
        control_cost = 0.01 * np.sum(np.square(action))
        
        # 5. Healthy Reward
        healthy_reward = 1.0
        
        reward = height_reward + upright_reward + healthy_reward - vel_penalty - ang_vel_penalty - control_cost
        
        # Done if torso falls below 0.8m or is too far from upright
        done = height < 0.8 or torso_up < 0.7

        return self._normalize_obs(obs), reward, done, {}
    
    def _normalize_obs(self, obs):
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    def save_stats(self, path):
        np.savez(path, mean=self.obs_mean, std=self.obs_std)

    def load_stats(self, path):
        stats = np.load(path)
        self.obs_mean = stats['mean']
        self.obs_std = stats['std']

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[2:], # Exclude x, y
            self.data.qvel
        ])

def make_env():
    return HumanoidEnv("humanoid.xml")