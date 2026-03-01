import gymnasium as gym
import numpy as np

class ImageEnv(gym.ObservationWrapper):
    """
    Preprocessing: Grayscale (using NumPy), Resize, Crop the UI.
    CarRacing-v3 observations are (96, 96, 3).
    """
    def __init__(self, env):
        super(ImageEnv, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        # Weighted average for grayscale
        gray_obs = (obs[..., 0] * 0.2989 + obs[..., 1] * 0.5870 + obs[..., 2] * 0.1140).astype(np.uint8)
        # Crop to 84x84 (avoiding the bottom status bar)
        return gray_obs[0:84, 6:90]

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super(SkipFrame, self).__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        obs = None
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(k, *shp), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.k
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=0)

def wrap_env(env_id, continuous=True, k=4, skip=4, max_episode_steps=None):
    env = gym.make(env_id, continuous=continuous, render_mode="rgb_array", max_episode_steps=max_episode_steps)
    env = SkipFrame(env, skip)
    env = ImageEnv(env)
    env = FrameStack(env, k)
    return env

def make_env(env_id, continuous=True, k=4, skip=4, seed=None, max_episode_steps=None):
    def thunk():
        env = wrap_env(env_id, continuous, k, skip, max_episode_steps=max_episode_steps)
        if seed is not None:
            env.action_space.seed(seed)
        return env
    return thunk
