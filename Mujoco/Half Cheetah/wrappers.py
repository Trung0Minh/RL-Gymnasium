import gymnasium as gym

class PosturePenaltyWrapper(gym.Wrapper):
    """
    Penalizes the cheetah for being upside down or extreme torso angles.
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        torso_angle = obs[1]
        if abs(torso_angle) > 1.0:
            reward -= 0.5 
        return obs, reward, terminated, truncated, info
