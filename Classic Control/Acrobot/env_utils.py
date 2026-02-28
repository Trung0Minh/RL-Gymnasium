import gymnasium as gym

class BalancingAcrobotWrapper(gym.Wrapper):
    def __init__(self, env, mode='full'):
        super().__init__(env)
        self.mode = mode
        
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Acrobot state: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
        cos_theta1 = state[0]
        sin_theta1 = state[1]
        cos_theta2 = state[2]
        sin_theta2 = state[3]
        th1_dot = state[4]
        th2_dot = state[5]
        
        # Calculate height of the tip (ranges from -2.0 to 2.0)
        # y = -cos(theta1) - cos(theta1 + theta2)
        cos_theta12 = cos_theta1 * cos_theta2 - sin_theta1 * sin_theta2
        y = -cos_theta1 - cos_theta12
        
        if self.mode == 'single':
            # Target: Link 1 down (cos_theta1=1), Link 2 up (cos_theta12=-1)
            # This results in y around 0.
            target_reward = (cos_theta1 - cos_theta12) / 2.0 # Range [-1, 1], 1 is ideal
            
            # Heavy penalty for any movement to ensure stability
            vel_penalty = 0.1 * (th1_dot**2 + th2_dot**2)
            balancing_reward = target_reward - vel_penalty
            
            # Extra stability bonus when near target
            if cos_theta1 > 0.8 and cos_theta12 < -0.8:
                stability = 1.0 - (abs(th1_dot) + abs(th2_dot)) / 2.0
                balancing_reward += 0.5 * max(0, stability)

        else: # mode == 'full'
            # Target: Both links up (y = 2.0)
            # Normalize y from [-2, 2] to [0, 1]
            balancing_reward = (y + 2.0) / 4.0
            
            # Harsh penalty for not staying high enough
            if y < 1.0:
                balancing_reward -= 1.0
                # Small incentive to keep swinging (moving) when low, to avoid sticking to local optima
                balancing_reward += 0.01 * (abs(th1_dot) + abs(th2_dot))
            
            # Velocity Penalty (only when high, to encourage settling at the top)
            if y > 1.0:
                vel_penalty = 0.05 * (th1_dot**2 + th2_dot**2)
                balancing_reward -= vel_penalty
            
            # Stability Bonus: If it is actually balancing near the top
            if y > 1.7:
                stability = 1.0 - (abs(th1_dot) + abs(th2_dot)) / 4.0
                balancing_reward += 2.0 * max(0, stability)

        # Ensure reward is not extremely negative
        balancing_reward = max(-3.0, balancing_reward)
        
        # STOPS early termination
        terminated = False 
        
        return state, balancing_reward, terminated, truncated, info
