import random
import numpy as np

# Ornstein-Uhlenbeck Noise
class OUNoise:
    def __init__(self, size, seed, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma 
        self.seed = random.seed(seed)
        self.reset()
        
    def reset(self):
        self.state = self.mu
        
    def sample(self):
        x = self.state
        dw = np.random.normal(size=self.mu.shape)
        dx = self.theta * (self.mu - x) + self.sigma * dw
        self.state = x + dx
        return self.state