import numpy as np
import random
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        
        state = torch.FloatTensor(np.stack(state)).to(self.device)
        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        action = torch.FloatTensor(np.stack(action)).to(self.device)
        reward = torch.FloatTensor(np.stack(reward)).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.stack(done)).unsqueeze(1).to(self.device)
        
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
