import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

class Model(ABC):
    def __init__(self):
        super().__init__()
        self.ckpt_dir = 'model_weight/'
        os.makedirs(self.ckpt_dir, exist_ok=True)
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), self.ckpt_dir + checkpoint_file)
    
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(self.ckpt_dir + checkpoint_file))

class Actor(nn.Module, Model):
    def __init__(self, state_size, action_size, hidden_size=256):
        nn.Module.__init__(self)
        Model.__init__(self)
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
    
class Critic(nn.Module, Model):
    def __init__(self, state_size, action_size, hidden_size=256):
        nn.Module.__init__(self)
        Model.__init__(self)
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state, action):
        x_state = F.relu(self.fc1(state))
        x = torch.cat((x_state, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)