import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class QNetwork(nn.Module):
    """Actor (Policy) Model (Dueling DQN)."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Shared feature layer
        self.fc1 = nn.Linear(state_size, fc1_units)
        
        # Value stream (V)
        self.value_fc2 = nn.Linear(fc1_units, fc2_units)
        self.value_fc3 = nn.Linear(fc2_units, 1)
        
        # Advantage stream (A)
        self.advantage_fc2 = nn.Linear(fc1_units, fc2_units)
        self.advantage_fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        
        v = F.relu(self.value_fc2(x))
        v = self.value_fc3(v)
        
        a = F.relu(self.advantage_fc2(x))
        a = self.advantage_fc3(a)
        
        # Combine V and A: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return v + (a - a.mean(dim=1, keepdim=True))

    def save_checkpoint(self, checkpoint_file):
        directory = os.path.dirname(checkpoint_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
