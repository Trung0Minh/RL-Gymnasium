import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        # Learnable log_std for Gaussian distribution - keep as 1D for broadcasting
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        mu = self.net(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, obs):
        return self.net(obs)
