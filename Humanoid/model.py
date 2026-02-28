import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        self.mu = nn.Linear(256, action_dim)

        # log_std là parameter tự do
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu(x)
        # Clamp log_std to prevent it from exploding or collapsing
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std)
        return mu, std
    
    def get_dist(self, obs):
        mu, std = self.forward(obs)
        return torch.distributions.Normal(mu, std)
    
    def get_action(self, obs):
        dist = self.get_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob
    
class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        return self.net(obs)
