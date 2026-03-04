import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared or separate backbones? For stability, often separate is better.
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim),
        )
        
        # Action standard deviation (learnable parameter)
        # Initialize to -1.0 (std ~ 0.36) for better initial stability
        self.log_std = nn.Parameter(torch.full((1, action_dim), -1.0))
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, obs):
        return self.actor(obs), self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        probs = Normal(mean, std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)

    def get_value(self, obs):
        return self.critic(obs)
