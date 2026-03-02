import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Actor network (Policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Added Tanh to bound the mean
        )
        # Log standard deviation
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic network (Value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, deterministic=False):
        mu = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        
        if deterministic:
            action = mu
        else:
            action = dist.sample()
            
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_val = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        mu = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
