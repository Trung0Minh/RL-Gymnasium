import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization is highly recommended for PPO.
    It helps preserve the variance of activations in deep networks.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()
        
        # The Critic (Value Function): Evaluates how good a state is.
        # Outputs a single scalar value.
        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # The Actor: Decides which action to take.
        # Outputs the MEAN of the action distribution.
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_actions), std=0.01),
        )
        
        # The STANDARD DEVIATION of the action distribution.
        # In standard continuous PPO, this is a standalone parameter, not dependent on the state.
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_actions))

    def get_value(self, x):
        """Returns the Critic's evaluation of the state."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Samples an action from the Gaussian distribution defined by the Actor,
        and returns the action, its log probability, entropy, and the Critic's value.
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Create a Normal (Gaussian) distribution
        probs = Normal(action_mean, action_std)
        
        # If no action is provided (during Rollout), sample one
        if action is None:
            action = probs.sample()
            
        # Sum the log probabilities across the action dimensions (6 joints for HalfCheetah)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
