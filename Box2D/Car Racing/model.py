import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class CNNBase(nn.Module):
    """
    Standard Nature DQN-like CNN for 84x84 grayscale images.
    """
    def __init__(self, n_stack=4):
        super(CNNBase, self).__init__()
        self.conv1 = nn.Conv2d(n_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Conv output for 84x84: 7x7x64
        self.fc = nn.Linear(64 * 7 * 7, 512)

    def forward(self, x):
        # x is (Batch, n_stack, 84, 84)
        # Convert to float and normalize to [0, 1]
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class SACActor(nn.Module):
    def __init__(self, n_stack=4, n_actions=3):
        super(SACActor, self).__init__()
        self.base = CNNBase(n_stack)
        self.mu = nn.Linear(512, n_actions)
        self.log_std = nn.Linear(512, n_actions)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        features = self.base(x)
        mu = self.mu(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, x):
        mu, log_std = self.forward(x)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        
        # Enforcing Action Bound (already done by tanh)
        # Log-probability for the squashed Gaussian
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mu)
        
        return action, log_prob, mean

class SACQNetwork(nn.Module):
    def __init__(self, n_stack=4, n_actions=3):
        super(SACQNetwork, self).__init__()
        self.base = CNNBase(n_stack)
        self.fc_action = nn.Linear(n_actions, 512)
        self.q = nn.Linear(512, 1)

    def forward(self, x, action):
        features = self.base(x)
        combined = features + self.fc_action(action)
        return self.q(F.relu(combined))

class SACDiscreteActor(nn.Module):
    def __init__(self, n_stack=4, n_actions=5):
        super(SACDiscreteActor, self).__init__()
        self.base = CNNBase(n_stack)
        self.actor = nn.Linear(512, n_actions)

    def forward(self, x):
        features = self.base(x)
        logits = self.actor(features)
        probs = F.softmax(logits, dim=-1)
        return probs

    def sample(self, x):
        probs = self.forward(x)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = torch.log(probs + 1e-8)
        return action, log_prob, probs

class SACDiscreteQNetwork(nn.Module):
    def __init__(self, n_stack=4, n_actions=5):
        super(SACDiscreteQNetwork, self).__init__()
        self.base = CNNBase(n_stack)
        self.q = nn.Linear(512, n_actions)

    def forward(self, x):
        features = self.base(x)
        return self.q(features)
