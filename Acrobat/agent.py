import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from model import QNetwork
from replay import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

class DQNAgent:
    def __init__(self, state_size, action_size, seed, buffer_size=int(1e5), batch_size=64, gamma=0.99, lr=5e-4, update_every=4):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.update_every = update_every
        self.t_step = 0
        
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        
        self.memory = ReplayBuffer(buffer_size)
        
    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(self.action_size)
        
    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences, self.gamma)
                
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        loss = F.mse_loss(Q_expected, Q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)
        
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)             