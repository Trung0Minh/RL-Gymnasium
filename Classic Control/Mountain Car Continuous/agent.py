import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from model import Actor, Critic
from replay import ReplayBuffer
from noise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DPGAgent:
    def __init__(self, state_size, action_size, seed, buffer_size=int(1e6), batch_size=64, 
                 gamma=0.99, lr_actor=1e-4, lr_critic=1e-3, tau=1e-3, update_every=4,
                 noise_sigma=0.5, noise_decay=0.999):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.update_every = update_every
        self.t_step = 0
        
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        
        self.optimizer_actor = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic_local.parameters(), lr=lr_critic)
        
        self.memory = ReplayBuffer(buffer_size)
        
        self.noise = OUNoise(action_size, seed, sigma=noise_sigma) # add noise to encourage exploration
        self.noise_sigma = noise_sigma
        self.noise_decay = noise_decay
        
    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample() * self.noise_sigma
        
        return np.clip(action, -1, 1)
    
    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences, self.gamma)
                
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        # update Critic Network
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions).detach() # detach to stop gradient
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        # update Actor Network
        predicted_actions = self.actor_local(states)
        actor_loss = -self.critic_local(states, predicted_actions).mean()
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        # soft update
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def reset_noise(self):
        self.noise.reset()
        
    def decay_noise(self):
        self.noise_sigma = max(self.noise_sigma * self.noise_decay, 0.01)
        self.noise.sigma = self.noise_sigma
              