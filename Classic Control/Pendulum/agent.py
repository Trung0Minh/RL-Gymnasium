import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import Actor, Critic
import copy
import random

class TD3Agent:
    def __init__(self, state_dim, action_dim, config, device):
        self.device = device
        self.config = config
        self.action_dim = action_dim
        
        self.actor = Actor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        
        self.critic = Critic(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)
        
        self.max_action = 2.0
        self.it = 0
        
    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()
        
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    def update(self, replay_buffer, iterations=1):
        for _ in range(iterations):
            self.it += 1
            state, action, next_state, reward, not_done = replay_buffer.sample(self.config.batch_size)
            
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(action) * self.config.policy_noise).clamp(-self.config.noise_clip, self.config.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
                
                # Compute the target Q value
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + not_done * self.config.gamma * target_q
                
            # Get current Q estimates
            current_q1, current_q2 = self.critic(state, action)
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Delayed policy updates
            if self.it % self.config.policy_freq == 0:
                # Compute actor loss
                pi = self.actor(state)
                q1_pi, _ = self.critic(state, pi)
                actor_loss = -q1_pi.mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
                    
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
