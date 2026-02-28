import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import config
from model import QNetwork, Actor
import copy

class TD3Agent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LR)
        
        self.critic = QNetwork(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LR)
        
        self.max_action = 1.0
        self.it = 0
        
    def select_action(self, state):
        # state can be (num_envs, state_dim)
        state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()
        
    def update(self, replay_buffer, iterations=1):
        for _ in range(iterations):
            self.it += 1
            state, action, next_state, reward, not_done = replay_buffer.sample(config.BATCH_SIZE)
            
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(action) * config.POLICY_NOISE).clamp(-config.NOISE_CLIP, config.NOISE_CLIP)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
                
                # Compute the target Q value
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + not_done * config.GAMMA * target_q
                
            # Get current Q estimates
            current_q1, current_q2 = self.critic(state, action)
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Delayed policy updates
            if self.it % config.POLICY_FREQ == 0:
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
                    target_param.data.copy_(config.TAU * param.data + (1 - config.TAU) * target_param.data)
                    
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(config.TAU * param.data + (1 - config.TAU) * target_param.data)
