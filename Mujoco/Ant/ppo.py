import torch
import torch.nn as nn
from model import ActorCritic
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, lam=0.95):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lam = lam  # GAE parameter
        
        # Initialize raw modules
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Create optimizer using raw parameters
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.log_std, 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        # Compile for speed if available (keeping references to original modules)
        self.policy_compiled = self.policy
        self.policy_old_compiled = self.policy_old
        
        if hasattr(torch, 'compile'):
            try:
                self.policy_compiled = torch.compile(self.policy)
                self.policy_old_compiled = torch.compile(self.policy_old)
            except Exception:
                pass
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            # Use compiled version for inference
            action, action_logprob, state_val = self.policy_old_compiled.act(state, deterministic)
        
        # Clip action to [-1, 1] for environment safety
        action_clipped = torch.clamp(action, -1.0, 1.0)
        
        return action_clipped.cpu().numpy(), action.cpu().numpy(), action_logprob.cpu().numpy(), state_val.cpu().numpy()

    def update(self, memory, next_state_values):
        # memory.states is [batch_0, batch_1, ..., batch_T-1] -> Shape: (T, Envs, dim)
        old_states = torch.FloatTensor(np.array(memory.states)).to(device)
        old_actions = torch.FloatTensor(np.array(memory.actions)).to(device)
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs)).to(device)
        old_state_values = torch.FloatTensor(np.array(memory.state_values)).to(device)
        rewards = torch.FloatTensor(np.array(memory.rewards)).to(device)
        is_terminals = torch.FloatTensor(np.array(memory.is_terminals)).to(device)
        next_state_values = torch.FloatTensor(next_state_values).to(device)

        T, num_envs = rewards.shape

        # Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards).to(device)
        last_gae = 0
        with torch.no_grad():
            for i in reversed(range(T)):
                mask = 1.0 - is_terminals[i]
                if i == T - 1:
                    next_value = next_state_values
                else:
                    next_value = old_state_values[i + 1]
                
                delta = rewards[i] + self.gamma * next_value * mask - old_state_values[i]
                last_gae = delta + self.gamma * self.lam * mask * last_gae
                advantages[i] = last_gae
            
        returns = advantages + old_state_values
        
        # Flatten tensors for training
        old_states = old_states.view(-1, old_states.shape[-1])
        old_actions = old_actions.view(-1, old_actions.shape[-1])
        old_logprobs = old_logprobs.view(-1)
        advantages = advantages.view(-1)
        returns = returns.view(-1)

        # Normalizing advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating using compiled version for speed
            logprobs, state_values, dist_entropy = self.policy_compiled.evaluate(old_states, old_actions)
            
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, returns) - 0.01*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Sync weights using the raw modules to avoid state_dict prefix issues
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy_old.load_state_dict(self.policy.state_dict())
