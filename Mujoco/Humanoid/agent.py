import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import ActorCritic

class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def select_action(self, obs):
        # obs is now (num_envs, obs_dim)
        obs = torch.Tensor(obs).to(self.device)
        with torch.no_grad():
            action, logprob, _, value = self.network.get_action_and_value(obs)
        return action.cpu().numpy(), logprob.cpu().numpy(), value.cpu().numpy().flatten()

    def update(self, obs_b, action_b, logprob_b, reward_b, done_b, value_b, next_value, next_done):
        # All inputs are now tensors of shape (num_steps, num_envs, ...)
        num_steps = obs_b.shape[0]
        num_envs = obs_b.shape[1]
        
        obs_b = obs_b.reshape((-1,) + obs_b.shape[2:]).to(self.device)
        action_b = action_b.reshape((-1,) + action_b.shape[2:]).to(self.device)
        logprob_b = logprob_b.reshape(-1).to(self.device)
        reward_b = reward_b.to(self.device)
        done_b = done_b.to(self.device)
        value_b = value_b.to(self.device)
        
        next_value = torch.Tensor(next_value).to(self.device)
        next_done = torch.Tensor(next_done).to(self.device)

        # GAE Calculation
        advantages = torch.zeros_like(reward_b).to(self.device)
        last_gae_lam = 0
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_v = next_value
            else:
                next_non_terminal = 1.0 - done_b[t]
                next_v = value_b[t+1]
            
            delta = reward_b[t] + self.gamma * next_v * next_non_terminal - value_b[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + value_b
        
        # Flatten for PPO update
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update
        batch_size = obs_b.shape[0]
        indices = np.arange(batch_size)
        mini_batch_size = 512
        for _ in range(10): # Update epochs
            np.random.shuffle(indices)
            for start in range(0, batch_size, mini_batch_size): # Mini-batching
                end = start + mini_batch_size
                idx = indices[start:end]
                
                _, new_logprob, entropy, new_value = self.network.get_action_and_value(obs_b[idx], action_b[idx])
                
                logratio = new_logprob - logprob_b[idx]
                ratio = logratio.exp()

                pg_loss1 = -advantages[idx] * ratio
                pg_loss2 = -advantages[idx] * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_value.squeeze() - returns[idx]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
