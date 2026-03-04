import torch
import torch.nn as nn
from torch.optim import Adam
from actor_critic import Actor, Critic
import numpy as np

class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, clip_ratio=0.2, target_kl=0.01, train_iters=80, 
                 ent_coef=0.0, batch_size=64):
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)
        self.optimizer = Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ], eps=1e-5)
        
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_iters = train_iters
        self.ent_coef = ent_coef
        self.batch_size = batch_size

    def update(self, data):
        obs, act, adv, ret, logp_old = data['obs'], data['act'], data['adv'], data['ret'], data['logp']
        
        buffer_size = obs.shape[0]
        indices = np.arange(buffer_size)

        for i in range(self.train_iters):
            np.random.shuffle(indices)
            kl_avg = 0
            for start in range(0, buffer_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                b_obs = obs[batch_idx]
                b_act = act[batch_idx]
                b_adv = adv[batch_idx]
                b_ret = ret[batch_idx]
                b_logp_old = logp_old[batch_idx]

                # Policy update
                dist = self.actor(b_obs)
                logp = dist.log_prob(b_act).sum(axis=-1)
                ratio = torch.exp(logp - b_logp_old)
                
                # Clipped surrogate objective
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * b_adv
                loss_pi = -(torch.min(surr1, surr2)).mean()
                
                # Entropy bonus
                entropy = dist.entropy().sum(axis=-1).mean()
                loss_pi = loss_pi - self.ent_coef * entropy

                # Value function update
                val = self.critic(b_obs).squeeze(-1)
                loss_v = ((val - b_ret)**2).mean()
                
                # Total loss
                loss = loss_pi + 0.5 * loss_v
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

                with torch.no_grad():
                    kl = (b_logp_old - logp).mean().item()
                    kl_avg += kl * (len(batch_idx) / buffer_size)
            
            if kl_avg > 1.5 * self.target_kl:
                # print(f"Early stopping at iter {i} due to KL {kl_avg:.4f}")
                break
