import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import ActorCritic

class PPOAgent:
    def __init__(self, num_inputs, num_actions, device, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, hidden_dim=256):
        self.device = device
        self.network = ActorCritic(num_inputs, num_actions, hidden_dim=hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def get_action_and_value(self, x, action=None):
        return self.network.get_action_and_value(x, action)

    def get_value(self, x):
        return self.network.get_value(x)

    def update(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, update_epochs=10, mini_batch_size=64):
        b_inds = np.arange(len(b_obs))
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs), mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
