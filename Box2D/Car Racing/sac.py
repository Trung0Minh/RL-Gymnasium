import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.stack(state), np.stack(action), np.stack(reward), np.stack(next_state), np.stack(done))

    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, actor, q1, q2, q1_target, q2_target, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, device="cpu", is_discrete=False):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.is_discrete = is_discrete

        self.actor = actor
        self.q1 = q1
        self.q2 = q2
        self.q1_target = q1_target
        self.q2_target = q2_target

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # Automatic Entropy Tuning
        if self.is_discrete:
            # target_entropy for discrete is often 0.98 * -log(1/|A|)
            self.target_entropy = -0.98 * np.log(1.0 / 5) # n_actions=5
        else:
            self.target_entropy = -3.0 # n_actions=3
            
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        if self.is_discrete:
            action = torch.LongTensor(action).unsqueeze(1).to(self.device)
            self._update_discrete(state, action, reward, next_state, done)
        else:
            action = torch.FloatTensor(action).to(self.device)
            self._update_continuous(state, action, reward, next_state, done)

        # Soft Target Update
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def _update_continuous(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            q1_next_target = self.q1_target(next_state, next_action)
            q2_next_target = self.q2_target(next_state, next_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob
            next_q_value = reward + (1 - done) * self.gamma * min_q_next_target

        q1_loss = F.mse_loss(self.q1(state, action), next_q_value)
        q2_loss = F.mse_loss(self.q2(state, action), next_q_value)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_action, log_prob, _ = self.actor.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

    def _update_discrete(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_probs = self.actor(next_state)
            next_log_probs = torch.log(next_probs + 1e-8)
            q1_next_target = self.q1_target(next_state)
            q2_next_target = self.q2_target(next_state)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            
            # V(s') = \sum \pi(a'|s') [Q(s', a') - \alpha \log \pi(a'|s')]
            next_v = (next_probs * (min_q_next_target - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            next_q_value = reward + (1 - done) * self.gamma * next_v

        curr_q1 = self.q1(state).gather(1, action)
        curr_q2 = self.q2(state).gather(1, action)
        q1_loss = F.mse_loss(curr_q1, next_q_value)
        q2_loss = F.mse_loss(curr_q2, next_q_value)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        probs = self.actor(state)
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            q1_val = self.q1(state)
            q2_val = self.q2(state)
            min_q_val = torch.min(q1_val, q2_val)
        
        # Policy loss: \sum \pi(a|s) [\alpha \log \pi(a|s) - Q(s, a)]
        actor_loss = (probs * (self.alpha * log_probs - min_q_val)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        entropy = -(probs * log_probs).sum(dim=1).detach()
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).mean())
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
