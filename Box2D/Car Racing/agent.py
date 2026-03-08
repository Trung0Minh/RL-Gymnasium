import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class SACAgent:
    def __init__(self, actor, q1, q2, q1_target, q2_target, config, device, is_discrete=False):
        self.config = config
        self.device = device
        self.is_discrete = is_discrete

        self.actor = actor
        self.q1 = q1
        self.q2 = q2
        self.q1_target = q1_target
        self.q2_target = q2_target

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=config.lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=config.lr)

        # Automatic Entropy Tuning
        if self.is_discrete:
            self.target_entropy = -0.98 * np.log(1.0 / 5) # n_actions=5
        else:
            self.target_entropy = -3.0 # n_actions=3
            
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr)
        self.alpha = config.alpha

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) == 3: # single env
            state = state.unsqueeze(0)
        with torch.no_grad():
            if self.is_discrete:
                probs = self.actor(state)
                action = probs.argmax(dim=-1).cpu().numpy()
            else:
                action, _, _ = self.actor.sample(state)
                action = action.cpu().numpy()
        return action

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.alpha = self.log_alpha.exp()

    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        if self.is_discrete:
            action = action.long().unsqueeze(1)
            self._update_discrete(state, action, reward, next_state, done)
        else:
            self._update_continuous(state, action, reward, next_state, done)

        # Soft Target Update
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau)

    def _update_continuous(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            q1_next_target = self.q1_target(next_state, next_action)
            q2_next_target = self.q2_target(next_state, next_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob
            next_q_value = reward + (1 - done) * self.config.gamma * min_q_next_target

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
            
            next_v = (next_probs * (min_q_next_target - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            next_q_value = reward + (1 - done) * self.config.gamma * next_v

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
