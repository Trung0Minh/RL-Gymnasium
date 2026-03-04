import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import ActorCritic

class PPOAgent:
    def __init__(self, num_inputs, num_actions, device):
        self.device = device
        self.network = ActorCritic(num_inputs, num_actions).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4, eps=1e-5)
        
        # PPO Hyperparameters
        self.gamma = 0.99           # Discount factor
        self.gae_lambda = 0.95      # GAE smoothing parameter
        self.clip_coef = 0.2        # PPO clipping coefficient
        self.ent_coef = 0.01        # Entropy coefficient (encourages exploration)
        self.vf_coef = 0.5          # Value function loss coefficient
        self.max_grad_norm = 0.5    # Gradient clipping threshold
        
    def compute_gae(self, rewards, values, dones, next_value, next_done):
        """
        Computes Generalized Advantage Estimation (GAE) backwards through the rollout.
        """
        num_steps = len(rewards)
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
                
            # Delta is the 1-step TD Error
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            
            # GAE recursively adds the discounted future deltas
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
        # The Returns (Target Values for the Critic) are Advantages + Value Estimates
        returns = advantages + values
        return advantages, returns

    def update(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values):
        """
        Performs the PPO Mini-batch Update step over multiple Epochs.
        """
        b_inds = np.arange(len(b_obs))
        
        update_epochs = 10
        num_minibatches = 32
        minibatch_size = len(b_obs) // num_minibatches
        
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds) # Shuffle the 2048 steps
            
            for start in range(0, len(b_obs), minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end] # Grab 64 steps

                # Evaluate the old actions using the NEW policy weights
                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                
                # Calculate the Ratio between New Policy and Old Policy probabilities
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                # Normalize advantages per minibatch (standard practice for stability)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # ---------------- Actor Loss (Clipped Objective) ----------------
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # ---------------- Critic Loss ----------------
                newvalue = newvalue.view(-1)
                # Unclipped Value Loss
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                # Clipped Value Loss (prevents the Critic from updating too aggressively)
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Total Loss
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
