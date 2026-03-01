import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, obs_dim, action_dim, size, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.max_size = size

        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)

        self.ptr = 0

    def store(self, obs, act, rew, logp, val, done):
        if self.ptr < self.max_size:
            self.obs_buf[self.ptr] = obs
            self.act_buf[self.ptr] = act
            self.rew_buf[self.ptr] = rew
            self.logp_buf[self.ptr] = logp
            self.val_buf[self.ptr] = val
            self.done_buf[self.ptr] = done
            self.ptr += 1

    def compute_advantages(self, last_value=0):
        adv = 0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
            else:
                next_value = self.val_buf[t+1]
            next_non_terminal = 1.0 - self.done_buf[t]

            delta = self.rew_buf[t] + self.gamma * next_value * next_non_terminal - self.val_buf[t]
            adv = delta + self.gamma * self.lam * next_non_terminal * adv
            self.adv_buf[t] = adv

        self.ret_buf[:self.ptr] = self.adv_buf[:self.ptr] + self.val_buf[:self.ptr]
        
        # Normalize advantages
        actual_adv = self.adv_buf[:self.ptr]
        self.adv_buf[:self.ptr] = (actual_adv - actual_adv.mean()) / (actual_adv.std() + 1e-8)

    def get(self):
        self.ptr = 0
        return {
            "obs": torch.tensor(self.obs_buf, dtype=torch.float32),
            "act": torch.tensor(self.act_buf, dtype=torch.float32),
            "logp": torch.tensor(self.logp_buf, dtype=torch.float32),
            "val": torch.tensor(self.val_buf, dtype=torch.float32),
            "adv": torch.tensor(self.adv_buf, dtype=torch.float32),
            "ret": torch.tensor(self.ret_buf, dtype=torch.float32),
        }
