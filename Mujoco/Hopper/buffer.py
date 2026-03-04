import torch
import numpy as np

class PPOBuffer:
    def __init__(self, obs_dim, action_dim, size, gamma=0.99, gae_lambda=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, action_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        # Compute GAE and Rewards-to-go
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE formula: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        
        # Compute Advantages
        adv = np.zeros_like(deltas)
        last_gae = 0
        for t in reversed(range(len(deltas))):
            adv[t] = deltas[t] + self.gamma * self.gae_lambda * last_gae
            last_gae = adv[t]
        self.adv_buf[path_slice] = adv
        
        # Compute Rewards-to-go (targets for the Critic)
        ret = np.zeros_like(adv)
        last_ret = last_val
        for t in reversed(range(len(rews[:-1]))):
            ret[t] = rews[t] + self.gamma * last_ret
            last_ret = ret[t]
        self.ret_buf[path_slice] = ret
        
        self.path_start_idx = self.ptr

    def get(self):
        # Reset pointers and return the buffer as tensors
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # Advantage normalization is important for stability
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
