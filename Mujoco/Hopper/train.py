import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from buffer import PPOBuffer
from ppo_agent import PPOAgent
from utils import RunningMeanStd

def train(env_name='Hopper-v5', steps_per_epoch=4000, epochs=500, ent_coef=0.01):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim, act_dim, ent_coef=ent_coef)
    buffer = PPOBuffer(obs_dim, act_dim, steps_per_epoch)
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    def normalize_obs(obs, update=True):
        if update:
            obs_rms.update(obs.reshape(1, -1))
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)

    obs, _ = env.reset()
    ep_ret, ep_len = 0, 0
    epoch_rets = []

    for epoch in range(epochs):
        rets = []
        for t in range(steps_per_epoch):
            # 1. Select action
            n_obs = normalize_obs(obs)
            obs_tensor = torch.as_tensor(n_obs, dtype=torch.float32)
            with torch.no_grad():
                dist = agent.actor(obs_tensor)
                act = dist.sample()
                logp = dist.log_prob(act).sum(axis=-1)
                val = agent.critic(obs_tensor)

            # 2. Step environment
            # Clip action to action space limits
            env_act = np.clip(act.numpy(), env.action_space.low, env.action_space.high)
            next_obs, rew, terminated, truncated, _ = env.step(env_act)
            done = terminated or truncated
            ep_ret += rew
            ep_len += 1

            # 3. Store transition
            buffer.store(n_obs, act.numpy(), rew, val.item(), logp.item())
            obs = next_obs

            # 4. Handle end of episode or epoch
            timeout = ep_len == env.spec.max_episode_steps
            terminal = done or timeout
            epoch_ended = (t == steps_per_epoch - 1)

            if terminal or epoch_ended:
                # If trajectory didn't reach terminal state, bootstrap value
                if timeout or epoch_ended:
                    with torch.no_grad():
                        n_obs_curr = normalize_obs(obs, update=False)
                        last_val = agent.critic(torch.as_tensor(n_obs_curr, dtype=torch.float32)).item()
                else:
                    last_val = 0
                
                buffer.finish_path(last_val)
                if terminal:
                    rets.append(ep_ret)
                    obs, _ = env.reset()
                    ep_ret, ep_len = 0, 0

        # Update policy and value function
        avg_ret = np.mean(rets) if rets else 0
        epoch_rets.append(avg_ret)
        
        # Real-time progress logging
        print(f"\rEpoch {epoch+1:3d}/{epochs} - Avg Return: {avg_ret:8.2f}", end='')
        
        # Periodic permanent summary
        if (epoch + 1) % 10 == 0:
            print(f"\rEpoch {epoch+1:3d}/{epochs} - Avg Return: {avg_ret:8.2f}")
        
        data = buffer.get()
        agent.update(data)
        
    print("Training finished!")
    
    # Plotting
    plt.plot(epoch_rets)
    plt.xlabel('Epoch')
    plt.ylabel('Avg Episode Return')
    plt.title(f'PPO Training on {env_name}')
    plt.savefig('ppo_learning_curve.png')
    plt.close()
    
    # Save the model and normalization stats
    torch.save(agent.actor.state_dict(), 'ppo_actor.pth')
    with open('obs_rms.pkl', 'wb') as f:
        pickle.dump(obs_rms, f)
    print("Model saved to ppo_actor.pth and stats to obs_rms.pkl")
    
    return agent

if __name__ == '__main__':
    train()
