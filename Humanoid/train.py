import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

from env import make_env
from model import Actor, Critic
from buffer import RolloutBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Plotting
plt.ion() # Interactive mode ON
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')
ax.set_xlabel('Updates')
ax.set_ylabel('Avg Episode Reward')
ax.set_title('Humanoid Training Progress')
plt.show(block=False)
plt.pause(0.1) # Small pause to allow window to render

def collect_rollout(env, actor, critic, buffer, steps):
    obs = env.reset()
    episode_rewards = []
    curr_rew = 0
    
    # Move models to CPU for collection to avoid GPU transfer overhead
    actor_cpu = actor.to("cpu")
    critic_cpu = critic.to("cpu")
    
    for _ in range(steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action, logp = actor_cpu.get_action(obs_tensor)
            value = critic_cpu(obs_tensor)

        action_np = action.numpy()[0]
        next_obs, reward, done, _ = env.step(action_np)
        curr_rew += reward

        buffer.store(obs, action_np, reward, logp.item(), value.item(), done)

        if done:
            episode_rewards.append(curr_rew)
            curr_rew = 0
            obs = env.reset()
        else:
            obs = next_obs
    
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        last_value = critic_cpu(obs_tensor).item()
    
    buffer.compute_advantages(last_value)
    
    # Move back to original device for training
    actor.to(device)
    critic.to(device)
    
    return np.mean(episode_rewards) if len(episode_rewards) > 0 else None

def train(resume=False):
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim).to(device)

    if resume and os.path.exists("actor.pth"):
        actor.load_state_dict(torch.load("actor.pth"))
        if os.path.exists("stats.npz"):
            env.load_stats("stats.npz")
        print("Resumed from checkpoint")

    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

    steps_per_epoch = 4096
    batch_size = 128
    buffer = RolloutBuffer(obs_dim, act_dim, steps_per_epoch)
    
    epochs = 5000
    train_iters = 10
    eps = 0.2
    ent_coef = 0.01

    history_rewards = []
    updates = []

    print(f"Training on {device}")

    for update in range(epochs):
        avg_reward = collect_rollout(env, actor, critic, buffer, steps_per_epoch)
        
        if avg_reward is not None:
            history_rewards.append(avg_reward)
            updates.append(update)

        data = buffer.get()
        obs = data['obs'].to(device)
        actions = data['act'].to(device)
        old_logp = data['logp'].to(device)
        advantages = data['adv'].to(device)
        returns = data['ret'].to(device)

        indices = np.arange(steps_per_epoch)
        for _ in range(train_iters):
            np.random.shuffle(indices)
            for start in range(0, steps_per_epoch, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                dist = actor.get_dist(obs[mb_idx])
                new_logp = dist.log_prob(actions[mb_idx]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                
                ratio = torch.exp(new_logp - old_logp[mb_idx])
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages[mb_idx]
                
                actor_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy
                actor_opt.zero_grad(); actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5); actor_opt.step()

                values = critic(obs[mb_idx]).squeeze()
                critic_loss = ((values - returns[mb_idx]) ** 2).mean()
                critic_opt.zero_grad(); critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5); critic_opt.step()

        # Update Plot less frequently
        if len(updates) > 0 and update % 5 == 0:
            line.set_xdata(updates)
            line.set_ydata(history_rewards)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.001) 

        if update % 10 == 0:
            rew_str = f"{avg_reward:.2f}" if avg_reward is not None else "N/A"
            print(f"Update {update} | Reward {rew_str} | Actor Loss {actor_loss.item():.3f} | Critic Loss {critic_loss.item():.3f}")
            torch.save(actor.state_dict(), "actor.pth")
            env.save_stats("stats.npz")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    import sys
    res = "--resume" in sys.argv
    train(resume=res)
