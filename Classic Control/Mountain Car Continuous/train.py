import torch
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import deque
from agent import DPGAgent
import config

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def dpg_train(n_episodes=2000, max_t=500, print_every=100, train_seed=1,
              buffer_size=int(1e6), batch_size=64, gamma=0.99, 
              lr_actor=1e-4, lr_critic=1e-3, tau=1e-3, update_every=4,
              noise_sigma=0.5, noise_decay=0.999):
    set_seeds(train_seed)
    scores = []
    scores_window = deque(maxlen=100)
    
    env = gym.make('MountainCarContinuous-v0', render_mode=None)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    agent = DPGAgent(state_size=state_size, action_size=action_size, seed=train_seed,
                     buffer_size=buffer_size, batch_size=batch_size, gamma=gamma,
                     lr_actor=lr_actor, lr_critic=lr_critic, tau=tau, 
                     update_every=update_every, noise_sigma=noise_sigma, 
                     noise_decay=noise_decay)
    
    print("Start training...")
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        agent.reset_noise()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break
            
        scores_window.append(score)
        scores.append(score)
        agent.decay_noise()
        
        mean_score_100 = np.mean(scores_window)
        print(f'\rEpisode {i_episode}\tAverage Score: {mean_score_100:.2f}', end="")
        
        if i_episode % print_every == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {mean_score_100:.2f}')
            
        if mean_score_100 >= 90.0:
            print(f'\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {mean_score_100:.2f}')
            agent.actor_local.save_checkpoint('dpg_actor_checkpoint.pth')
            agent.critic_local.save_checkpoint('dpg_critic_checkpoint.pth')
            break
    
    env.close()
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DPG Mountain Car Continuous Training')
    
    # Training parameters
    parser.add_argument('--n_episodes', type=int, default=config.N_EPISODES, help='Number of episodes')
    parser.add_argument('--max_t', type=int, default=config.MAX_T, help='Max timesteps per episode')
    parser.add_argument('--print_every', type=int, default=config.PRINT_EVERY, help='Print interval')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed')
    
    # Agent parameters
    parser.add_argument('--buffer_size', type=int, default=config.BUFFER_SIZE, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--gamma', type=float, default=config.GAMMA, help='Discount factor')
    parser.add_argument('--tau', type=float, default=config.TAU, help='Soft update parameter')
    parser.add_argument('--lr_actor', type=float, default=config.LR_ACTOR, help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=config.LR_CRITIC, help='Critic learning rate')
    parser.add_argument('--update_every', type=int, default=config.UPDATE_EVERY, help='Update frequency')
    parser.add_argument('--noise_sigma', type=float, default=config.NOISE_SIGMA, help='Initial noise sigma')
    parser.add_argument('--noise_decay', type=float, default=config.NOISE_DECAY, help='Noise decay rate')
    
    args = parser.parse_args()

    scores = dpg_train(n_episodes=args.n_episodes, max_t=args.max_t, print_every=args.print_every,
                       train_seed=args.seed, buffer_size=args.buffer_size, batch_size=args.batch_size,
                       gamma=args.gamma, lr_actor=args.lr_actor, lr_critic=args.lr_critic,
                       tau=args.tau, update_every=args.update_every, noise_sigma=args.noise_sigma,
                       noise_decay=args.noise_decay)

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title("DPG Training Scores for MountainCarContinuous-v0")
    plt.savefig('rewards.png')
    print("\nTraining plot saved as rewards.png")
    plt.show()
