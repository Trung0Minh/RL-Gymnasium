import gymnasium as gym
import random
import torch
import numpy as np
import argparse
import os
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def dqn(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, resume=False):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        resume (bool): whether to resume from checkpoint
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    best_score = -np.inf               # track best average score
    checkpoint_path = 'best_checkpoint.pth'

    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
        agent.qnetwork_target.load_state_dict(torch.load(checkpoint_path))
        eps = eps_end
        best_score = 0.0

    print(f"Starting DQN training for {n_episodes} episodes...")
    for i_episode in range(1, n_episodes+1):
        state, info = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # --- Reward Shaping ---
            x, x_dot, theta, theta_dot = next_state
            
            x_threshold = env.unwrapped.x_threshold
            theta_threshold = env.unwrapped.theta_threshold_radians
            
            r1 = (x_threshold - abs(x)) / x_threshold - 0.8
            r2 = (theta_threshold - abs(theta)) / theta_threshold - 0.5
            custom_reward = r1 + r2
            
            done = terminated or truncated
            agent.step(state, action, custom_reward, next_state, done)
            state = next_state
            score += reward 
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        current_avg_score = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, current_avg_score), end="")
        
        # Save best model
        if current_avg_score > best_score and i_episode >= 100:
            best_score = current_avg_score
            torch.save(agent.qnetwork_local.state_dict(), checkpoint_path)

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, current_avg_score))

    return scores

def main():
    parser = argparse.ArgumentParser(description='DQN Cart Pole Training')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    # Training parameters
    parser.add_argument('--n_episodes', type=int, default=config.N_EPISODES, help='Number of episodes')
    parser.add_argument('--max_t', type=int, default=config.MAX_T, help='Max timesteps per episode')
    parser.add_argument('--eps_start', type=float, default=config.EPS_START, help='Starting epsilon')
    parser.add_argument('--eps_end', type=float, default=config.EPS_END, help='Minimum epsilon')
    parser.add_argument('--eps_decay', type=float, default=config.EPS_DECAY, help='Epsilon decay rate')
    
    # Agent parameters
    parser.add_argument('--buffer_size', type=int, default=config.BUFFER_SIZE, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--gamma', type=float, default=config.GAMMA, help='Discount factor')
    parser.add_argument('--tau', type=float, default=config.TAU, help='Soft update parameter')
    parser.add_argument('--lr', type=float, default=config.LR, help='Learning rate')
    parser.add_argument('--update_every', type=int, default=config.UPDATE_EVERY, help='Update frequency')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed')
    
    args = parser.parse_args()

    env = gym.make('CartPole-v1', max_episode_steps=args.max_t)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = Agent(state_size=state_size, action_size=action_size, seed=args.seed,
                  buffer_size=args.buffer_size, batch_size=args.batch_size,
                  gamma=args.gamma, tau=args.tau, lr=args.lr, update_every=args.update_every)
    
    scores = dqn(env, agent, n_episodes=args.n_episodes, max_t=args.max_t,
                 eps_start=args.eps_start, eps_end=args.eps_end, eps_decay=args.eps_decay, resume=args.resume)

    # Plot the scores
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title("DQN Training Scores for Cart Pole")
    plt.savefig('rewards.png')
    print("\nTraining plot saved as rewards.png")
    plt.show()

if __name__ == "__main__":
    main()
