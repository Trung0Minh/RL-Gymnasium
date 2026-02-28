import gymnasium as gym
import random
import torch
import numpy as np
import argparse
import os
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train(n_episodes=4000, print_every=100, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.999,
          buffer_size=int(1e5), batch_size=128, gamma=0.99, tau=1e-3, lr=5e-4, 
          update_every=4, seed=0, resume=False):
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
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed,
                     buffer_size=buffer_size, batch_size=batch_size, gamma=gamma,
                     tau=tau, lr=lr, update_every=update_every)
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    best_score = -np.inf
    checkpoint_path = 'best_checkpoint.pth'

    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
        agent.qnetwork_target.load_state_dict(torch.load(checkpoint_path))
        eps = eps_end
        best_score = -110.0 # Approximate average score when nearly solved

    print(f"Starting DQN training for {n_episodes} episodes...")
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            # Scale velocity to be in a similar range to position for the NN
            scaled_state = state.copy()
            scaled_state[1] *= 15.0 # Scale ~0.07 to ~1.0
            
            action = agent.act(scaled_state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Guided Reward Shaping:
            position, velocity = next_state
            modified_reward = reward # -1.0
            
            # Velocity hint
            modified_reward += 10.0 * abs(velocity)
            
            # Height hint
            if position > -0.4:
                modified_reward += (position + 0.4) * 2.0
            
            # CRITICAL: Keep total step reward negative to prevent loitering
            modified_reward = min(modified_reward, -0.01)

            # 4. Large terminal bonus
            if terminated:
                modified_reward += 100.0
            
            scaled_next_state = next_state.copy()
            scaled_next_state[1] *= 15.0
            
            agent.step(scaled_state, action, modified_reward, scaled_next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) 
        
        mean_score_100 = np.mean(scores_window)
        print(f'\rEpisode {i_episode}\tAverage Score: {mean_score_100:.2f}', end="")
        
        if i_episode % print_every == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {mean_score_100:.2f}')
        
        # Save only the best model based on 100-episode average
        if mean_score_100 > best_score and i_episode >= 100:
            best_score = mean_score_100
            agent.qnetwork_local.save_checkpoint(checkpoint_path)

        if mean_score_100 >= -110.0:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {mean_score_100:.2f}')
            break

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN Mountain Car Training')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    # Training parameters
    parser.add_argument('--n_episodes', type=int, default=config.N_EPISODES, help='Number of episodes')
    parser.add_argument('--max_t', type=int, default=config.MAX_T, help='Max timesteps per episode')
    parser.add_argument('--eps_start', type=float, default=config.EPS_START, help='Starting epsilon')
    parser.add_argument('--eps_end', type=float, default=config.EPS_END, help='Minimum epsilon')
    parser.add_argument('--eps_decay', type=float, default=config.EPS_DECAY, help='Epsilon decay rate')
    parser.add_argument('--print_every', type=int, default=config.PRINT_EVERY, help='Print interval')
    
    # Agent parameters
    parser.add_argument('--buffer_size', type=int, default=config.BUFFER_SIZE, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--gamma', type=float, default=config.GAMMA, help='Discount factor')
    parser.add_argument('--tau', type=float, default=config.TAU, help='Soft update parameter')
    parser.add_argument('--lr', type=float, default=config.LR, help='Learning rate')
    parser.add_argument('--update_every', type=int, default=config.UPDATE_EVERY, help='Update frequency')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed')
    
    args = parser.parse_args()

    scores = train(n_episodes=args.n_episodes, print_every=args.print_every, max_t=args.max_t,
                   eps_start=args.eps_start, eps_end=args.eps_end, eps_decay=args.eps_decay,
                   buffer_size=args.buffer_size, batch_size=args.batch_size, gamma=args.gamma,
                   tau=args.tau, lr=args.lr, update_every=args.update_every, seed=args.seed,
                   resume=args.resume)

    # plot the scores
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title("DQN Training Scores for Mountain Car")
    plt.savefig('rewards.png')
    print("\nTraining plot saved as rewards.png")
    plt.show()
