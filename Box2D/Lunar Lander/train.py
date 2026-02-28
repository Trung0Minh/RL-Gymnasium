import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agent import Agent
import config
import argparse

def dqn(n_episodes=config.EPISODES, max_t=config.MAX_T, eps_start=config.EPS_START, eps_end=config.EPS_END, eps_decay=config.EPS_DECAY, args=None):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agent with potential CLI overrides
    agent = Agent(state_size=state_size, action_size=action_size, seed=0,
                  buffer_size=args.buffer_size if args else config.BUFFER_SIZE,
                  batch_size=args.batch_size if args else config.BATCH_SIZE,
                  gamma=args.gamma if args else config.GAMMA,
                  tau=args.tau if args else config.TAU,
                  lr=args.lr if args else config.LR,
                  update_every=args.update_every if args else config.UPDATE_EVERY)
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    best_avg_score = -np.inf           # track best average score
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        current_avg = np.mean(scores_window)
        print(f'\rEpisode {i_episode}\tAverage Score: {current_avg:.2f}', end="")
        
        # Save only the best checkpoint
        if current_avg > best_avg_score and i_episode >= 100:
            best_avg_score = current_avg
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {current_avg:.2f}')
            
        if current_avg >= 200.0:
            print(f'\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {current_avg:.2f}')
            break
            
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=config.EPISODES)
    parser.add_argument("--max_t", type=int, default=config.MAX_T)
    parser.add_argument("--eps_start", type=float, default=config.EPS_START)
    parser.add_argument("--eps_end", type=float, default=config.EPS_END)
    parser.add_argument("--eps_decay", type=float, default=config.EPS_DECAY)
    parser.add_argument("--buffer_size", type=int, default=config.BUFFER_SIZE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--gamma", type=float, default=config.GAMMA)
    parser.add_argument("--tau", type=float, default=config.TAU)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--update_every", type=int, default=config.UPDATE_EVERY)
    args = parser.parse_args()

    scores = dqn(n_episodes=args.episodes, max_t=args.max_t, eps_start=args.eps_start, 
                 eps_end=args.eps_end, eps_decay=args.eps_decay, args=args)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores.png')
    print("\nTraining complete. Scores plot saved as 'scores.png'.")
