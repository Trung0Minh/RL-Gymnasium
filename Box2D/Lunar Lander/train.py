import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agent import Agent
import config
import argparse
import os

def dqn(args):
    """Deep Q-Learning.
    
    Params
    ======
        args: argparse namespace with all hyperparameters
    """
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_size if hasattr(env, 'action_size') else env.action_space.n
    
    # Initialize agent with potential CLI overrides
    agent = Agent(state_size=state_size, action_size=action_size, seed=0,
                  buffer_size=args.buffer_size,
                  batch_size=args.batch_size,
                  gamma=args.gamma,
                  tau=args.tau,
                  lr=args.lr,
                  update_every=args.update_every)
    
    print(f"Using device: {agent.device}")

    # Load checkpoint if requested
    if args.load and os.path.isfile(args.load):
        print(f"Loading checkpoint from {args.load}")
        agent.qnetwork_local.load_state_dict(torch.load(args.load, map_location=agent.device))
        agent.qnetwork_target.load_state_dict(agent.qnetwork_local.state_dict())

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = args.eps_start               # initialize epsilon
    best_avg_score = -np.inf           # track best average score
    
    for i_episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(args.max_t):
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
        eps = max(args.eps_end, args.eps_decay*eps) # decrease epsilon
        
        current_avg = np.mean(scores_window)
        print(f"\rEpisode {i_episode} | Reward: {score:.2f} | Running Avg: {current_avg:.2f}", end="")
        
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode} | Reward: {score:.2f} | Running Avg: {current_avg:.2f}")
            
        # Save only the best checkpoint
        if current_avg > best_avg_score and len(scores_window) >= 100:
            best_avg_score = current_avg
            os.makedirs('weights', exist_ok=True)
            torch.save(agent.qnetwork_local.state_dict(), 'weights/checkpoint.pth')
            
    env.close()
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
    parser.add_argument("--load", type=str, default=None, help="Path to checkpoint to continue training")
    args = parser.parse_args()

    scores = dqn(args)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores.png')
    print(f"\nTraining complete. Scores plot saved as 'scores.png'.")
