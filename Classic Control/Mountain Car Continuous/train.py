import gymnasium as gym
import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from collections import deque
from agent import Agent
import config

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(env, agent, n_episodes=2000, max_t=500, print_every=100, resume=False):
    """
    Standardized Training Loop for Mountain Car Continuous (DPG).
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    checkpoint_path = 'model_weight/checkpoint.pth'
    
    os.makedirs('model_weight', exist_ok=True)

    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        agent.load(checkpoint_path)

    print(f"Starting training on {device} for {n_episodes} episodes...")
    
    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset()
        agent.reset_noise()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
                
        scores_window.append(score)
        scores.append(score)
        agent.decay_noise()
        
        avg_score = np.mean(scores_window)
        print(f"\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}", end="")
        
        if i_episode % print_every == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}")
            agent.save(checkpoint_path) # Periodic save (overwrite)

    return scores

def main():
    parser = argparse.ArgumentParser(description='DPG Mountain Car Continuous Training')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--n_episodes', type=int, default=config.N_EPISODES, help='Number of episodes')
    parser.add_argument('--max_t', type=int, default=config.MAX_T, help='Max timesteps per episode')
    parser.add_argument('--print_every', type=int, default=100, help='Print interval and save frequency')
    
    # Hyperparameters
    parser.add_argument('--buffer_size', type=int, default=config.BUFFER_SIZE)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--gamma', type=float, default=config.GAMMA)
    parser.add_argument('--tau', type=float, default=config.TAU)
    parser.add_argument('--lr_actor', type=float, default=config.LR_ACTOR)
    parser.add_argument('--lr_critic', type=float, default=config.LR_CRITIC)
    parser.add_argument('--noise_sigma', type=float, default=config.NOISE_SIGMA)
    parser.add_argument('--noise_decay', type=float, default=config.NOISE_DECAY)
    parser.add_argument('--seed', type=int, default=config.SEED)
    
    args = parser.parse_args()

    env = gym.make('MountainCarContinuous-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    agent = Agent(state_size=state_size, action_size=action_size, seed=args.seed,
                  buffer_size=args.buffer_size, batch_size=args.batch_size,
                  gamma=args.gamma, tau=args.tau, lr_actor=args.lr_actor, 
                  lr_critic=args.lr_critic, noise_sigma=args.noise_sigma, 
                  noise_decay=args.noise_decay)
    
    scores = train(env, agent, n_episodes=args.n_episodes, max_t=args.max_t,
                  print_every=args.print_every, resume=args.resume)

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title("DPG Training Scores for Mountain Car Continuous")
    plt.savefig('rewards.png')
    print(f"\nTraining plot saved as rewards.png")

if __name__ == "__main__":
    main()
