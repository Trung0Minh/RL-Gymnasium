import gymnasium as gym
import torch
import numpy as np
import argparse
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent
import config

def train_ddpg(n_episodes=500, max_t=200, print_every=100, seed=2,
               buffer_size=int(1e6), batch_size=128, gamma=0.99,
               tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, weight_decay=0):
    """Deep Deterministic Policy Gradient (DDPG) Training Loop.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): interval to print average score
    """
    env = gym.make('Pendulum-v1')
    agent = Agent(state_size=3, action_size=1, random_seed=seed,
                  buffer_size=buffer_size, batch_size=batch_size, gamma=gamma,
                  tau=tau, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay)
    scores_deque = deque(maxlen=100)
    scores = []
    
    print("Starting training...")
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            
            # The Pendulum-v1 environment expects actions in range [-2, 2].
            # Our Actor network uses tanh, which outputs in range [-1, 1].
            rescaled_action = action * 2.0
            
            next_state, reward, terminated, truncated, _ = env.step(rescaled_action)
            done = terminated or truncated
            reward = float(reward)
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
                
        scores_deque.append(score)
        scores.append(score)
        agent.decay_noise()
        
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}", end="")
        
        if i_episode % print_every == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}')
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            
        if np.mean(scores_deque) >= -200.0 and i_episode >= 100:
            print('\nEnvironment solved in {:d} episodes!	Average Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
            
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG Pendulum Training')
    
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
    parser.add_argument('--weight_decay', type=float, default=config.WEIGHT_DECAY, help='L2 weight decay')
    
    args = parser.parse_args()

    scores = train_ddpg(n_episodes=args.n_episodes, max_t=args.max_t, print_every=args.print_every,
                        seed=args.seed, buffer_size=args.buffer_size, batch_size=args.batch_size,
                        gamma=args.gamma, tau=args.tau, lr_actor=args.lr_actor,
                        lr_critic=args.lr_critic, weight_decay=args.weight_decay)
    
    # Plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Reward')
    plt.xlabel('Episode #')
    plt.savefig('rewards.png')
    print("\nTraining plot saved as rewards.png")
    plt.show()
