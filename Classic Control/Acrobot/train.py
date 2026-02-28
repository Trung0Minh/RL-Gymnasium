import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent import DQNAgent
from env_utils import BalancingAcrobotWrapper
import argparse
import os
import config

def dqn(mode='full', n_episodes=2000, max_t=500, eps_start=1.0, eps_end=0.01, eps_decay=0.998, 
        resume=False, buffer_size=int(1e5), batch_size=64, gamma=0.99, lr=5e-4, 
        update_every=4, tau=1e-3, seed=0):
    scores = []
    scores_window = deque(maxlen=100)
    epsilon = eps_start
    best_mean_score = -np.inf
    checkpoint_path = f'best_{mode}.pth'
    
    env = gym.make('Acrobot-v1', render_mode=None)
    env = BalancingAcrobotWrapper(env, mode=mode)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed,
                     buffer_size=buffer_size, batch_size=batch_size, gamma=gamma,
                     lr=lr, update_every=update_every, tau=tau)
    
    if resume and os.path.exists(checkpoint_path):
        print(f"Tiếp tục huấn luyện từ checkpoint: {checkpoint_path}")
        agent.qnetwork_local.load_checkpoint(checkpoint_path)
        agent.qnetwork_target.load_checkpoint(checkpoint_path)
        epsilon = eps_end # Bắt đầu với epsilon thấp khi resume
        best_mean_score = 0.0 
    
    print(f"Bắt đầu huấn luyện DQN cho mục tiêu: {mode} trong {n_episodes} episodes...")
    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
            
        scores_window.append(score)
        scores.append(score)
        epsilon = max(eps_end, eps_decay*epsilon)
        
        mean_score_100 = np.mean(scores_window)
        print(f'\rEpisode {i_episode}\tAverage Score: {mean_score_100:.2f}', end="")
        
        # Chỉ lưu khi đạt kết quả tốt nhất mới
        if i_episode >= 100 and mean_score_100 > best_mean_score:
            best_mean_score = mean_score_100
            agent.qnetwork_local.save_checkpoint(checkpoint_path)

        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {mean_score_100:.2f}')
            
        if mean_score_100 >= 480.0:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {mean_score_100:.2f}')
            break
            
    env.close()
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN Acrobot Balancing')
    parser.add_argument('--mode', type=str, default='full', choices=['single', 'full'], help='Mục tiêu: single (link 2 up) hoặc full (cả 2 up)')
    parser.add_argument('--resume', action='store_true', help='Tiếp tục từ checkpoint tương ứng')
    
    # Training parameters
    parser.add_argument('--n_episodes', type=int, default=config.N_EPISODES, help='Số lượng episode')
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

    scores = dqn(mode=args.mode, n_episodes=args.n_episodes, max_t=args.max_t,
                 eps_start=args.eps_start, eps_end=args.eps_end, eps_decay=args.eps_decay,
                 resume=args.resume, buffer_size=args.buffer_size, batch_size=args.batch_size,
                 gamma=args.gamma, lr=args.lr, update_every=args.update_every,
                 tau=args.tau, seed=args.seed)

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title("DQN Training Scores for Acrobot Balancing")
    plt.savefig('rewards.png')
    print("\nTraining plot saved as rewards.png")
    plt.show()
