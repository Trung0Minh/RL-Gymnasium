import torch
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent import DPGAgent

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # For GPU

def dpg_train(n_episodes=2000, max_t=500, print_every=100, train_seed=1):
    set_seeds(train_seed)
    scores = []
    scores_window = deque(maxlen=100)
    
    env = gym.make('MountainCarContinuous-v0', render_mode=None)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    agent = DPGAgent(state_size=state_size, action_size=action_size, seed=train_seed)
    
    print("Start trainging...")
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
    scores = dpg_train()

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title("DPG Training Scores for MountainCarContinuous-v0")
    plt.show()