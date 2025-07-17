import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent import DQNAgent

def dqn(n_episodes=2000, max_t=500, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    epsilon = eps_start
    
    env = gym.make('Acrobot-v1', render_mode=None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0)
    
    print("Bắt đầu huấn luyện DQN...")
    for i_episode in range(n_episodes):
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
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {mean_score_100:.2f}')
        if mean_score_100 >= -100.0: # Acrobot được coi là giải quyết nếu đạt trung bình -100 (tức 100 bước)
            print(f'\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {mean_score_100:.2f}')
            # Có thể lưu model tại đây: torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    
    agent.qnetwork_local.save_checkpoint('checkpoint.pth')      
    env.close()
    return scores

if __name__ == '__main__':
    scores = dqn()

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title("DQN Training Scores for Acrobot-v1")
    plt.show()
        