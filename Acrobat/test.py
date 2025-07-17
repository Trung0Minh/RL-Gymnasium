import gymnasium as gym
from agent import DQNAgent

print("\nChạy thử agent đã huấn luyện...")
env_test = gym.make("Acrobot-v1", render_mode="human")
agent_test = DQNAgent(state_size=env_test.observation_space.shape[0], action_size=env_test.action_space.n, seed=0)

checkpoint_file = 'checkpoint.pth'
agent_test.qnetwork_local.load_checkpoint(checkpoint_file)

test_episodes = 5
for ep in range(test_episodes):
    state, info = env_test.reset()
    done = False
    total_reward = 0
    step_count = 0
    while not done and step_count < 500:
        action = agent_test.act(state)
        state, reward, terminated, truncated, info = env_test.step(action)
        total_reward += reward
        done = terminated or truncated
        step_count += 1
        
    print(f"Test Episode {ep+1}: Total Reward = {total_reward}, Steps = {step_count}")
env_test.close()