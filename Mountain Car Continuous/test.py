import gymnasium as gym
from agent import DPGAgent

print("\nChạy thử agent đã huấn luyện...")
env_test = gym.make("MountainCarContinuous-v0", render_mode="human") 
state_size_test = env_test.observation_space.shape[0]
action_size_test = env_test.action_space.shape[0]

agent_test = DPGAgent(state_size=state_size_test, action_size=action_size_test, seed=0)
try:
    agent_test.actor_local.load_checkpoint('dpg_actor_checkpoint.pth')
    agent_test.critic_local.load_checkpoint('dpg_critic_checkpoint.pth')
    print("Đã tải thành công model đã huấn luyện.")
except FileNotFoundError:
    print("Không tìm thấy model đã lưu, chạy với model khởi tạo.")

test_episodes = 5
for ep in range(test_episodes):
    state, info = env_test.reset()
    done = False
    total_reward = 0
    step_count = 0
    while not done and step_count < 500: 
        action = agent_test.act(state, add_noise=False) # remove noise for inference
        state, reward, terminated, truncated, info = env_test.step(action)
        total_reward += reward
        done = terminated or truncated
        step_count += 1
        
    print(f"Test Episode {ep+1}: Total Reward = {total_reward}, Steps = {step_count}")
env_test.close()