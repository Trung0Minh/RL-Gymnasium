import gymnasium as gym
from agent import DQNAgent
from env_utils import BalancingAcrobotWrapper
import argparse
import os

def test(mode='full', test_episodes=5):
    print(f"\nTesting trained agent for objective: {mode} for {test_episodes} episodes...")
    env_test = gym.make("Acrobot-v1", render_mode="human")
    env_test = BalancingAcrobotWrapper(env_test, mode=mode)
    agent_test = DQNAgent(state_size=env_test.observation_space.shape[0], action_size=env_test.action_space.n, seed=0)

    checkpoint_file = f'model_weight/best_{mode}.pth'
    if not os.path.exists(checkpoint_file):
        print(f"Error: Couldn't find checkpoint {checkpoint_file}")
        return

    agent_test.qnetwork_local.load_checkpoint(checkpoint_file)

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
            
        print(f"Test Episode {ep+1}: Total Reward = {total_reward:.2f}, Steps = {step_count}")
    env_test.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test DQN Acrobot Balancing')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'full'], help='Objective: single or full')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to test')
    args = parser.parse_args()
    test(mode=args.mode, test_episodes=args.episodes)