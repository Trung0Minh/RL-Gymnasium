import gymnasium as gym
import numpy as np

def test_env():
    env_id = "HalfCheetah-v5"
    print("\nTesting with SyncVectorEnv...")
    def make_env():
        env = gym.make(env_id, max_episode_steps=10) # Shorten for testing
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    venv = gym.vector.SyncVectorEnv([make_env])
    obs, info = venv.reset()
    for i in range(20): 
        action = venv.action_space.sample()
        obs, reward, terminated, truncated, info = venv.step(action)
        if i == 0:
            print(f"Info keys at step 0: {info.keys()}")
        if any(terminated) or any(truncated):
            print(f"Episode finished at step {i}")
            print(f"Info keys: {info.keys()}")
            if "final_info" in info:
                print(f"final_info type: {type(info['final_info'])}")
                print(f"final_info: {info['final_info']}")
                for item in info["final_info"]:
                    if item and "episode" in item:
                        print(f"Found return: {item['episode']['r']}")
            if "episode" in info:
                print(f"episode info in info: {info['episode']}")
            break

if __name__ == "__main__":
    test_env()
