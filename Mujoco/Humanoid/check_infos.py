import gymnasium as gym
import numpy as np

def check_infos():
    env_id = "Humanoid-v5"
    num_envs = 4
    
    def make_env():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    envs.reset()
    
    print("Running for 200 steps...")
    for _ in range(200):
        actions = envs.action_space.sample()
        obs, rewards, terminated, truncated, infos = envs.step(actions)
        
        if "final_info" in infos:
            print("Found final_info!")
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"Episode reward: {info['episode']['r']}")
            break
        
        # Check alternative structure
        if "episode" in infos:
            print("Found episode in infos!")
            print(f"Infos keys: {infos.keys()}")
            print(f"Episode keys: {infos['episode'].keys()}")
            if "_episode" in infos:
                print(f"_episode (mask): {infos['_episode']}")
            break
            
    envs.close()

if __name__ == "__main__":
    check_infos()
