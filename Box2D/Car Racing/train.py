import os
import argparse
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from wrappers import make_env
from model import SACActor, SACQNetwork, SACDiscreteActor, SACDiscreteQNetwork
from sac import SACAgent, ReplayBuffer
import config

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["continuous", "discrete"], default="continuous")
    parser.add_argument("--episodes", type=int, default=config.EPISODES)
    parser.add_argument("--max_t", type=int, default=config.MAX_T)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--gamma", type=float, default=config.GAMMA)
    parser.add_argument("--tau", type=float, default=config.TAU)
    parser.add_argument("--alpha", type=float, default=config.ALPHA)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num_envs", type=int, default=config.NUM_ENVS)
    parser.add_argument("--capacity", type=int, default=config.CAPACITY)
    parser.add_argument("--start_steps", type=int, default=config.START_STEPS)
    parser.add_argument("--updates_per_step", type=int, default=config.UPDATES_PER_STEP)
    parser.add_argument("--load", type=str, default=None, help="Path to actor checkpoint to continue training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.set_num_threads(1)
    
    print(f"Training with SAC in {args.mode} mode using {args.num_envs} envs")

    is_continuous = (args.mode == "continuous")
    env_fns = [make_env("CarRacing-v3", continuous=is_continuous, seed=i, max_episode_steps=args.max_t) for i in range(args.num_envs)]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    
    if is_continuous:
        actor = SACActor(n_stack=4, n_actions=3).to(device)
        q1 = SACQNetwork(n_stack=4, n_actions=3).to(device)
        q2 = SACQNetwork(n_stack=4, n_actions=3).to(device)
        q1_target = SACQNetwork(n_stack=4, n_actions=3).to(device)
        q2_target = SACQNetwork(n_stack=4, n_actions=3).to(device)
    else:
        actor = SACDiscreteActor(n_stack=4, n_actions=5).to(device)
        q1 = SACDiscreteQNetwork(n_stack=4, n_actions=5).to(device)
        q2 = SACDiscreteQNetwork(n_stack=4, n_actions=5).to(device)
        q1_target = SACDiscreteQNetwork(n_stack=4, n_actions=5).to(device)
        q2_target = SACDiscreteQNetwork(n_stack=4, n_actions=5).to(device)

    # Load checkpoint if requested
    if args.load and os.path.isfile(args.load):
        print(f"Loading checkpoint from {args.load}")
        actor.load_state_dict(torch.load(args.load, map_location=device))

    agent = SACAgent(actor, q1, q2, q1_target, q2_target, lr=args.lr, gamma=args.gamma, tau=args.tau, alpha=args.alpha, device=device, is_discrete=not is_continuous)
    buffer = ReplayBuffer(capacity=args.capacity)

    scores = []
    scores_window = deque(maxlen=100)
    states, _ = envs.reset()
    episode_rewards = np.zeros(args.num_envs)
    best_reward = -float('inf')
    total_steps = 0
    finished_episodes = 0
    
    while finished_episodes < args.episodes:
        if total_steps < args.start_steps and not args.load:
            actions = envs.action_space.sample()
        else:
            states_t = torch.tensor(states, dtype=torch.float32).to(device)
            with torch.no_grad():
                if is_continuous:
                    actions_t, _, _ = actor.sample(states_t)
                    actions = actions_t.cpu().numpy()
                    actions[:, 1] = (actions[:, 1] + 1) / 2
                    actions[:, 2] = (actions[:, 2] + 1) / 2
                else:
                    actions_t, _, _ = actor.sample(states_t)
                    actions = actions_t.cpu().numpy()
        
        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
        dones = terminateds | truncateds
        
        for i in range(args.num_envs):
            real_next_state = next_states[i]
            if dones[i] and "final_observation" in infos:
                real_next_state = infos["final_observation"][i]
            
            buffer.push(states[i], actions[i], rewards[i], real_next_state, terminateds[i])
            episode_rewards[i] += rewards[i]
            
            if dones[i]:
                finished_episodes += 1
                scores_window.append(episode_rewards[i])
                scores.append(episode_rewards[i])
                
                avg_rew = np.mean(scores_window)
                print(f"\rEpisode {finished_episodes} | Reward: {episode_rewards[i]:.2f} | Running Avg: {avg_rew:.2f}", end="")
                
                if finished_episodes % 100 == 0:
                    print(f"\rEpisode {finished_episodes} | Reward: {episode_rewards[i]:.2f} | Running Avg: {avg_rew:.2f}")

                if avg_rew > best_reward and len(scores_window) >= 100:
                    best_reward = avg_rew
                    os.makedirs('weights', exist_ok=True)
                    checkpoint_path = f"weights/best_sac_{args.mode}.pth"
                    torch.save(actor.state_dict(), checkpoint_path)
                
                episode_rewards[i] = 0

        states = next_states
        total_steps += args.num_envs

        if total_steps >= args.start_steps or args.load:
            for _ in range(args.updates_per_step * args.num_envs):
                agent.update(buffer, args.batch_size)

    envs.close()

    # Plot results
    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores.png')
    print(f"\nTraining complete. Scores plot saved as 'scores.png'.")

if __name__ == "__main__":
    train()
