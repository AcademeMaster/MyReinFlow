#!/usr/bin/env python3
"""
综合训练脚本：先进行离线训练，然后进行在线训练
"""

import numpy as np
import torch
import argparse
import os
import minari

import utils
from TD3_BC import TD3_BC


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    # Load Minari dataset and recover environment
    minari_dataset = minari.load_dataset(env_name)
    eval_env = minari_dataset.recover_environment(eval_env=True)
    # Set seed for evaluation environment
    eval_env.reset(seed=seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        while not done:
            # Handle different state formats
            if isinstance(state, tuple):
                # For newer gym versions that return tuple
                state_obs = state[0] if isinstance(state, tuple) and len(state) > 0 else state
            else:
                # For direct state observation
                state_obs = state

            # Ensure state is a numpy array with correct shape
            state_array = np.asarray(state_obs, dtype=np.float32)
            if state_array.ndim == 0:
                state_array = state_array.reshape(1, -1)
            elif state_array.ndim == 1:
                state_array = state_array.reshape(1, -1)
            state_normalized = (state_array - mean) / std
            action = policy.select_action(state_normalized)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def online_train_policy(policy, env, replay_buffer, mean, std, max_timesteps, 
                       eval_freq, expl_noise, start_timesteps=10000, batch_size=256):
    """
    在线训练策略，与环境交互并学习
    """
    # 设置为在线训练模式，不使用BC损失
    policy.train_mode = "online"
    
    # 用于记录训练统计信息
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    total_timesteps = 0
    
    state, _ = env.reset()
    done = False
    
    evaluations = []
    
    for t in range(int(max_timesteps)):
        episode_timesteps += 1
        total_timesteps += 1
        
        # 处理状态格式
        if isinstance(state, tuple):
            state_obs = state[0] if isinstance(state, tuple) and len(state) > 0 else state
        else:
            state_obs = state
            
        state_array = np.asarray(state_obs, dtype=np.float32)
        if state_array.ndim == 0:
            state_array = state_array.reshape(1, -1)
        elif state_array.ndim == 1:
            state_array = state_array.reshape(1, -1)
        state_normalized = (state_array - mean) / std
        
        # 选择动作
        if t < start_timesteps:
            # 在开始阶段使用随机动作进行探索
            action = env.action_space.sample()
        else:
            # 使用策略选择动作并添加噪声
            action = (
                policy.select_action(state_normalized) + 
                np.random.normal(0, expl_noise, size=env.action_space.shape[0])
            ).clip(-policy.max_action, policy.max_action)
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 添加经验到回放缓冲区
        # 确保状态被正确归一化
        next_state_array = np.asarray(next_state[0] if isinstance(next_state, tuple) else next_state, dtype=np.float32)
        if next_state_array.ndim == 0:
            next_state_array = next_state_array.reshape(1, -1)
        elif next_state_array.ndim == 1:
            next_state_array = next_state_array.reshape(1, -1)
        next_state_normalized = (next_state_array - mean) / std
        
        replay_buffer.add(state_normalized.flatten(), action, next_state_normalized.flatten(), 
                         reward, float(done))
        
        state = next_state
        episode_reward += reward
        
        # 训练策略 (只有当缓冲区中有足够样本时才开始训练)
        if t >= start_timesteps and replay_buffer.size >= batch_size:
            policy.train(replay_buffer, batch_size)
        
        if done:
            # 重置环境
            print(f"Total T: {total_timesteps} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            
            # 定期评估策略
            if (episode_num + 1) % max(1, int(eval_freq / 1000)) == 0:
                eval_reward = eval_policy(policy, args.env, args.seed, mean, std, eval_episodes=5)
                evaluations.append(eval_reward)
                np.save(f"models/{file_name}_online_eval", evaluations)
                
                # 保存模型
                policy.save(f"./models/{file_name}_online")
            
            state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            
    return replay_buffer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="TD3_BC")  # Policy name
    parser.add_argument("--env", default="mujoco/pusher/expert-v0")  # Minari environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=2e5, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true", default=True)  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    # TD3
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--horizon", default=2, type=int)
    # TD3 + BC
    parser.add_argument("--alpha", default=2.5)
    parser.add_argument("--normalize", default=True)
    # Online training
    parser.add_argument("--start_timesteps", default=10e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--online_timesteps", default=2e5, type=int)  # Time steps for online training
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    # Replace slashes in filename to avoid path issues
    file_name = file_name.replace("/", "_")
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("models"):
        os.makedirs("models")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    # Load Minari dataset and recover environment
    minari_dataset = minari.load_dataset(args.env)
    env = minari_dataset.recover_environment()

    # Set seeds
    env.reset(seed=args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        # TD3 + BC
        "alpha": args.alpha,
        # 初始设置为离线训练模式
        "train_mode": "offline",
        "horizon": args.horizon,
    }

    # Initialize policy
    policy = TD3_BC(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, horizon=args.horizon)

    # 第一阶段：离线训练
    print("开始离线训练阶段...")
    replay_buffer.convert_minari(minari_dataset)
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    # 保存归一化参数以便后续使用
    if args.normalize:
        np.savez(f"models/{file_name}_norm", mean=mean, std=std)

    evaluations = []
    for t in range(int(args.max_timesteps)):
        # 统一训练调用，不需要区分单步和多步
        policy.train(replay_buffer, args.batch_size)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            evaluations.append(eval_policy(policy, args.env, args.seed, mean, std))
            np.save(f"models/{file_name}", evaluations)
            if args.save_model:
                policy.save(f"./models/{file_name}")

    # 保存离线训练后的模型
    if args.save_model:
        policy.save(f"./models/{file_name}_offline")
        print(f"离线训练完成，模型已保存到 ./models/{file_name}_offline")
        
    # 第二阶段：在线训练
    print("��始在线训练阶段...")
    # 重新初始化环境用于在线交互
    env = minari_dataset.recover_environment()
    env.reset(seed=args.seed + 1000)  # 使用不同的种子以获得不同的交互体验

    # 开始在线训练
    online_train_policy(policy, env, replay_buffer, mean, std,
                        args.online_timesteps, args.eval_freq, 
                        args.expl_noise * max_action,
                        args.start_timesteps, args.batch_size)

    # 保存最终模型
    if args.save_model:
        policy.save(f"./models/{file_name}_final")
        print(f"在线训练完成，最终模型已保存到 ./models/{file_name}_final")