#!/usr/bin/env python3
"""
可视化评估脚本，用于渲染和查看训练好的策略
"""

import argparse
import os

import minari
import numpy as np
import torch

import TD3_BC


def eval_policy_visualize(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    """
    可视化评估策略，渲染环境并显示
    
    Args:
        policy: 训练好的策略
        env_name: 环境名称
        seed: 随机种子
        mean: 状态归一化均值
        std: 状态归一化标准差
        seed_offset: 种子偏移量
        eval_episodes: 评估剧集数
    """
    # Load Minari dataset and recover environment with rendering
    minari_dataset = minari.load_dataset(env_name)
    # 恢复带渲染的环境
    eval_env = minari_dataset.recover_environment(render_mode="human")
    
    # Set seed for evaluation environment
    eval_env.reset(seed=seed + seed_offset)

    avg_reward = 0.
    for ep in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        print(f"开始第 {ep+1} 个评估剧集...")
        n_step=1
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
            action = policy.select_action(state_normalized, n_step=n_step)

            # # 通过该方法实现动态精度
            # action_tensor = torch.tensor(action, dtype=torch.float32, device=policy.actor.device).unsqueeze(0)
            # state_tensor = torch.tensor(state_normalized, dtype=torch.float32, device=policy.actor.device)
            # q_value=policy.critic.Q1(state_tensor, action_tensor)
            # num_step=torch.sigmoid(q_value+20)
            # n_step=max(1, int(num_step*20))
            # print(f"  步数: {step_count},  Q值: {q_value.item():.3f}, n_step: {n_step}")
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            
            # 可选：添加延迟以更好地观察
            # import time
            # time.sleep(0.01)
            
        avg_reward += episode_reward
        print(f"  剧集 {ep+1} 结束，步数: {step_count}, 奖励: {episode_reward:.3f}")

    avg_reward /= eval_episodes

    print("=" * 50)
    print(f"可视化评估结果 (共 {eval_episodes} 个剧集):")
    print(f"平均奖励: {avg_reward:.3f}")
    print("=" * 50)
    
    # 关闭环境
    eval_env.close()
    
    return avg_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="mujoco/pusher/expert-v0", help="Minari环境名称")
    parser.add_argument("--seed", default=0, type=int, help="随机种子")
    parser.add_argument("--checkpoint", default="", help="模型检查点文件路径")
    parser.add_argument("--episodes", default=20, type=int, help="评估剧集数")
    args = parser.parse_args()

    # Load Minari dataset and recover environment to get dimensions
    minari_dataset = minari.load_dataset(args.env)
    env = minari_dataset.recover_environment()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    
    print("---------------------------------------")
    print(f"环境: {args.env}, 种子: {args.seed}")
    print("---------------------------------------")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }

    # Initialize policy
    policy = TD3_BC.TD3_BC(**kwargs)

    # Load model
    if args.checkpoint:
        if os.path.exists(args.checkpoint + "_actor"):
            policy.load(args.checkpoint)
            print(f"成功加载模型: {args.checkpoint}")
        else:
            print(f"找不到模型文件: {args.checkpoint}")
            return
    else:
        # Try to find the latest model
        model_dir = "./models"
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith("_actor")]
            if model_files:
                # Sort by name to get the latest
                model_files.sort()
                latest_model = model_files[-1].replace("_actor", "")
                policy.load(f"{model_dir}/{latest_model}")
                print(f"加载最新模型: {latest_model}")
            else:
                print("找不到任何模型文件，使用随机策略进行评估")
        else:
            print("模型目录不存在，使用随机策略进行评估")
    
    # Load normalization stats if they exist
    results_dir = "./models"
    mean, std = 0., 1.
    if os.path.exists(results_dir):
        norm_files = [f for f in os.listdir(results_dir) if f.endswith("_norm.npz")]
        if norm_files:
            # Sort by name to get the latest
            norm_files.sort()
            latest_norm = norm_files[-1]
            norm_data = np.load(f"{results_dir}/{latest_norm}")
            mean, std = norm_data['mean'], norm_data['std']
            print(f"加载归一化参数: {latest_norm}")
        else:
            print("未找到归一化参数，使用默认值 (mean=0, std=1)")
    else:
        print("结果目录不存在，使用默认归一化参数 (mean=0, std=1)")
    
    # Run visualization
    eval_policy_visualize(
        policy, 
        args.env, 
        args.seed, 
        mean, 
        std, 
        eval_episodes=args.episodes
    )


if __name__ == "__main__":
    main()