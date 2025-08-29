#!/usr/bin/env python3
"""
独立的在线评估脚本
"""

import argparse
import collections
import minari
import numpy as np
import torch
import lightning as L
import os
import glob

from config import Config
from meanflow_ql import LitMeanFQL


def find_latest_checkpoint(checkpoint_dir="checkpoints/meanflow_ql"):
    """查找最新的检查点文件"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # 查找所有.ckpt文件
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "**", "*.ckpt"), recursive=True)
    
    if not checkpoint_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint


def evaluate_online(model: LitMeanFQL, config: Config, render_mode: str = "human"):
    """Online evaluation in the environment"""
    # 使用Minari数据集的恢复环境功能
    minari_dataset = minari.load_dataset(config.dataset_name)
    
    # 尝试使用指定的渲染模式恢复环境
    eval_env = None
    if render_mode is not None and render_mode != "none":
        try:
            eval_env = minari_dataset.recover_environment(eval_env=True, render_mode=render_mode)
        except TypeError:
            # 兼容旧版本不支持 render_mode 参数
            try:
                eval_env = minari_dataset.recover_environment(eval_env=True)
            except Exception:
                eval_env = minari_dataset.recover_environment()
    else:
        eval_env = minari_dataset.recover_environment(eval_env=True)

    total_rewards = []
    for ep in range(config.test_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        step = 0


        # 初始化动作索引
        action_idx = 0
        # 初始化动作块
        action_chunk = None

        while not done:
            # Prepare observation tensor - 注意这里直接使用观测，不需要构建序列
            obs_tensor = torch.tensor(obs).float().unsqueeze(0)  # [1, obs_dim] - 添加批次维度
            
            # 检查是否需要生成新的动作块
            if action_chunk is None or action_idx >= len(action_chunk):
                # 使用Best-of-N采样方法获取动作块
                with torch.no_grad():
                    obs_tensor = obs_tensor.to(model.device)
                    action_chunk = model(obs_tensor)  # [1, pred_horizon, action_dim]
                    action_chunk = action_chunk[0].cpu().detach().numpy()  # [pred_horizon, action_dim]
                action_idx = 0
            
            # 从动作块中获取当前动作
            action = action_chunk[action_idx]  # [action_dim]
            action_idx += 1

            next_obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            obs = next_obs
            step += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        print(f"Episode {ep + 1}: Reward = {episode_reward}")

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards) if len(total_rewards) > 1 else 0.0
    print(f"Average Reward over {config.test_episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")
    eval_env.close()
    return avg_reward


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="在线评估MeanFlow QL模型")
    parser.add_argument("--checkpoint", help="模型检查点路径（如果不指定，则使用最新的检查点）")
    parser.add_argument("--dataset", default="mujoco/pusher/expert-v0", help="Minari数据集名称")
    parser.add_argument("--test-episodes", type=int, default=20, help="测试轮数")  # 改为20以匹配config.py中的默认值
    parser.add_argument("--render", choices=["none", "human", "rgb_array"], default="human",
                        help="渲染模式 (默认: human)")
    parser.add_argument("--inference-steps", type=int, default=1, help="推理步数")
    # 添加hidden_dim和time_dim参数，以便与检查点匹配
    parser.add_argument("--hidden-dim", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--time-dim", type=int, default=64, help="时间嵌入维度")
    args = parser.parse_args()

    # 初始化配置
    config = Config(
        dataset_name=args.dataset,
        test_episodes=args.test_episodes,
        inference_steps=args.inference_steps,
        hidden_dim=args.hidden_dim,
        time_dim=args.time_dim
    )

    print("=" * 50)
    print("配置参数:")
    print(config)
    print("=" * 50)

    # Infer obs_dim and action_dim from dataset
    minari_dataset = minari.load_dataset(config.dataset_name)
    sample_episode = next(minari_dataset.iterate_episodes())
    obs_dim = sample_episode.observations.shape[-1]
    action_dim = sample_episode.actions.shape[-1]
    config.action_dim = action_dim

    # 确定要使用的检查点文件
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        # 尝试找到最新的检查点
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir=config.checkpoint_dir)
        if latest_checkpoint:
            checkpoint_path = latest_checkpoint
            print(f"使用最新检查点: {checkpoint_path}")
        else:
            print("错误: 没有找到检查点文件，请先训练模型或手动指定检查点路径")
            return

    # 加载模型
    model = LitMeanFQL.load_from_checkpoint(
        checkpoint_path, 
        obs_dim=obs_dim, 
        action_dim=action_dim,
        cfg=config
    )
    
    # 确保模型在评估模式
    model.eval()
    
    # 进行在线评估
    print("\nOnline Evaluation:")
    evaluate_online(model, config, render_mode=args.render)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision('high')
    main()