"""
ReplayBuffer使用示例：Action Chunking和Multi-Step Rewards
演示如何使用改进后的ReplayBuffer进行扩散模型训练和多步离线强化学习
"""

import numpy as np
import torch
from utils import ReplayBuffer

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")

    # 创建buffer，设置action chunk大小为4，奖励预测步数为5
    buffer = ReplayBuffer(
        state_dim=10,
        action_dim=3,
        max_size=1000,
        action_chunk_size=4,  # 动作块大小
        reward_horizon=5      # 多步奖励的步数
    )

    # 模拟添加数据
    episode_start = True
    for i in range(100):
        state = np.random.randn(10).astype(np.float32)
        action = np.random.randn(3).astype(np.float32)
        next_state = np.random.randn(10).astype(np.float32)
        reward = np.random.randn()
        done = (i % 20 == 19)  # 每20步结束一个episode

        buffer.add(state, action, next_state, reward, done, episode_start)
        episode_start = done  # 下一个开始标志

    print(f"Buffer size: {buffer.size}")

    # 传统采样（不使用chunks）
    states, actions, next_states, rewards, dones = buffer.sample(batch_size=16)
    print(f"传统采样 - Actions shape: {actions.shape}")  # [16, 3]
    print(f"传统采样 - Rewards shape: {rewards.shape}")  # [16, 1]

    # 使用action chunking和multi-step rewards的采样
    states, action_chunks, next_states, multi_step_rewards, dones = buffer.sample(
        batch_size=16,
        return_chunks=True,
        gamma=0.99
    )
    print(f"Chunk采样 - Action chunks shape: {action_chunks.shape}")  # [16, 4, 3]
    print(f"Chunk采样 - Multi-step rewards shape: {multi_step_rewards.shape}")  # [16, 1]

def example_diffusion_training():
    """扩散模型训练示例"""
    print("\n=== 扩散模型训练示例 ===")

    buffer = ReplayBuffer(
        state_dim=20,
        action_dim=6,
        max_size=10000,
        action_chunk_size=8,  # 扩散模型常用较长的动作序列
        reward_horizon=1      # 单步奖励就足够
    )

    # 模拟加载D4RL数据集
    fake_d4rl_data = {
        'observations': np.random.randn(5000, 20).astype(np.float32),
        'actions': np.random.randn(5000, 6).astype(np.float32),
        'next_observations': np.random.randn(5000, 20).astype(np.float32),
        'rewards': np.random.randn(5000).astype(np.float32),
        'terminals': np.random.choice([0, 1], size=5000, p=[0.95, 0.05]).astype(np.float32)
    }

    buffer.convert_D4RL(fake_d4rl_data)
    print(f"加载D4RL数据后buffer size: {buffer.size}")

    # 为扩散模型训练采样action chunks
    for epoch in range(3):
        states, action_chunks, _, _, _ = buffer.sample(
            batch_size=64,
            return_chunks=True
        )
        print(f"Epoch {epoch+1}: 采样到action chunks shape: {action_chunks.shape}")

        # 这里可以训练扩散模型
        # diffusion_model.train_step(states, action_chunks)

def example_multi_step_rl():
    """多步强化学习示例"""
    print("\n=== 多步强化学习示例 ===")

    buffer = ReplayBuffer(
        state_dim=8,
        action_dim=4,
        max_size=5000,
        action_chunk_size=1,   # 单步动作
        reward_horizon=10      # 10步奖励累积
    )

    # 添加一些数据
    episode_start = True
    for episode in range(20):
        for step in range(50):
            state = np.random.randn(8).astype(np.float32)
            action = np.random.randn(4).astype(np.float32)
            next_state = np.random.randn(8).astype(np.float32)
            reward = np.random.exponential(1.0)  # 正奖励
            done = (step == 49)

            buffer.add(state, action, next_state, reward, done, episode_start)
            episode_start = False
        episode_start = True

    # 采样多步奖励用于训练
    states, actions, next_states, multi_step_rewards, dones = buffer.sample(
        batch_size=32,
        return_chunks=True,
        gamma=0.95  # 折扣因子
    )

    print(f"单步奖励平均值: {buffer.reward[:buffer.size].mean():.3f}")
    print(f"多步奖励平均值: {multi_step_rewards.mean().item():.3f}")
    print(f"多步奖励标准差: {multi_step_rewards.std().item():.3f}")

def example_sequential_sampling():
    """序列采样示例（用于基于序列的算法）"""
    print("\n=== 序列采样示例 ===")

    buffer = ReplayBuffer(state_dim=5, action_dim=2, max_size=1000)

    # 添加连续的轨迹数据
    for _ in range(200):
        state = np.random.randn(5).astype(np.float32)
        action = np.random.randn(2).astype(np.float32)
        next_state = np.random.randn(5).astype(np.float32)
        reward = np.random.randn()
        done = False

        buffer.add(state, action, next_state, reward, done)

    # 采样序列数据
    seq_states, seq_actions, seq_rewards, seq_dones = buffer.sample_sequential(
        batch_size=8,
        sequence_length=16
    )

    print(f"序列状态shape: {seq_states.shape}")    # [8, 16, 5]
    print(f"序列动作shape: {seq_actions.shape}")   # [8, 16, 2]
    print(f"序列奖励shape: {seq_rewards.shape}")   # [8, 16, 1]

if __name__ == "__main__":
    example_basic_usage()
    example_diffusion_training()
    example_multi_step_rl()
    example_sequential_sampling()

    print("\n=== 所有示例运行完成 ===")
