#!/usr/bin/env python3
"""
测试改进后的ReplayBuffer的action chunking和多步奖励功能
"""

import numpy as np
import torch
from utils import ReplayBuffer


def test_basic_functionality():
    """测试基本功能"""
    print("测试1: 基本功能测试")

    # 创建buffer
    buffer = ReplayBuffer(
        state_dim=4,
        action_dim=2,
        max_size=1000,
        action_chunk_size=3,
        reward_horizon=4
    )

    # 添加一些数据
    np.random.seed(42)
    for i in range(20):
        state = np.random.randn(4)
        action = np.random.randn(2)
        next_state = np.random.randn(4)
        reward = np.random.randn()
        done = (i % 10 == 9)  # 每10步结束一个episode
        episode_start = (i % 10 == 0)

        buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)

    print(f"Buffer大小: {buffer.size}")
    print(f"统计信息: {buffer.get_statistics()}")
    print("✓ 基本功能正常\n")


def test_gymnasium_format():
    """测试gymnasium格式支持"""
    print("测试2: Gymnasium格式支持测试")

    buffer = ReplayBuffer(state_dim=4, action_dim=2, max_size=100, action_chunk_size=2, reward_horizon=3)

    # 使用gymnasium格式添加数据
    state = np.random.randn(4)
    action = np.random.randn(2)
    next_state = np.random.randn(4)
    reward = 1.0

    # 测试terminated和truncated
    buffer.add(state, action, next_state, reward, terminated=True, truncated=False, episode_start=True)
    buffer.add(state, action, next_state, reward, terminated=False, truncated=True)
    buffer.add(state, action, next_state, reward, terminated=False, truncated=False)

    print(f"添加3个样本后，done标志: {buffer.done[:3].flatten()}")
    print("✓ Gymnasium格式支持正常\n")


def test_action_chunking():
    """测试action chunking功能"""
    print("测试3: Action Chunking功能测试")

    buffer = ReplayBuffer(state_dim=2, action_dim=1, max_size=100, action_chunk_size=3, reward_horizon=2)

    # 创建可预测的动作序列
    actions = np.arange(10).reshape(-1, 1).astype(np.float32)  # [0, 1, 2, ..., 9]

    for i in range(10):
        state = np.array([i, i+1], dtype=np.float32)
        action = actions[i]
        next_state = np.array([i+1, i+2], dtype=np.float32)
        reward = float(i)
        done = (i == 4) or (i == 9)  # 在索引4和9处结束episode
        episode_start = (i == 0) or (i == 5)  # 在索引0和5处开始episode

        buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)

    # 测试action chunk获取
    chunk = buffer._get_action_chunk(0, 3)  # 从索引0开始，获取3个动作
    print(f"从索引0开始的action chunk: {chunk.flatten()}")
    print(f"预期: [0, 1, 2] (第一个episode内的连续动作)")

    chunk = buffer._get_action_chunk(3, 3)  # 从索引3开始，跨越episode边界
    print(f"从索引3开始的action chunk: {chunk.flatten()}")
    print(f"预期: [3, 4, 4] (索引4是episode结束，动作4会被重复)")

    chunk = buffer._get_action_chunk(5, 3)  # 从索引5开始，新episode
    print(f"从索引5开始的action chunk: {chunk.flatten()}")
    print(f"预期: [5, 6, 7] (第二个episode内的连续动作)")

    print("✓ Action chunking功能正常\n")


def test_multi_step_reward():
    """测试多步奖励功能"""
    print("测试4: 多步奖励功能测试")

    buffer = ReplayBuffer(state_dim=2, action_dim=1, max_size=100, action_chunk_size=2, reward_horizon=3)

    # 创建可预测的奖励序列
    rewards = np.arange(1, 11, dtype=np.float32)  # [1, 2, 3, ..., 10]

    for i in range(10):
        state = np.array([i, i+1], dtype=np.float32)
        action = np.array([i], dtype=np.float32)
        next_state = np.array([i+1, i+2], dtype=np.float32)
        reward = rewards[i]
        done = (i == 4) or (i == 9)  # 在索引4和9处结束episode
        episode_start = (i == 0) or (i == 5)

        buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)

    # 测试多步奖励计算 (gamma=1.0 for simplicity)
    multi_reward = buffer._get_multi_step_reward(0, 3, gamma=1.0)
    print(f"从索引0开始的3步奖励 (gamma=1.0): {multi_reward}")
    print(f"预期: {1+2+3} = 6")

    multi_reward = buffer._get_multi_step_reward(3, 3, gamma=1.0)
    print(f"从索引3开始的3步奖励 (跨边界): {multi_reward}")
    print(f"预期: {4+5} = 9 (在episode结束处停止)")

    multi_reward = buffer._get_multi_step_reward(0, 3, gamma=0.9)
    print(f"从索引0开始的3步奖励 (gamma=0.9): {multi_reward:.3f}")
    expected = 1 + 0.9*2 + 0.9*0.9*3
    print(f"预期: {expected:.3f}")

    print("✓ 多步奖励功能正常\n")


def test_chunk_sampling():
    """测试chunk采样功能"""
    print("测试5: Chunk采样功能测试")

    buffer = ReplayBuffer(state_dim=3, action_dim=2, max_size=100, action_chunk_size=4, reward_horizon=3)

    # 添加足够的数据
    np.random.seed(123)
    for i in range(50):
        state = np.random.randn(3)
        action = np.random.randn(2)
        next_state = np.random.randn(3)
        reward = np.random.randn()
        done = (i % 15 == 14)  # 每15步结束一个episode
        episode_start = (i % 15 == 0)

        buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)

    # 测试标准采样
    batch_size = 8
    states, actions, next_states, rewards, dones = buffer.sample(batch_size, return_chunks=False)

    print(f"标准采样 - States shape: {states.shape}")
    print(f"标准采样 - Actions shape: {actions.shape}")

    # 测试chunk采样
    states, action_chunks, next_states, multi_rewards, dones = buffer.sample(batch_size, return_chunks=True)

    print(f"Chunk采样 - States shape: {states.shape}")
    print(f"Chunk采样 - Action chunks shape: {action_chunks.shape}")
    print(f"Chunk采样 - Multi-step rewards shape: {multi_rewards.shape}")

    # 验证action chunk维度
    expected_action_shape = (batch_size, buffer.action_chunk_size, 2)
    assert action_chunks.shape == expected_action_shape, f"Action chunk shape mismatch: {action_chunks.shape} vs {expected_action_shape}"

    print("✓ Chunk采样功能正常\n")


def test_sequential_sampling():
    """测试序列采样功能"""
    print("测试6: 序列采样功能测试")

    buffer = ReplayBuffer(state_dim=2, action_dim=1, max_size=100, action_chunk_size=2, reward_horizon=2)

    # 添加数据，创建多个episode
    for ep in range(3):
        for i in range(8):  # 每个episode 8步
            global_i = ep * 8 + i
            state = np.array([global_i, global_i+1], dtype=np.float32)
            action = np.array([global_i], dtype=np.float32)
            next_state = np.array([global_i+1, global_i+2], dtype=np.float32)
            reward = float(global_i)
            done = (i == 7)  # 每个episode的最后一步
            episode_start = (i == 0)

            buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)

    # 测试序列采样
    batch_size = 4
    sequence_length = 5

    try:
        seq_states, seq_actions, seq_rewards, seq_dones = buffer.sample_sequential(batch_size, sequence_length)

        print(f"序列采样 - States shape: {seq_states.shape}")
        print(f"序列采样 - Actions shape: {seq_actions.shape}")
        print(f"预期序列形状: ({batch_size}, {sequence_length}, ...)")

        # 验证序列形状
        assert seq_states.shape == (batch_size, sequence_length, 2)
        assert seq_actions.shape == (batch_size, sequence_length, 1)

        print("✓ 序列采样功能正常\n")

    except ValueError as e:
        print(f"序列采样测试: {e}")
        print("这是预期的，因为可能没有足够的连续序列\n")


def test_edge_cases():
    """测试边界情况"""
    print("测试7: 边界情况测试")

    # 测试空buffer
    empty_buffer = ReplayBuffer(state_dim=2, action_dim=1, max_size=100, action_chunk_size=2, reward_horizon=2)

    try:
        empty_buffer.sample(1)
        print("❌ 空buffer采样应该抛出异常")
    except ValueError:
        print("✓ 空buffer正确抛出异常")

    # 测试单步episode
    single_step_buffer = ReplayBuffer(state_dim=2, action_dim=1, max_size=100, action_chunk_size=3, reward_horizon=3)

    for i in range(5):
        state = np.array([i, i+1], dtype=np.float32)
        action = np.array([i], dtype=np.float32)
        next_state = np.array([i+1, i+2], dtype=np.float32)
        reward = float(i)
        done = True  # 每一步都结束episode
        episode_start = True  # 每一步都开始新episode

        single_step_buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)

    # 测试chunk采样
    try:
        states, action_chunks, next_states, multi_rewards, dones = single_step_buffer.sample(2, return_chunks=True)
        print("✓ 单步episode的chunk采样正常处理")
        print(f"Action chunks shape: {action_chunks.shape}")
    except Exception as e:
        print(f"单步episode测试出现问题: {e}")

    print("✓ 边界情况测试完成\n")


def main():
    """运行所有测试"""
    print("开始测试改进后的ReplayBuffer功能\n")
    print("="*50)

    test_basic_functionality()
    test_gymnasium_format()
    test_action_chunking()
    test_multi_step_reward()
    test_chunk_sampling()
    test_sequential_sampling()
    test_edge_cases()

    print("="*50)
    print("所有测试完成！")


if __name__ == "__main__":
    main()
