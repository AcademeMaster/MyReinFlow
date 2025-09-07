#!/usr/bin/env python3
"""
验证next_state和done的存储逻辑测试
"""

import numpy as np
from utils import ReplayBuffer


def test_storage_logic():
    """测试存储逻辑：next_state和done应该存储下一步的信息"""
    print("测试存储逻辑：next_state和done的存储机制")
    print("=" * 50)

    buffer = ReplayBuffer(
        state_dim=2,
        action_dim=1,
        max_size=100,
        horizon=3  # 3步horizon
    )

    # 创建一个清晰的episode
    print("创建episode数据:")
    print("Step | State | Action | Next_State | Reward | Done")
    print("-" * 50)

    episode_data = []
    for step in range(8):
        state = np.array([step, step + 0.1], dtype=np.float32)
        action = np.array([step], dtype=np.float32)
        next_state = np.array([step + 1, step + 1.1], dtype=np.float32)  # 下一步状态
        reward = float(step + 1)  # 奖励值
        done = (step == 7)  # 最后一步结束episode
        episode_start = (step == 0)

        episode_data.append((state, action, next_state, reward, done))

        print(f"{step:4d} | [{step:3.0f},{step+0.1:3.1f}] | [{step:3.0f}] | [{step+1:3.0f},{step+1.1:3.1f}] | {step+1:6.0f} | {done}")

        buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)

    print("\n验证存储的数据:")
    for i in range(8):
        stored_state = buffer.state[i]
        stored_action = buffer.action[i]
        stored_next_state = buffer.next_state[i]
        stored_reward = buffer.reward[i, 0]
        stored_done = buffer.done[i, 0]

        print(f"Index {i}: state=[{stored_state[0]:.0f},{stored_state[1]:.1f}], "
              f"action=[{stored_action[0]:.0f}], "
              f"next_state=[{stored_next_state[0]:.0f},{stored_next_state[1]:.1f}], "
              f"reward={stored_reward:.0f}, done={stored_done:.0f}")

    print(f"\nBuffer统计: {buffer.get_statistics()}")

    # 验证action chunking - 应该是滑动窗口
    print("\n验证Action Chunking (滑动窗口):")
    for start_idx in [0, 1, 2, 5]:
        chunk = buffer._get_action_chunk(start_idx)
        print(f"从索引{start_idx}开始的3步actions: {chunk.flatten()}")

        if start_idx <= 5:  # 可以获取完整3步
            expected = [start_idx, start_idx+1, start_idx+2]
            print(f"  预期: {expected}")
        else:
            print(f"  接近episode末尾")

    # 验证multi-step reward - 应该是累积h步奖励
    print("\n验证Multi-step Reward (累积奖励):")
    for start_idx in [0, 1, 2, 5]:
        multi_reward = buffer._get_multi_step_reward(start_idx, gamma=1.0)
        print(f"从索引{start_idx}开始的3步累积奖励: {multi_reward}")

        if start_idx <= 5:  # 可以获取完整3步
            expected = sum(range(start_idx+1, start_idx+4))  # rewards are step+1
            print(f"  预期: {expected} (奖励{start_idx+1}+{start_idx+2}+{start_idx+3})")
        else:
            remaining_steps = 8 - start_idx
            expected = sum(range(start_idx+1, start_idx+1+remaining_steps))
            print(f"  预期: {expected} (剩余{remaining_steps}步)")

    # 验证chunk采样
    print("\n验证Chunk采样:")
    states, action_chunks, next_states, multi_rewards, dones = buffer.sample(3, return_chunks=True)

    print(f"采样结果:")
    print(f"  States shape: {states.shape}")
    print(f"  Action chunks shape: {action_chunks.shape}")  # 应该是 (3, 3, 1)
    print(f"  Next states shape: {next_states.shape}")
    print(f"  Multi-step rewards shape: {multi_rewards.shape}")  # 应该是 (3, 1)
    print(f"  Dones shape: {dones.shape}")

    print("\nAction chunks详细信息:")
    for i in range(3):
        print(f"  Sample {i}: actions={action_chunks[i].flatten()}, multi_reward={multi_rewards[i].item():.1f}")


def test_episode_boundary_handling():
    """测试跨episode边界的处理"""
    print("\n\n测试Episode边界处理")
    print("=" * 50)

    buffer = ReplayBuffer(state_dim=1, action_dim=1, max_size=100, horizon=3)

    # 创建两个短episode
    print("创建两个短episode:")

    # Episode 1: 3步
    for step in range(3):
        state = np.array([step], dtype=np.float32)
        action = np.array([step], dtype=np.float32)
        next_state = np.array([step + 1], dtype=np.float32)
        reward = float(step + 1)
        done = (step == 2)
        episode_start = (step == 0)

        buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)
        print(f"Episode 1, Step {step}: action={step}, reward={step+1}, done={done}")

    # Episode 2: 4步
    for step in range(4):
        state = np.array([step + 10], dtype=np.float32)  # 用不同的值区分episode
        action = np.array([step + 10], dtype=np.float32)
        next_state = np.array([step + 11], dtype=np.float32)
        reward = float(step + 11)
        done = (step == 3)
        episode_start = (step == 0)

        buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)
        print(f"Episode 2, Step {step}: action={step+10}, reward={step+11}, done={done}")

    print(f"\nBuffer统计: {buffer.get_statistics()}")

    # 测试跨episode边界的action chunk
    print("\n测试跨Episode边界的Action Chunk:")
    for start_idx in [1, 2, 3, 4]:
        chunk = buffer._get_action_chunk(start_idx)
        multi_reward = buffer._get_multi_step_reward(start_idx, gamma=1.0)

        print(f"从索引{start_idx}开始:")
        print(f"  Action chunk: {chunk.flatten()}")
        print(f"  Multi-step reward: {multi_reward}")

        # 分析期望结果
        if start_idx == 2:  # Episode 1的最后一步
            print(f"  分析: Episode 1结束，action应该重复最后一个有效值")
        elif start_idx == 3:  # Episode 2的第一步
            print(f"  分析: Episode 2开始，不应跨越到Episode 1")


if __name__ == "__main__":
    test_storage_logic()
    test_episode_boundary_handling()