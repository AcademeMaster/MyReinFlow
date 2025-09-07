#!/usr/bin/env python3
"""
验证多步动作和奖励的处理逻辑
"""

import numpy as np
from utils import ReplayBuffer


def test_multi_step_processing():
    """测试多步动作和奖励的处理逻辑"""
    print("测试多步动作和奖励的处理逻辑")
    print("=" * 50)

    # 创建一个ReplayBuffer，设置horizon为3
    buffer = ReplayBuffer(
        state_dim=2,
        action_dim=1,
        max_size=100,
        horizon=3
    )

    # 创建测试数据，模拟你给出的例子
    print("创建测试数据:")
    print("Step | State     | Action | Next_State | Reward | Done")
    print("-" * 55)

    # 添加6个步骤的数据
    for step in range(6):
        state = np.array([step, step + 0.1], dtype=np.float32)
        action = np.array([step], dtype=np.float32)
        next_state = np.array([step + 1, step + 1.1], dtype=np.float32)
        reward = float(step + 1)
        done = False
        episode_start = (step == 0)

        print(f"{step:4d} | [{step:2.0f},{step+0.1:2.1f}] | [{step:3.0f}] | [{step+1:2.0f},{step+1.1:2.1f}] | {step+1:6.0f} | {done}")

        buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)

    print("\n" + "=" * 50)
    print("验证多步动作和奖励:")
    
    # 检查索引0处的多步动作和奖励
    print("\n检查索引0处的多步数据:")
    action_chunk = buffer._get_action_chunk(0)
    multi_reward = buffer._get_multi_step_reward(0, gamma=1.0)  # 不折扣奖励以便验证
    
    print(f"状态: [0, 0.1]")
    print(f"动作块 (3步): {action_chunk.flatten()}")
    print(f"多步奖励 (3步): {multi_reward}")
    print(f"预期动作块: [0, 1, 2]")
    print(f"预期奖励: 6 (1+2+3)")
    
    # 检查索引1处的多步动作和奖励
    print("\n检查索引1处的多步数据:")
    action_chunk = buffer._get_action_chunk(1)
    multi_reward = buffer._get_multi_step_reward(1, gamma=1.0)
    
    print(f"状态: [1, 1.1]")
    print(f"动作块 (3步): {action_chunk.flatten()}")
    print(f"多步奖励 (3步): {multi_reward}")
    print(f"预期动作块: [1, 2, 3]")
    print(f"预期奖励: 9 (2+3+4)")
    
    # 检查索引2处的多步动作和奖励
    print("\n检查索引2处的多步数据:")
    action_chunk = buffer._get_action_chunk(2)
    multi_reward = buffer._get_multi_step_reward(2, gamma=1.0)
    
    print(f"状态: [2, 2.1]")
    print(f"动作块 (3步): {action_chunk.flatten()}")
    print(f"多步奖励 (3步): {multi_reward}")
    print(f"预期动作块: [2, 3, 4]")
    print(f"预期奖励: 12 (3+4+5)")
    
    # 验证采样功能
    print("\n" + "=" * 50)
    print("验证批量采样功能:")
    states, action_chunks, next_states, multi_rewards, dones = buffer.sample(2, return_chunks=True, gamma=1.0)
    
    print(f"采样结果形状:")
    print(f"  States: {states.shape}")
    print(f"  Action chunks: {action_chunks.shape}")
    print(f"  Next states: {next_states.shape}")
    print(f"  Multi-step rewards: {multi_rewards.shape}")
    print(f"  Dones: {dones.shape}")
    
    print(f"\n详细数据:")
    for i in range(2):
        print(f"  样本 {i}:")
        print(f"    状态: {states[i].cpu().numpy()}")
        print(f"    动作块: {action_chunks[i].cpu().numpy().flatten()}")
        print(f"    下一状态: {next_states[i].cpu().numpy()}")
        print(f"    多步奖励: {multi_rewards[i].cpu().numpy()[0]}")
        print(f"    完成标志: {dones[i].cpu().numpy()[0]}")


def explain_sampling_strategy():
    """解释采样策略"""
    print("\n" + "=" * 50)
    print("关于采样策略的说明:")
    print("=" * 50)
    print("1. 当前实现中，ReplayBuffer.sample() 方法会随机选择起始索引")
    print("2. 这意味着即使原始数据有时间相关性，采样后的批次也会打乱这种相关性")
    print("3. 对于多步数据，连续的步骤仍然保持其顺序关系")
    print("4. 这种策略有助于打破时间相关性，提高训练的稳定性")
    print("\n例如:")
    print("  原始序列: [s0,a0,r0], [s1,a1,r1], [s2,a2,r2], [s3,a3,r3]")
    print("  采样可能得到: [s2,a2,r2], [s0,a0,r0] (顺序被打乱)")
    print("  但对于每个样本，多步动作/奖励保持连续: [a0,a1,a2], [a2,a3,a4] 等")


if __name__ == "__main__":
    test_multi_step_processing()
    explain_sampling_strategy()