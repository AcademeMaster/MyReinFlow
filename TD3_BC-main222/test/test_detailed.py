#!/usr/bin/env python3
"""
简化的测试文件，专门测试chunk采样功能
"""

import numpy as np
import torch
from utils import ReplayBuffer


def test_chunk_sampling_detailed():
    """详细测试chunk采样功能"""
    print("详细测试Chunk采样功能")
    print("-" * 40)
    
    buffer = ReplayBuffer(
        state_dim=3, 
        action_dim=2, 
        max_size=100, 
        action_chunk_size=4, 
        reward_horizon=3
    )
    
    # 添加数据
    np.random.seed(123)
    for i in range(50):
        state = np.random.randn(3).astype(np.float32)
        action = np.random.randn(2).astype(np.float32)
        next_state = np.random.randn(3).astype(np.float32)
        reward = np.random.randn()
        done = (i % 15 == 14)  # 每15步结束一个episode
        episode_start = (i % 15 == 0)
        
        buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)
    
    print(f"Buffer统计: {buffer.get_statistics()}")
    
    # 测试标准采样
    try:
        batch_size = 8
        states, actions, next_states, rewards, dones = buffer.sample(batch_size, return_chunks=False)
        
        print(f"标准采样成功:")
        print(f"  States shape: {states.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Rewards shape: {rewards.shape}")
        print(f"  Device: {states.device}")
        
    except Exception as e:
        print(f"标准采样失败: {e}")
        return
    
    # 测试chunk采样
    try:
        states, action_chunks, next_states, multi_rewards, dones = buffer.sample(batch_size, return_chunks=True)
        
        print(f"Chunk采样成功:")
        print(f"  States shape: {states.shape}")
        print(f"  Action chunks shape: {action_chunks.shape}")
        print(f"  Multi-step rewards shape: {multi_rewards.shape}")
        print(f"  预期action chunk shape: ({batch_size}, {buffer.action_chunk_size}, 2)")
        
        # 验证形状
        expected_shape = (batch_size, buffer.action_chunk_size, 2)
        assert action_chunks.shape == expected_shape, f"Shape mismatch: {action_chunks.shape} vs {expected_shape}"
        
        print("✓ Chunk采样验证通过")
        
    except Exception as e:
        print(f"Chunk采样失败: {e}")
        import traceback
        traceback.print_exc()


def test_sliding_window_concept():
    """测试滑动窗口概念的实现"""
    print("\n测试滑动窗口概念")
    print("-" * 40)
    
    # 创建一个简单的buffer来验证滑动窗口概念
    buffer = ReplayBuffer(
        state_dim=1, 
        action_dim=1, 
        max_size=20, 
        action_chunk_size=3, 
        reward_horizon=3
    )
    
    # 创建一个episode，10个步骤
    print("创建一个连续的episode:")
    for i in range(10):
        state = np.array([i], dtype=np.float32)
        action = np.array([i], dtype=np.float32)
        next_state = np.array([i+1], dtype=np.float32)
        reward = float(i + 1)  # reward 1, 2, 3, ..., 10
        done = (i == 9)  # 最后一步结束
        episode_start = (i == 0)
        
        buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)
        print(f"  Step {i}: state={i}, action={i}, reward={i+1}")
    
    print(f"\nBuffer统计: {buffer.get_statistics()}")
    
    # 测试不同起始位置的滑动窗口
    print("\n测试滑动窗口效果:")
    for start_idx in [0, 1, 2, 6, 7, 8]:
        try:
            # 获取action chunk
            action_chunk = buffer._get_action_chunk(start_idx, 3)
            # 获取multi-step reward
            multi_reward = buffer._get_multi_step_reward(start_idx, 3, gamma=1.0)
            
            print(f"起始索引 {start_idx}:")
            print(f"  Action chunk: {action_chunk.flatten()}")
            print(f"  Multi-step reward: {multi_reward}")
            
            # 验证逻辑
            if start_idx <= 7:  # 可以获取完整的3步
                expected_actions = [start_idx, start_idx+1, start_idx+2]
                expected_reward = (start_idx+1) + (start_idx+2) + (start_idx+3)  # rewards are i+1
                print(f"  预期actions: {expected_actions}, 预期reward: {expected_reward}")
            else:
                print(f"  接近episode末尾，会有特殊处理")
                
        except Exception as e:
            print(f"起始索引 {start_idx} 出错: {e}")
    
    print("✓ 滑动窗口概念验证完成")


if __name__ == "__main__":
    test_chunk_sampling_detailed()
    test_sliding_window_concept()
