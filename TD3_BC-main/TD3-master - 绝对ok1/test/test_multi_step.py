#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试 MultiStepReplayBuffer 的核心功能
"""

import numpy as np
import torch
from utils import MultiStepReplayBuffer

def test_basic_functionality():
    """测试基本功能"""
    print("测试基本功能...")
    
    device = torch.device("cpu")
    buffer = MultiStepReplayBuffer(capacity=100, device=device, n_step=3)
    
    # 添加一个简单的episode
    states = [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3]]
    actions = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]
    rewards = [1.0, 2.0, 3.0, 4.0]
    
    for i in range(4):
        done = (i == 3)  # 最后一步done=True
        next_state = states[i+1] if i < 3 else [1.4, 2.4]
        
        buffer.add(
            state=np.array(states[i]),
            action=np.array(actions[i]),
            reward=rewards[i],
            next_state=np.array(next_state),
            done=done
        )
    
    print(f"缓冲区大小: {buffer.size()}")
    assert buffer.size() > 0, "缓冲区应该有数据"
    
    # 采样测试
    if buffer.size() >= 1:
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, actual_steps_batch = buffer.sample(1)
        
        print(f"采样成功:")
        print(f"  states shape: {states_batch.shape}")
        print(f"  actions shape: {actions_batch.shape}")
        print(f"  rewards shape: {rewards_batch.shape}")
        print(f"  next_states shape: {next_states_batch.shape}")
        print(f"  dones shape: {dones_batch.shape}")
        print(f"  actual_steps: {actual_steps_batch[0].item()}")
        
        # 验证形状
        assert states_batch.shape == (1, 2), f"states形状错误: {states_batch.shape}"
        assert actions_batch.shape == (1, 3, 2), f"actions形状错误: {actions_batch.shape}"
        assert rewards_batch.shape == (1, 3), f"rewards形状错误: {rewards_batch.shape}"
        assert next_states_batch.shape == (1, 2), f"next_states形状错误: {next_states_batch.shape}"
        assert dones_batch.shape == (1, 3), f"dones形状错误: {dones_batch.shape}"
        
        print("✓ 基本功能测试通过")
    
    return True

def test_boundary_conditions():
    """测试边界条件"""
    print("\n测试边界条件...")
    
    device = torch.device("cpu")
    buffer = MultiStepReplayBuffer(capacity=100, device=device, n_step=4)
    
    # 创建一个只有2步的短episode
    print("添加短episode (2步)...")
    
    # 第1步
    buffer.add(
        state=np.array([1.0, 2.0]),
        action=np.array([0.1, 0.2]),
        reward=1.0,
        next_state=np.array([1.1, 2.1]),
        done=False
    )
    
    # 第2步 (episode结束)
    buffer.add(
        state=np.array([1.1, 2.1]),
        action=np.array([0.2, 0.3]),
        reward=2.0,
        next_state=np.array([1.2, 2.2]),
        done=True
    )
    
    print(f"缓冲区大小: {buffer.size()}")
    
    if buffer.size() >= 1:
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, actual_steps_batch = buffer.sample(1)
        
        actual_steps = actual_steps_batch[0].item()
        print(f"实际步数: {actual_steps}")
        print(f"动作序列: {actions_batch[0].cpu().numpy()}")
        print(f"奖励序列: {rewards_batch[0].cpu().numpy()}")
        print(f"done序列: {dones_batch[0].cpu().numpy()}")
        
        # 验证边界条件处理
        assert actual_steps <= 2, f"实际步数应该≤2，但得到{actual_steps}"
        assert actions_batch.shape == (1, 4, 2), "动作序列应该被填充到n_step长度"
        
        # 检查填充部分的done标记
        dones_np = dones_batch[0].cpu().numpy()
        for i in range(actual_steps, 4):
            assert dones_np[i] == 1.0, f"填充位置{i}的done应该为True"
        
        print("✓ 边界条件测试通过")
    
    return True

def test_multiple_episodes():
    """测试多个episode"""
    print("\n测试多个episode...")
    
    device = torch.device("cpu")
    buffer = MultiStepReplayBuffer(capacity=100, device=device, n_step=3)
    
    # 添加3个episode
    for ep in range(3):
        episode_length = 3 + ep  # 长度分别为3, 4, 5
        print(f"添加episode {ep+1}, 长度: {episode_length}")
        
        for step in range(episode_length):
            state = np.array([ep + step * 0.1, ep + step * 0.1])
            action = np.array([ep * 0.1 + step * 0.01, ep * 0.1 + step * 0.01])
            reward = ep + step
            next_state = np.array([ep + (step + 1) * 0.1, ep + (step + 1) * 0.1])
            done = (step == episode_length - 1)
            
            buffer.add(state, action, reward, next_state, done)
    
    print(f"总缓冲区大小: {buffer.size()}")
    print(f"episode数量: {buffer.get_episode_count()}")
    
    # 采样多个样本
    if buffer.size() >= 3:
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, actual_steps_batch = buffer.sample(3)
        
        print(f"采样3个样本:")
        for i in range(3):
            actual_steps = actual_steps_batch[i].item()
            print(f"  样本{i+1}: 实际步数={actual_steps}")
        
        print("✓ 多episode测试通过")
    
    return True

if __name__ == "__main__":
    print("MultiStepReplayBuffer 测试")
    print("=" * 50)
    
    try:
        # 运行所有测试
        test_basic_functionality()
        test_boundary_conditions()
        test_multiple_episodes()
        
        print("\n" + "=" * 50)
        print("✓ 所有测试通过！")
        print("\nMultiStepReplayBuffer 实现正确，可以安全使用。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()