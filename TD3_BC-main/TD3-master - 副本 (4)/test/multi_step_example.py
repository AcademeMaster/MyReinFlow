#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多步采样ReplayBuffer使用示例

这个示例展示了如何使用MultiStepReplayBuffer进行多步采样，
支持actor输出多步动作序列，用于n步TD学习。
"""

import numpy as np
import torch
from utils import MultiStepReplayBuffer

def demo_multi_step_buffer():
    """演示MultiStepReplayBuffer的使用"""
    
    # 初始化参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capacity = 1000
    n_step = 4  # 多步长度
    state_dim = 3
    action_dim = 2
    
    # 创建多步缓冲区
    buffer = MultiStepReplayBuffer(capacity=capacity, device=device, n_step=n_step)
    
    print(f"创建MultiStepReplayBuffer: capacity={capacity}, n_step={n_step}")
    print(f"设备: {device}")
    print("="*60)
    
    # 模拟几个episode的数据收集
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        
        # 模拟一个episode，长度随机
        episode_length = np.random.randint(5, 12)  # 5-11步
        print(f"Episode长度: {episode_length}")
        
        for step in range(episode_length):
            # 生成模拟数据
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = (step == episode_length - 1)  # 最后一步done=True
            
            # 添加到缓冲区
            buffer.add(state, action, reward, next_state, done)
            
            print(f"  步骤 {step + 1}: done={done}")
        
        print(f"Episode {episode + 1} 完成，缓冲区大小: {buffer.size()}")
    
    print(f"\n总共收集了 {buffer.get_episode_count()} 个episodes")
    print(f"可用的多步样本数量: {buffer.size()}")
    
    # 演示采样
    if buffer.size() >= 4:
        print("\n" + "="*60)
        print("演示多步采样:")
        
        batch_size = 4
        states, actions, rewards, next_states, dones, actual_steps = buffer.sample(batch_size)
        
        print(f"\n采样批次大小: {batch_size}")
        print(f"states shape: {states.shape}")
        print(f"actions shape: {actions.shape}")
        print(f"rewards shape: {rewards.shape}")
        print(f"next_states shape: {next_states.shape}")
        print(f"dones shape: {dones.shape}")
        print(f"actual_steps shape: {actual_steps.shape}")
        
        print("\n详细数据:")
        for i in range(batch_size):
            print(f"\n样本 {i + 1}:")
            print(f"  实际步数: {actual_steps[i].item()}")
            print(f"  初始状态: {states[i].cpu().numpy()}")
            print(f"  动作序列: {actions[i].cpu().numpy()}")
            print(f"  奖励序列: {rewards[i].cpu().numpy()}")
            print(f"  最终状态: {next_states[i].cpu().numpy()}")
            print(f"  done序列: {dones[i].cpu().numpy()}")
    
    return buffer

def demo_boundary_conditions():
    """演示边界条件处理"""
    print("\n" + "="*60)
    print("演示边界条件处理:")
    
    device = torch.device("cpu")
    buffer = MultiStepReplayBuffer(capacity=100, device=device, n_step=4)
    
    # 创建一个很短的episode（只有2步）
    print("\n创建一个只有2步的短episode:")
    
    # 第1步
    buffer.add(
        state=np.array([1.0, 2.0]), 
        action=np.array([0.1, 0.2]), 
        reward=1.0,
        next_state=np.array([1.1, 2.1]), 
        done=False
    )
    
    # 第2步（episode结束）
    buffer.add(
        state=np.array([1.1, 2.1]), 
        action=np.array([0.2, 0.3]), 
        reward=2.0,
        next_state=np.array([1.2, 2.2]), 
        done=True
    )
    
    print(f"缓冲区大小: {buffer.size()}")
    
    if buffer.size() > 0:
        # 采样一个样本
        states, actions, rewards, next_states, dones, actual_steps = buffer.sample(1)
        
        print(f"\n采样结果:")
        print(f"实际步数: {actual_steps[0].item()} (期望: ≤2)")
        print(f"动作序列形状: {actions[0].shape} (应该是 [4, 2])")
        print(f"奖励序列: {rewards[0].cpu().numpy()}")
        print(f"done序列: {dones[0].cpu().numpy()}")
        
        # 检查填充
        print("\n填充检查:")
        for i in range(4):
            if i < actual_steps[0].item():
                print(f"  步骤 {i+1}: 有效数据")
            else:
                print(f"  步骤 {i+1}: 填充数据 (done={dones[0][i].item()})")

def demo_usage_with_agent():
    """演示如何与智能体配合使用"""
    print("\n" + "="*60)
    print("与智能体配合使用的伪代码示例:")
    
    code_example = """
# 1. 初始化多步缓冲区
buffer = MultiStepReplayBuffer(capacity=100000, device=device, n_step=4)

# 2. 训练循环中的数据收集
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Actor输出多步动作序列
        multi_actions = actor.predict_multi_step(state, n_step=4)
        
        # 依次执行每个动作
        for action in multi_actions:
            next_state, reward, done, _ = env.step(action)
            
            # 存储单步转换
            buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            if done:
                break

# 3. 训练时采样多步数据
if buffer.size() > batch_size:
    states, actions, rewards, next_states, dones, actual_steps = buffer.sample(batch_size)
    
    # 使用多步数据进行n步TD学习
    # states: [batch_size, state_dim] - 起始状态
    # actions: [batch_size, n_step, action_dim] - 动作序列
    # rewards: [batch_size, n_step] - 奖励序列
    # next_states: [batch_size, state_dim] - 最终状态
    # dones: [batch_size, n_step] - done标记序列
    # actual_steps: [batch_size] - 实际有效步数
    
    loss = compute_n_step_td_loss(states, actions, rewards, next_states, dones, actual_steps)
    optimizer.step()
"""
    
    print(code_example)

if __name__ == "__main__":
    print("MultiStepReplayBuffer 演示")
    print("="*60)
    
    # 基本使用演示
    buffer = demo_multi_step_buffer()
    
    # 边界条件演示
    demo_boundary_conditions()
    
    # 使用示例
    demo_usage_with_agent()
    
    print("\n" + "="*60)
    print("演示完成！")
    
    print("\n主要特性总结:")
    print("1. 支持多步采样: state[i], action[i:i+h], reward[i:i+h], next_state[i+h], done[i:i+h]")
    print("2. 自动处理边界条件: episode结束时自动填充")
    print("3. 保持轨迹顺序: 确保多步动作来自同一策略")
    print("4. 高效采样: 预处理数据，支持快速随机采样")
    print("5. 灵活配置: 可调节n_step参数")