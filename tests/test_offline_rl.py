#!/usr/bin/env python3
"""
离线强化学习功能测试脚本
"""

import sys
import os
import torch
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from offlineflowrl.meanflow_ql import (
    Config, 
    ImprovedMeanFlowPolicyAgent, 
    DoubleCriticObsAct, 
    ConservativeMeanFQLModel,
    ReplayBuffer
)


def test_model_creation():
    """测试模型创建"""
    print("测试模型创建...")
    
    # 创建配置
    config = Config()
    config.pred_horizon = 5
    config.hidden_dim = 128
    config.time_dim = 32
    
    # 设置维度
    obs_dim = 10
    action_dim = 4
    action_horizon = 5
    
    try:
        # 创建组件
        actor = ImprovedMeanFlowPolicyAgent(obs_dim, action_dim, config)
        critic = DoubleCriticObsAct(obs_dim, action_dim, config.hidden_dim, action_horizon)
        model = ConservativeMeanFQLModel(actor, critic, config)
        
        print("✓ 模型创建成功")
        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False


def test_forward_pass():
    """测试前向传播"""
    print("测试前向传播...")
    
    # 创建配置
    config = Config()
    config.pred_horizon = 5
    config.hidden_dim = 128
    
    # 设置维度
    obs_dim = 10
    action_dim = 4
    action_horizon = 5
    batch_size = 2  # 减小批次大小以避免问题
    
    # 确定设备
    device = torch.device(config.device)
    
    try:
        # 创建组件
        actor = ImprovedMeanFlowPolicyAgent(obs_dim, action_dim, config)
        critic = DoubleCriticObsAct(obs_dim, action_dim, config.hidden_dim, action_horizon)
        model = ConservativeMeanFQLModel(actor, critic, config)
        
        # 创建测试数据并移到正确设备
        obs = torch.randn(batch_size, obs_dim).to(device)
        actions = torch.randn(batch_size, action_horizon, action_dim).to(device)
        next_obs = torch.randn(batch_size, obs_dim).to(device)
        rewards = torch.randn(batch_size, action_horizon).to(device)
        terminated = torch.zeros(batch_size, 1).to(device)
        
        # 测试critic loss
        critic_loss, critic_info = model.loss_critic(
            obs, actions, next_obs, rewards, terminated, config.gamma
        )
        
        # 测试actor loss - 创建正确的批次数据
        batch = {
            "observations": obs,
            "actions": actions
        }
        actor_loss, actor_info = model.loss_actor(obs, actions)
        
        print("✓ 前向传播测试成功")
        print(f"  Critic loss: {critic_loss.item():.6f}")
        print(f"  Actor loss: {actor_loss.item():.6f}")
        return True
    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_replay_buffer():
    """测试经验回放缓冲区"""
    print("测试经验回放缓冲区...")
    
    try:
        # 创建缓冲区
        buffer = ReplayBuffer(capacity=1000)
        
        # 添加一些测试数据
        for i in range(5):
            transition = {
                "observations": np.random.randn(1, 10).astype(np.float32),
                "actions": np.random.randn(1, 5, 4).astype(np.float32),
                "next_observations": np.random.randn(1, 10).astype(np.float32),
                "rewards": np.random.randn(1, 5).astype(np.float32),
                "terminated": np.random.choice([0, 1], size=(1, 1)).astype(np.float32)
            }
            buffer.add(transition)
        
        # 测试采样
        batch = buffer.sample(3)
        
        print("✓ 经验回放缓冲区测试成功")
        print(f"  缓冲区大小: {len(buffer)}")
        print(f"  采样批次观测形状: {batch['observations'].shape}")
        return True
    except Exception as e:
        print(f"✗ 经验回放缓冲区测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_sampling():
    """测试智能体采样"""
    print("测试智能体采样...")
    
    # 创建配置
    config = Config()
    config.pred_horizon = 5
    config.hidden_dim = 128
    
    # 设置维度
    obs_dim = 10
    action_dim = 4
    
    try:
        # 创建智能体
        agent = ImprovedMeanFlowPolicyAgent(obs_dim, action_dim, config)
        
        # 创建测试批次 - 使用正确的维度 [batch_size, obs_horizon, obs_dim]
        batch = {
            "observations": torch.randn(2, 1, obs_dim)  # [batch_size, obs_horizon, obs_dim]
        }
        
        # 测试动作预测
        actions = agent.predict_action_chunk(batch, n_steps=5)
        
        print("✓ 智能体采样测试成功")
        print(f"  动作形状: {actions.shape}")
        return True
    except Exception as e:
        print(f"✗ 智能体采样测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始离线强化学习功能测试")
    print("=" * 50)
    
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行各项测试
    tests = [
        test_model_creation,
        test_agent_sampling,
        test_replay_buffer,
        test_forward_pass
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过!")
        return 0
    else:
        print("❌ 部分测试失败!")
        return 1


if __name__ == "__main__":
    sys.exit(main())