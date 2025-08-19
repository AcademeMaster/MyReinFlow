#!/usr/bin/env python3
"""
测试TimeConditionedFlowModel和FlowPolicyAgent的独立脚本
"""

import torch
from normalflow import TimeConditionedFlowModel, FlowPolicyAgent
from config import Config


def test_time_conditioned_flow_model():
    """测试TimeConditionedFlowModel功能"""
    print("测试TimeConditionedFlowModel...")
    
    # 创建配置
    config = Config()
    config.action_dim = 7
    
    # 创建模型
    obs_dim = 23
    action_dim = 7
    model = TimeConditionedFlowModel(obs_dim, action_dim, config)
    
    # 创建测试数据
    batch_size = 2
    obs = torch.randn(batch_size, obs_dim)
    t = torch.rand(batch_size)
    noise = torch.randn(batch_size, config.pred_horizon, action_dim)
    
    # 测试前向传播
    output = model(obs, t, noise)
    print(f"输入观测维度: {obs.shape}")
    print(f"输入时间维度: {t.shape}")
    print(f"输入噪声维度: {noise.shape}")
    print(f"输出速度场维度: {output.shape}")
    
    print("TimeConditionedFlowModel测试完成!")


def test_flow_policy_agent():
    """测试FlowPolicyAgent功能"""
    print("\n测试FlowPolicyAgent...")
    
    # 创建配置
    config = Config()
    config.action_dim = 7
    
    # 创建代理
    obs_dim = 23
    action_dim = 7
    agent = FlowPolicyAgent(obs_dim, action_dim, config)
    
    # 创建测试批次
    batch_size = 2
    batch = {
        "observations": torch.randn(batch_size, config.obs_horizon, obs_dim),
        "actions": torch.randn(batch_size, config.pred_horizon, action_dim)
    }
    
    # 测试代理前向传播
    loss = agent.forward(batch)
    print(f"损失值: {loss}")
    
    # 测试动作预测
    predicted_actions = agent.predict_action_chunk(batch)
    print(f"预测动作维度: {predicted_actions.shape}")
    
    # 测试单个动作选择
    single_action = agent.select_action(batch)
    print(f"单个动作维度: {single_action.shape}")
    
    print("FlowPolicyAgent测试完成!")


if __name__ == "__main__":
    test_time_conditioned_flow_model()
    test_flow_policy_agent()
    
    print("\n所有测试完成!")