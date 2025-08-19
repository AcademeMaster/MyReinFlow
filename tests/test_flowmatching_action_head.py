#!/usr/bin/env python3
"""
测试FlowmatchingActionHead的独立脚本
"""

import torch
from transformers.feature_extraction_utils import BatchFeature

from action_head.flow_matching_action_head import FlowmatchingActionHead, FlowmatchingActionHeadConfig


def test_flowmatching_action_head():
    """测试FlowmatchingActionHead功能"""
    print("测试FlowmatchingActionHead...")
    
    # DiT模型配置
    dit_config = {
        "num_attention_heads": 4,
        "attention_head_dim": 32,
        "num_layers": 2,
        "dropout": 0.0,
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "upcast_attention": False,
        "norm_type": "ada_norm",
        "norm_elementwise_affine": False,
        "norm_eps": 1e-5,
        "max_num_positional_embeddings": 16,
        "compute_dtype": torch.float32,
        "final_dropout": False,
        "cross_attention_dim": 256,
    }
    
    # SelfAttentionTransformer配置
    vl_self_attention_cfg = {
        "num_attention_heads": 4,
        "attention_head_dim": 32,
        "num_layers": 2,
        "dropout": 0.0,
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "upcast_attention": False,
        "max_num_positional_embeddings": 16,
        "compute_dtype": torch.float32,
        "final_dropout": False,
    }
    
    config = FlowmatchingActionHeadConfig(
        diffusion_model_cfg=dit_config,
        input_embedding_dim=128,
        backbone_embedding_dim=128,
        hidden_size=128,
        action_dim=7,
        action_horizon=16,
        num_inference_timesteps=5,
        max_num_embodiments=8,
        num_target_vision_tokens=4,
        max_state_dim=10,  # 添加状态维度
        vl_self_attention_cfg=vl_self_attention_cfg,
    )
    
    # 创建模型
    model = FlowmatchingActionHead(config)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 8
    embedding_dim = config.input_embedding_dim
    action_horizon = config.action_horizon
    action_dim = config.action_dim
    
    # 模拟backbone输出
    backbone_output = BatchFeature({
        "backbone_features": torch.randn(batch_size, seq_len, embedding_dim),
        "backbone_attention_mask": torch.ones(batch_size, seq_len),
    })
    
    # 模拟动作输入
    action_input = BatchFeature({
        "state": torch.randn(batch_size, 10),  # 状态
        "action": torch.randn(batch_size, action_horizon, action_dim),  # 动作
        "embodiment_id": torch.randint(0, 8, (batch_size,)),  # 体现ID
        "action_mask": torch.ones(batch_size, action_horizon),  # 动作掩码
    })
    
    # 测试前向传播
    print("测试前向传播...")
    try:
        output = model(backbone_output, action_input)
        print(f"损失值: {output.loss}")
    except Exception as e:
        print(f"前向传播出错: {e}")
    
    # 测试动作生成
    print("测试动作生成...")
    try:
        with torch.no_grad():
            actions = model.get_action(backbone_output, action_input)
            print(f"生成的动作维度: {actions.action_pred.shape}")
    except Exception as e:
        print(f"动作生成出错: {e}")
    
    print("\nFlowmatchingActionHead测试完成!")


if __name__ == "__main__":
    test_flowmatching_action_head()