#!/usr/bin/env python3
"""
简化版DiT模型测试
"""

import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

from action_head.cross_attention_dit import DiT


def test_simple_dit():
    """测试简化版DiT模型"""
    print("测试简化版DiT模型...")
    
    # DiT模型配置
    dit_config = {
        "num_attention_heads": 4,
        "attention_head_dim": 32,
        "output_dim": 7,
        "num_layers": 2,
        "dropout": 0.0,
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "upcast_attention": False,
        "norm_type": "ada_norm",
        "norm_elementwise_affine": False,
        "norm_eps": 1e-5,
        "max_num_positional_embeddings": 32,  # 增加到32以匹配测试序列长度
        "compute_dtype": torch.float32,
        "final_dropout": False,
        "cross_attention_dim": 128,
        "positional_embeddings": "sinusoidal",  # 明确指定位置编码类型
    }
    
    # 创建模型
    model = DiT(**dit_config)
    print(f"DiT模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 创建测试数据
    batch_size = 2
    hidden_seq_len = 20
    encoder_seq_len = 8
    hidden_dim = 128  # num_attention_heads * attention_head_dim
    
    # 隐藏状态（包含状态、未来标记和动作特征）
    hidden_states = torch.randn(batch_size, hidden_seq_len, hidden_dim)
    
    # 编码器隐藏状态（视觉和语言嵌入）
    encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, hidden_dim)
    
    # 时间步
    timestep = torch.randint(0, 1000, (batch_size,))
    
    print(f"隐藏状态维度: {hidden_states.shape}")
    print(f"编码器隐藏状态维度: {encoder_hidden_states.shape}")
    print(f"时间步维度: {timestep.shape}")
    
    # 测试前向传播
    print("测试前向传播...")
    try:
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep
        )
        print(f"输出维度: {output.shape}")
        print("DiT模型测试成功!")
    except Exception as e:
        print(f"DiT模型测试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_simple_dit()