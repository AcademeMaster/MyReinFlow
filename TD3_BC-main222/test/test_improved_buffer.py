"""
测试改进后的relaybuffer，验证action chunking和multi-step rewards的滑动窗口功能
以及gymnasium格式的支持
"""

import numpy as np
import torch
from relaybuffer import Dataset, ReplayBuffer

def test_gymnasium_format():
    """测试gymnasium格式的terminated和truncated处理"""
    print("=== 测试gymnasium格式处理 ===")

    # 创建gymnasium格式的数据
    data = {
        'observations': np.random.randn(100, 10).astype(np.float32),
        'actions': np.random.randn(100, 4).astype(np.float32),
        'next_observations': np.random.randn(100, 10).astype(np.float32),
        'rewards': np.random.randn(100).astype(np.float32),
        'terminated': np.zeros(100, dtype=np.bool_),  # gymnasium格式
        'truncated': np.zeros(100, dtype=np.bool_),   # gymnasium格式
        'masks': np.ones(100, dtype=np.float32),
    }

    # 设置一些episode结束点
    data['terminated'][19] = True
    data['terminated'][49] = True
    data['terminated'][79] = True
    data['terminated'][99] = True

    data['truncated'][29] = True  # 某些步骤被截断
    data['truncated'][69] = True

    # 创建Dataset
    dataset = Dataset.create(**data)

    print(f"原始terminated数量: {data['terminated'].sum()}")
    print(f"原始truncated数量: {data['truncated'].sum()}")
    print(f"合并后terminals数量: {dataset['terminals'].sum()}")
    print(f"Episode数量: {len(dataset.terminal_locs)}")
    print(f"Terminal位置: {dataset.terminal_locs}")
    print()

def test_sliding_window_chunks():
    """测试滑动窗口的action chunking和multi-step rewards"""
    print("=== 测试滑动窗口功能 ===")

    # 创建多个episode的数据
    total_steps = 200
    data = {
        'observations': np.random.randn(total_steps, 8).astype(np.float32),
        'actions': np.arange(total_steps * 3).reshape(total_steps, 3).astype(np.float32),  # 方便验��
        'next_observations': np.random.randn(total_steps, 8).astype(np.float32),
        'rewards': np.arange(total_steps, dtype=np.float32),  # 递增奖励，方便验证
        'terminals': np.zeros(total_steps, dtype=np.float32),
        'masks': np.ones(total_steps, dtype=np.float32),
    }

    # 设置episode边界 (每50步一个episode)
    for i in range(49, total_steps, 50):
        data['terminals'][i] = 1.0

    dataset = Dataset.create(**data)
    dataset.set_chunking_params(action_chunk_size=4, reward_horizon=5)

    print(f"数据集大小: {dataset.size}")
    print(f"Episode边界: {dataset.terminal_locs}")
    print(f"Episode起始: {dataset.initial_locs}")

    # 测试滑动窗口采样
    batch = dataset.sample_chunks(batch_size=8, action_chunk_size=4, reward_horizon=5, gamma=0.9)

    print(f"采样批次大小: {len(batch['observations'])}")
    print(f"Action chunks形状: {batch['action_chunks'].shape}")  # 应该是[8, 4, 3]
    print(f"Multi-step rewards形状: {batch['multi_step_rewards'].shape}")  # 应该是[8, 1]

    # 验证第一个样本的action chunk
    first_obs_idx = np.where((dataset['observations'] == batch['observations'][0]).all(axis=1))[0][0]
    print(f"\n第一个样本的起始索引: {first_obs_idx}")

    expected_actions = data['actions'][first_obs_idx:first_obs_idx+4]
    actual_actions = batch['action_chunks'][0].numpy()
    print(f"期望的action chunk:\n{expected_actions}")
    print(f"实际的action chunk:\n{actual_actions}")
    print(f"Action chunk正确性: {np.allclose(expected_actions, actual_actions)}")

    # 验证multi-step reward
    expected_reward = sum(data['rewards'][first_obs_idx + i] * (0.9 ** i) for i in range(5))
    actual_reward = batch['multi_step_rewards'][0].item()
    print(f"期望的multi-step reward: {expected_reward}")
    print(f"实际的multi-step reward: {actual_reward}")
    print(f"Multi-step reward正确性: {abs(expected_reward - actual_reward) < 1e-5}")
    print()

def test_episode_boundary_handling():
    """测试episode边界处理"""
    print("=== 测试episode边界处理 ===")

    # 创建短episode来测试边界
    episode_lengths = [10, 15, 8, 12]
    total_steps = sum(episode_lengths)

    data = {
        'observations': np.random.randn(total_steps, 5).astype(np.float32),
        'actions': np.arange(total_steps * 2).reshape(total_steps, 2).astype(np.float32),
        'next_observations': np.random.randn(total_steps, 5).astype(np.float32),
        'rewards': np.ones(total_steps, dtype=np.float32),
        'terminals': np.zeros(total_steps, dtype=np.float32),
        'masks': np.ones(total_steps, dtype=np.float32),
    }

    # 设置episode结束点
    cumulative = 0
    for length in episode_lengths:
        cumulative += length
        data['terminals'][cumulative - 1] = 1.0

    dataset = Dataset.create(**data)

    # 测试跨episode边界的处理
    chunk_size = 6
    valid_indices = dataset._get_valid_start_indices(chunk_size)

    print(f"Episode长度: {episode_lengths}")
    print(f"总步数: {total_steps}")
    print(f"Chunk大小: {chunk_size}")
    print(f"有效起始索引数量: {len(valid_indices)}")
    print(f"有效起始索引: {valid_indices}")

    # 验证每个有效索引确实不会跨episode
    for idx in valid_indices[:5]:  # 检查前5个
        episode_start = dataset.initial_locs[np.searchsorted(dataset.initial_locs, idx, side='right') - 1]
        episode_end = dataset.terminal_locs[np.searchsorted(dataset.terminal_locs, idx)]
        print(f"索引{idx}: episode [{episode_start}, {episode_end}], chunk终点: {idx + chunk_size - 1}")

    # 尝试采样
    if len(valid_indices) > 0:
        try:
            batch = dataset.sample_chunks(batch_size=min(4, len(valid_indices)),
                                        action_chunk_size=chunk_size,
                                        reward_horizon=chunk_size)
            print(f"成功采样批次，action_chunks形状: {batch['action_chunks'].shape}")
        except Exception as e:
            print(f"采样失败: {e}")
    print()

def test_torch_compatibility():
    """测试PyTorch张量兼容性"""
    print("=== 测试PyTorch张量兼容性 ===")

    # 创建PyTorch张量数据
    total_steps = 50
    data = {
        'observations': torch.randn(total_steps, 6, dtype=torch.float32),
        'actions': torch.randn(total_steps, 3, dtype=torch.float32),
        'next_observations': torch.randn(total_steps, 6, dtype=torch.float32),
        'rewards': torch.randn(total_steps, dtype=torch.float32),
        'terminated': torch.zeros(total_steps, dtype=torch.bool),
        'truncated': torch.zeros(total_steps, dtype=torch.bool),
        'masks': torch.ones(total_steps, dtype=torch.float32),
    }

    # 设置episode边界
    data['terminated'][19] = True
    data['terminated'][39] = True
    data['terminated'][49] = True

    data['truncated'][29] = True

    dataset = Dataset.create(**data)

    print(f"PyTorch数据类型: {type(dataset['observations'])}")
    print(f"Terminals数据类型: {type(dataset['terminals'])}")
    print(f"Dataset大小: {dataset.size}")

    # 测试采样
    batch = dataset.sample_chunks(batch_size=4, action_chunk_size=3, reward_horizon=3)

    print(f"采样结果类型: {type(batch['action_chunks'])}")
    print(f"Action chunks形状: {batch['action_chunks'].shape}")
    print(f"Multi-step rewards形状: {batch['multi_step_rewards'].shape}")
    print()

def test_replay_buffer_integration():
    """测试ReplayBuffer集成"""
    print("=== 测试ReplayBuffer集成 ===")

    # 创建示例transition
    transition = {
        'observations': np.random.randn(10),
        'actions': np.random.randn(4),
        'next_observations': np.random.randn(10),
        'rewards': 1.0,
        'terminated': False,
        'truncated': False,
        'masks': 1.0,
    }

    # 创建ReplayBuffer
    buffer = ReplayBuffer.create(transition, size=100)
    buffer.set_chunking_params(action_chunk_size=3, reward_horizon=4)

    # 添加一些transitions
    for i in range(50):
        trans = {
            'observations': np.random.randn(10),
            'actions': np.random.randn(4),
            'next_observations': np.random.randn(10),
            'rewards': float(i % 10),  # 循环奖励模式
            'terminated': (i % 20 == 19),  # 每20步结束episode
            'truncated': (i % 25 == 24),   # 某些步骤截断
            'masks': 1.0,
        }
        buffer.add_transition(trans)

    print(f"Buffer大小: {buffer.size}")
    print(f"Buffer指针位置: {buffer.pointer}")

    # 测试传统采样
    traditional_batch = buffer.sample(8)
    print(f"传统采样 - observations形状: {traditional_batch['observations'].shape}")

    # 测试chunk采样
    try:
        chunk_batch = buffer.sample_chunks(batch_size=4, action_chunk_size=3, reward_horizon=4)
        print(f"Chunk采样 - action_chunks形状: {chunk_batch['action_chunks'].shape}")
        print(f"Chunk采样 - multi_step_rewards形状: {chunk_batch['multi_step_rewards'].shape}")
    except Exception as e:
        print(f"Chunk采样失败: {e}")

    print()

if __name__ == "__main__":
    test_gymnasium_format()
    test_sliding_window_chunks()
    test_episode_boundary_handling()
    test_torch_compatibility()
    test_replay_buffer_integration()

    print("=== 所有测试完成 ===")
