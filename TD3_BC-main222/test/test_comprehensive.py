#!/usr/bin/env python3
"""
全面测试ReplayBuffer的逻辑漏洞、边界条件和bug
"""

import numpy as np
import torch
from utils import ReplayBuffer


def test_circular_buffer_behavior():
    """测试循环buffer的边界行为"""
    print("测试1: 循环Buffer边界行为")
    print("-" * 40)

    # 创建很小的buffer来测试循环行为
    buffer = ReplayBuffer(state_dim=2, action_dim=1, max_size=5, action_chunk_size=3, reward_horizon=3)

    # 填充超过max_size的数据
    for i in range(8):
        state = np.array([i, i+0.1], dtype=np.float32)
        action = np.array([i], dtype=np.float32)
        next_state = np.array([i+1, i+1.1], dtype=np.float32)
        reward = float(i + 1)
        done = (i == 3)  # 在索引3处结束episode
        episode_start = (i == 0) or (i == 4)  # 在索引0和4处开始新episode

        buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)
        print(f"添加索引{i}: ptr={buffer.ptr}, size={buffer.size}")

    print(f"\n最终状态: ptr={buffer.ptr}, size={buffer.size}")
    print("Buffer内容:")
    for i in range(buffer.size):
        print(f"  [{i}]: state={buffer.state[i]}, action={buffer.action[i]}, "
              f"episode_start={buffer.episode_starts[i]}, episode_end={buffer.episode_ends[i]}")

    # 测试循环buffer中的episode边界
    print("\n测试循环buffer中的action chunk:")
    try:
        for start_idx in range(buffer.size):
            chunk = buffer._get_action_chunk(start_idx, 3)
            multi_reward = buffer._get_multi_step_reward(start_idx, 3, gamma=1.0)
            print(f"从索引{start_idx}: actions={chunk.flatten()}, reward={multi_reward}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def test_edge_cases():
    """测试各种边界情况"""
    print("\n\n测试2: 边界情况")
    print("-" * 40)

    # 测试1: 空buffer
    empty_buffer = ReplayBuffer(state_dim=2, action_dim=1, max_size=10, action_chunk_size=3, reward_horizon=3)

    try:
        empty_buffer.sample(1)
        print("❌ 空buffer应该抛出异常")
    except ValueError as e:
        print(f"✓ 空buffer正确抛出异常: {e}")

    try:
        empty_buffer.normalize_states()
        print("❌ 空buffer标准化应该抛出异常")
    except ValueError as e:
        print(f"✓ 空buffer标准化正确抛出异常: {e}")

    # 测试2: 单个样本
    single_buffer = ReplayBuffer(state_dim=2, action_dim=1, max_size=10, action_chunk_size=3, reward_horizon=3)
    single_buffer.add(
        np.array([1, 2], dtype=np.float32),
        np.array([1], dtype=np.float32),
        np.array([2, 3], dtype=np.float32),
        1.0,
        done=True,
        episode_start=True
    )

    try:
        states, action_chunks, next_states, multi_rewards, dones = single_buffer.sample(1, return_chunks=True)
        print(f"✓ 单样本chunk采样成功: action_chunk shape={action_chunks.shape}")
        print(f"  Action chunk: {action_chunks[0].flatten()}")
        print(f"  Multi reward: {multi_rewards[0].item()}")
    except Exception as e:
        print(f"❌ 单样本chunk采样失败: {e}")

    # 测试3: chunk_size > buffer_size
    small_buffer = ReplayBuffer(state_dim=2, action_dim=1, max_size=10, action_chunk_size=5, reward_horizon=7)
    for i in range(3):
        small_buffer.add(
            np.array([i, i+0.1], dtype=np.float32),
            np.array([i], dtype=np.float32),
            np.array([i+1, i+1.1], dtype=np.float32),
            float(i+1),
            done=(i == 2),
            episode_start=(i == 0)
        )

    try:
        states, action_chunks, next_states, multi_rewards, dones = small_buffer.sample(2, return_chunks=True)
        print(f"✓ chunk_size > buffer_size 采样成功")
        print(f"  Action chunks shape: {action_chunks.shape}")
        for i in range(2):
            print(f"  Sample {i}: actions={action_chunks[i].flatten()}, reward={multi_rewards[i].item()}")
    except Exception as e:
        print(f"❌ chunk_size > buffer_size 采样失败: {e}")


def test_episode_boundary_logic():
    """测试episode边界逻辑的正确性"""
    print("\n\n测试3: Episode边界逻辑")
    print("-" * 40)

    buffer = ReplayBuffer(state_dim=1, action_dim=1, max_size=20, action_chunk_size=4, reward_horizon=4)

    # 创建复杂的episode结构
    episode_structure = [
        # Episode 1: 3步
        (0, False, True),   # step 0, not done, episode start
        (1, False, False),  # step 1
        (2, True, False),   # step 2, done

        # Episode 2: 2步 (短episode)
        (10, False, True),  # step 3, episode start
        (11, True, False),  # step 4, done

        # Episode 3: 5步
        (20, False, True),  # step 5, episode start
        (21, False, False), # step 6
        (22, False, False), # step 7
        (23, False, False), # step 8
        (24, True, False),  # step 9, done
    ]

    print("创建episode结构:")
    for i, (action_val, done, episode_start) in enumerate(episode_structure):
        state = np.array([action_val], dtype=np.float32)
        action = np.array([action_val], dtype=np.float32)
        next_state = np.array([action_val + 1], dtype=np.float32)
        reward = float(action_val + 1)

        buffer.add(state, action, next_state, reward, done=done, episode_start=episode_start)
        episode_num = 1 if i <= 2 else (2 if i <= 4 else 3)
        print(f"  索引{i}: Episode {episode_num}, action={action_val}, done={done}, start={episode_start}")

    print(f"\nBuffer统计: {buffer.get_statistics()}")

    # 测试关键边界位置
    test_positions = [1, 2, 3, 4, 5, 6, 7]  # 跨越不同episode边界

    for pos in test_positions:
        if pos < buffer.size:
            chunk = buffer._get_action_chunk(pos, 4)
            reward = buffer._get_multi_step_reward(pos, 4, gamma=1.0)
            print(f"\n从索引{pos}开始:")
            print(f"  Action chunk: {chunk.flatten()}")
            print(f"  Multi reward: {reward}")

            # 分析期望结果
            if pos == 2:  # Episode 1 末尾
                print(f"  分析: Episode 1结束，应该重复action=2")
            elif pos == 3:  # Episode 2 开始
                print(f"  分析: Episode 2开始，不应跨越到Episode 1")
            elif pos == 4:  # Episode 2 末尾
                print(f"  分析: Episode 2结束，应该重复action=11")
            elif pos == 5:  # Episode 3 开始
                print(f"  分析: Episode 3开始，不应跨越到Episode 2")


def test_data_type_consistency():
    """测试数据类型一致性"""
    print("\n\n测试4: 数据类型一致性")
    print("-" * 40)

    buffer = ReplayBuffer(state_dim=3, action_dim=2, max_size=100, action_chunk_size=3, reward_horizon=3)

    # 添加各种数据类型
    test_cases = [
        # (state, action, next_state, reward, done, terminated, truncated)
        (np.array([1, 2, 3], dtype=np.int32), np.array([0.5, -0.3], dtype=np.float64),
         np.array([2, 3, 4], dtype=np.float32), 1, None, True, False),
        (np.array([1.5, 2.5, 3.5]), np.array([0.1, 0.2]),
         np.array([2.5, 3.5, 4.5]), 2.5, None, False, True),
        (np.array([0, 0, 0]), np.array([0, 0]),
         np.array([0, 0, 1]), 0, True, None, None),
    ]

    for i, (state, action, next_state, reward, done, terminated, truncated) in enumerate(test_cases):
        try:
            buffer.add(state, action, next_state, reward, done=done,
                      terminated=terminated, truncated=truncated, episode_start=(i==0))
            print(f"✓ 测试用例{i+1}添加成功")
        except Exception as e:
            print(f"❌ 测试用例{i+1}添加失败: {e}")

    # 验证存储的数据类型
    print("\n验证存储的数据类型:")
    print(f"  States dtype: {buffer.state.dtype}")
    print(f"  Actions dtype: {buffer.action.dtype}")
    print(f"  Next states dtype: {buffer.next_state.dtype}")
    print(f"  Rewards dtype: {buffer.reward.dtype}")
    print(f"  Dones dtype: {buffer.done.dtype}")

    # 测试采样的数据类型
    try:
        states, actions, next_states, rewards, dones = buffer.sample(2)
        print(f"\n采样数据类型:")
        print(f"  States: {states.dtype}, device: {states.device}")
        print(f"  Actions: {actions.dtype}, device: {actions.device}")
        print(f"  Next states: {next_states.dtype}, device: {next_states.device}")
        print(f"  Rewards: {rewards.dtype}, device: {rewards.device}")
        print(f"  Dones: {dones.dtype}, device: {dones.device}")
    except Exception as e:
        print(f"❌ 采样失败: {e}")


def test_parameter_validation():
    """测试参数验证"""
    print("\n\n测试5: 参数验证")
    print("-" * 40)

    # 测试无效参数组合
    try:
        buffer = ReplayBuffer(state_dim=0, action_dim=1, max_size=100)
        print("❌ state_dim=0 应该引起问题")
    except:
        print("✓ state_dim=0 被正确处理")

    try:
        buffer = ReplayBuffer(state_dim=1, action_dim=0, max_size=100)
        print("❌ action_dim=0 应该引起问题")
    except:
        print("✓ action_dim=0 被正确处理")

    try:
        buffer = ReplayBuffer(state_dim=1, action_dim=1, max_size=0)
        print("❌ max_size=0 应该引起问题")
    except:
        print("✓ max_size=0 被正确处理")

    # 测试add方法的参数验证
    buffer = ReplayBuffer(state_dim=2, action_dim=1, max_size=10)

    try:
        # 缺少done参数
        buffer.add(
            np.array([1, 2], dtype=np.float32),
            np.array([1], dtype=np.float32),
            np.array([2, 3], dtype=np.float32),
            1.0
        )
        print("❌ 缺少done参数应该抛出异常")
    except ValueError as e:
        print(f"✓ 缺少done参数正确抛出异常: {e}")

    # 测试维度不匹配
    try:
        buffer.add(
            np.array([1, 2, 3], dtype=np.float32),  # 错误维度
            np.array([1], dtype=np.float32),
            np.array([2, 3], dtype=np.float32),
            1.0,
            done=True
        )
        print("❌ 维度不匹配应该引起问题")
    except Exception as e:
        print(f"✓ 维度不匹配被检测: {type(e).__name__}")


def test_numerical_stability():
    """测试数值稳定性"""
    print("\n\n测试6: 数值稳定性")
    print("-" * 40)

    buffer = ReplayBuffer(state_dim=2, action_dim=1, max_size=100, action_chunk_size=3, reward_horizon=5)

    # 添加极端数值
    extreme_values = [
        (1e-10, 1e10, -1e10),   # 非常小和非常大的数值
        (np.inf, -np.inf, np.nan), # 无限大和NaN
        (0.0, -0.0, 1e-15),     # 零值和极小值
    ]

    for i, (reward, state_val, action_val) in enumerate(extreme_values):
        try:
            # 跳过包含nan和inf的测试，因为它们会导致问题
            if np.isnan(reward) or np.isinf(reward) or np.isnan(state_val) or np.isinf(state_val):
                print(f"跳过极端值测试{i+1}: 包含nan/inf")
                continue

            state = np.array([state_val, state_val + 1], dtype=np.float32)
            action = np.array([action_val], dtype=np.float32)
            next_state = np.array([state_val + 1, state_val + 2], dtype=np.float32)

            buffer.add(state, action, next_state, reward, done=(i==2), episode_start=(i==0))
            print(f"✓ 极端值测试{i+1}添加成功")
        except Exception as e:
            print(f"❌ 极端值测试{i+1}失败: {e}")

    # 测试gamma边界值
    if buffer.size > 0:
        try:
            # gamma = 0
            reward0 = buffer._get_multi_step_reward(0, 3, gamma=0.0)
            print(f"✓ gamma=0.0 计算成功: {reward0}")

            # gamma = 1
            reward1 = buffer._get_multi_step_reward(0, 3, gamma=1.0)
            print(f"✓ gamma=1.0 计算成功: {reward1}")

            # gamma接近1
            reward_close = buffer._get_multi_step_reward(0, 3, gamma=0.9999)
            print(f"✓ gamma=0.9999 计算成功: {reward_close}")

        except Exception as e:
            print(f"❌ gamma边界值测试失败: {e}")


def main():
    """运行所有测试"""
    print("开始全面测试ReplayBuffer")
    print("=" * 50)

    test_circular_buffer_behavior()
    test_edge_cases()
    test_episode_boundary_logic()
    test_data_type_consistency()
    test_parameter_validation()
    test_numerical_stability()

    print("\n" + "=" * 50)
    print("所有测试完成！")


if __name__ == "__main__":
    main()
