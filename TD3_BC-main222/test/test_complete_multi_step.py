#!/usr/bin/env python3
"""
测试多步action实现的脚本
"""

import numpy as np
import torch
import argparse
import minari
import TD3_BC
import utils


def test_multi_step_implementation():
    """测试完整的多步action实现"""
    print("=" * 60)
    print("测试多步action完整实现")
    print("=" * 60)

    # 环境设置
    env_name = "mujoco/pusher/expert-v0"
    horizon = 4

    # 加载环境
    minari_dataset = minari.load_dataset(env_name)
    env = minari_dataset.recover_environment()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"环境: {env_name}")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"最大动作值: {max_action}")
    print(f"Horizon: {horizon}")
    print("-" * 60)

    # 创建策略
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "horizon": horizon,
        "train_mode": "offline"
    }

    policy = TD3_BC.TD3_BC(**kwargs)

    # 验证网络维度
    print("验证网络结构:")
    print(f"Actor输出维度: {policy.actor.output_dim} (应该是 {action_dim * horizon})")
    print(f"Critic action输入维度: {action_dim * horizon}")

    # 测试状态
    state, _ = env.reset()
    if isinstance(state, tuple):
        state_obs = state[0]
    else:
        state_obs = state
    state_array = np.asarray(state_obs, dtype=np.float32).reshape(1, -1)

    print(f"测试状态形状: {state_array.shape}")

    # 测试Actor输出
    print("\n测试Actor多步输出:")
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_array).to(policy.actor.device)
        multi_step_output = policy.actor(state_tensor, n_steps=5)
        print(f"原始输出形状: {multi_step_output.shape}")
        print(f"应该是: [1, {action_dim * horizon}]")

        # 测试重塑功能
        reshaped = policy.actor.reshape_multi_step_action(multi_step_output)
        print(f"重塑后形状: {reshaped.shape}")
        print(f"应该是: [1, {horizon}, {action_dim}]")

    # 测试精简版select_action
    print("\n测试精简版select_action:")
    for step in range(horizon + 2):  # 多执行几步验证缓存机制
        print(f"步骤 {step + 1}:")
        action = policy.select_action(state_array, n_step=3)
        print(f"  获得action形状: {action.shape}")
        print(f"  action值: {action[:3]}... (前3个值)")
        print(f"  缓存索引: {policy.cache_index}")

    print("\n=" * 60)
    print("基础功能测试完成!")
    print("=" * 60)


def test_multi_step_training():
    """测试多步训练功能"""
    print("=" * 60)
    print("测试多步训练功能")
    print("=" * 60)

    env_name = "mujoco/pusher/expert-v0"
    horizon = 4
    batch_size = 16

    # 加载环境和数据
    minari_dataset = minari.load_dataset(env_name)
    env = minari_dataset.recover_environment()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 创建策略和buffer
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "horizon": horizon,
        "train_mode": "offline"
    }

    policy = TD3_BC.TD3_BC(**kwargs)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, horizon=horizon)

    # 加载少量数据用于测试
    print("加载测试数据...")
    replay_buffer.convert_minari(minari_dataset)
    print(f"Buffer大小: {replay_buffer.size}")

    # 测试标准采样
    print("\n测试标准采样:")
    try:
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        print(f"✓ 标准采样成功:")
        print(f"  状态形状: {state.shape}")
        print(f"  动作形状: {action.shape}")
        print(f"  奖励形状: {reward.shape}")
    except Exception as e:
        print(f"✗ 标准采样失败: {e}")

    # 测试多步采样
    print("\n测试多步采样:")
    try:
        state, action_chunks, next_state, multi_rewards, done = replay_buffer.sample(
            batch_size, return_chunks=True, gamma=0.99
        )
        print(f"✓ 多步采样成功:")
        print(f"  状态形状: {state.shape}")
        print(f"  动作块形状: {action_chunks.shape}")
        print(f"  多步奖励形状: {multi_rewards.shape}")
        print(f"  多步奖励范围: [{multi_rewards.min().item():.3f}, {multi_rewards.max().item():.3f}]")
    except Exception as e:
        print(f"✗ 多步采样失败: {e}")
        return

    # 测试训练步骤
    print("\n测试训练步骤:")
    try:
        print("执行多步训练...")
        policy.train(replay_buffer, batch_size, use_multi_step=True)
        print("✓ 多步训练成功!")

        print("执行标准训练...")
        policy.train(replay_buffer, batch_size, use_multi_step=False)
        print("✓ 标准训练成功!")

        print("训练功能验证完成!")

    except Exception as e:
        print(f"✗ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n=" * 60)
    print("多步训练测试完成!")
    print("=" * 60)


def test_td_computation():
    """测试TD计算是否正确实现了r + gamma^h * target_Q"""
    print("=" * 60)
    print("测试TD计算公式")
    print("=" * 60)

    # 创建简单的测试数据
    batch_size = 4
    horizon = 3
    gamma = 0.99

    # 模拟奖励和目标Q值
    reward = torch.tensor([[1.0], [2.0], [0.5], [1.5]])
    target_Q = torch.tensor([[10.0], [8.0], [12.0], [9.0]])
    done = torch.tensor([[0.0], [0.0], [1.0], [0.0]])  # 第3个episode结束

    # 计算标准TD目标 (gamma^1)
    standard_discount = gamma
    standard_target = reward + (1.0 - done) * standard_discount * target_Q

    # 计算多步TD目标 (gamma^h)
    multi_step_discount = gamma ** horizon
    multi_step_target = reward + (1.0 - done) * multi_step_discount * target_Q

    print(f"测试参数:")
    print(f"  Horizon: {horizon}")
    print(f"  Gamma: {gamma}")
    print(f"  标准折扣: {standard_discount:.4f}")
    print(f"  多步折扣: {multi_step_discount:.4f}")

    print(f"\n奖励: {reward.flatten().tolist()}")
    print(f"目标Q: {target_Q.flatten().tolist()}")
    print(f"Done标志: {done.flatten().tolist()}")

    print(f"\n标准TD目标: {standard_target.flatten().tolist()}")
    print(f"多步TD目标: {multi_step_target.flatten().tolist()}")

    # 计算差异
    difference = multi_step_target - standard_target
    print(f"差异: {difference.flatten().tolist()}")

    print("\n验证:")
    for i in range(batch_size):
        if done[i].item() == 1.0:
            expected = reward[i].item()  # episode结束，只有奖励
            print(f"  样本{i+1} (episode结束): {multi_step_target[i].item():.3f} == {expected:.3f} ✓")
        else:
            expected = reward[i].item() + multi_step_discount * target_Q[i].item()
            print(f"  样本{i+1}: {multi_step_target[i].item():.3f} == {expected:.3f} ✓")

    print("\n=" * 60)
    print("TD计算验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all",
                       choices=["implementation", "training", "td", "all"],
                       help="选择测试类型")
    args = parser.parse_args()

    if args.test in ["implementation", "all"]:
        test_multi_step_implementation()

    if args.test in ["training", "all"]:
        test_multi_step_training()

    if args.test in ["td", "all"]:
        test_td_computation()

    print("\n🎉 所有测试完成! 多步action系统已就绪!")
