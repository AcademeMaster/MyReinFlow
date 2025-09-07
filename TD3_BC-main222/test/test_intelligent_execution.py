#!/usr/bin/env python3
"""
测试智能化多步action执行的脚本
"""

import numpy as np
import torch
import argparse
import minari
import TD3_BC


def test_intelligent_multi_step_execution():
    """测试智能化多步action执行功能"""
    print("=" * 60)
    print("测试智能化多步action执行功能")
    print("=" * 60)
    
    # 环境设置
    env_name = "mujoco/pusher/expert-v0"
    horizon = 4  # 4步执行一次推理

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
    print(f"执行Horizon: {horizon}")
    print("-" * 60)

    # 创建策略
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "horizon": horizon,  # 设置多步执行
        "train_mode": "offline"
    }

    policy = TD3_BC.TD3_BC(**kwargs)

    # 测试智能化执行
    print("开始测试智能化多步执行:")
    print(f"设置: 每 {horizon} 步执行一次推理")

    state, _ = env.reset()
    if isinstance(state, tuple):
        state_obs = state[0]
    else:
        state_obs = state
    state_array = np.asarray(state_obs, dtype=np.float32).reshape(1, -1)

    # 模拟执行多个步骤
    total_steps = 15  # 执行15步，应该触发4次推理 (步骤1-4, 5-8, 9-12, 13-15)
    actions = []

    print(f"\n开始执行 {total_steps} 个步骤:")

    for step in range(total_steps):
        print(f"\n--- 步骤 {step + 1} ---")

        # 调用select_action，内部会自动管理推理时机
        action = policy.select_action(state_array, n_step=5)  # 使用5步推理
        actions.append(action)

        print(f"获得action: {action[:3]}... (显示前3个值)")

        # 模拟状态变化（实际中这会是环境step的结果）
        # 这里我们只是稍微修改state来模拟
        state_array = state_array + np.random.normal(0, 0.01, state_array.shape)

    print(f"\n总共执行了 {len(actions)} 个动作")
    print("根据horizon=4的设置，应该触发了4次推理")

    # 测试动态修改horizon
    print(f"\n{'='*60}")
    print("测试动态修改horizon功能")
    print(f"{'='*60}")

    print("当前horizon:", policy.horizon)
    print("修改horizon为2...")
    policy.set_horizon(2)
    print("新horizon:", policy.horizon)

    print("\n继续执行6步，应该触发3次推理:")
    for step in range(6):
        print(f"\n--- 额外步骤 {step + 1} ---")
        action = policy.select_action(state_array, n_step=3)
        print(f"获得action: {action[:3]}...")

    # 测试动态修改推理步数
    print(f"\n{'='*60}")
    print("测试动态修改推理步数功能")
    print(f"{'='*60}")

    print("当前推理步数:", policy.n_step)
    print("修改推理步数为10...")
    policy.set_inference_steps(10)
    print("新推理步数:", policy.n_step)

    print("\n继续执行2步，使用新的推理步数:")
    for step in range(2):
        print(f"\n--- 高精度步骤 {step + 1} ---")
        action = policy.select_action(state_array)  # 使用默认n_step=10
        print(f"获得action: {action[:3]}...")

    # 测试重置缓存功能
    print(f"\n{'='*60}")
    print("测试重置缓存功能")
    print(f"{'='*60}")

    print("当前缓存状态 - count:", policy.count)
    print("手动重置缓存...")
    policy.reset_action_cache()
    print("重置后 - count:", policy.count, "cached_actions:", policy.cached_actions)

    print("\n执行一步，应该触发重新推理:")
    action = policy.select_action(state_array, n_step=1)
    print(f"获得action: {action[:3]}...")

    print(f"\n{'='*60}")
    print("智能化多步action执行测试完成!")
    print("功能总结:")
    print("✓ 自动管理推理时机 - 当执行步数达到horizon时自动重新推理")
    print("✓ 统一的外部接口 - select_action始终返回单步action")
    print("✓ 动态配置 - 可以运行时修改horizon和推理步数")
    print("✓ 缓存管理 - 支持手动重置缓存以强制重新推理")
    print(f"{'='*60}")


def test_comparison_with_traditional():
    """比较智能化执行和传统执行的差异"""
    print("=" * 60)
    print("比较智能化执行vs传统执行")
    print("=" * 60)

    # 环境设置
    env_name = "mujoco/pusher/expert-v0"
    minari_dataset = minari.load_dataset(env_name)
    env = minari_dataset.recover_environment()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 创建两个策略实例用于比较
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "train_mode": "offline"
    }

    # 智能化策略 (horizon=3)
    smart_policy = TD3_BC.TD3_BC(horizon=3, **kwargs)

    # 传统策略 (horizon=1，每次都推理)
    traditional_policy = TD3_BC.TD3_BC(horizon=1, **kwargs)

    state, _ = env.reset()
    if isinstance(state, tuple):
        state_obs = state[0]
    else:
        state_obs = state
    state_array = np.asarray(state_obs, dtype=np.float32).reshape(1, -1)

    steps = 9

    print(f"执行 {steps} 个步骤的比较:")
    print("\n智能化策略 (horizon=3, 每3步推理一次):")
    smart_actions = []
    for i in range(steps):
        print(f"步骤 {i+1}:")
        action = smart_policy.select_action(state_array, n_step=2)
        smart_actions.append(action)

    print(f"\n传统策略 (horizon=1, 每步都推理):")
    traditional_actions = []
    for i in range(steps):
        print(f"步骤 {i+1}:")
        action = traditional_policy.select_action(state_array, n_step=2)
        traditional_actions.append(action)

    print(f"\n结果分析:")
    print(f"- 智能化策略推理次数: {steps // 3 + (1 if steps % 3 > 0 else 0)} 次")
    print(f"- 传统策略推理次数: {steps} 次")
    print(f"- 计算效率提升: {steps / (steps // 3 + (1 if steps % 3 > 0 else 0)):.1f}x")

    print(f"\n{'='*60}")
    print("比较测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all", choices=["intelligent", "comparison", "all"],
                       help="选择测试类型")
    args = parser.parse_args()

    if args.test in ["intelligent", "all"]:
        test_intelligent_multi_step_execution()

    if args.test in ["comparison", "all"]:
        test_comparison_with_traditional()
