import numpy as np
from utils import MultiStepReplayBuffer

def test_sliding_window_concept():
    """测试滑动窗口概念的核心思想"""
    print("测试滑动窗口概念...")
    print("="*50)
    
    # 创建一个简单的episode: 5步，n_step=3
    buffer = MultiStepReplayBuffer(capacity=100, device='cpu', n_step=3)
    
    # 添加5步数据
    for i in range(5):
        state = np.array([i, i])
        action = np.array([i*0.1, i*0.1])
        reward = i
        next_state = np.array([i+1, i+1])
        done = (i == 4)  # 最后一步done=True
        
        buffer.add(state, action, reward, next_state, done)
        print(f"添加步骤 {i+1}: state={state}, action={action}, reward={reward}, done={done}")
    
    print(f"\nEpisode结束，缓冲区大小: {len(buffer)}")
    
    print("\n理论上的滑动窗口样本:")
    print("样本1: s0, a[0:3], r[0:3], s3")
    print("样本2: s1, a[1:4], r[1:4], s4")
    print("样本3: s2, a[2:5], r[2:5], s5 (但a[4],r[4]不存在，需要填充)")
    
    print("\n实际生成的样本:")
    for i in range(min(3, len(buffer))):  # 只显示前3个样本
        states, actions, rewards, next_states, dones, actual_steps, early_term = buffer.sample(1)
        print(f"\n样本 {i+1}:")
        print(f"  起始状态: {states[0]}")
        print(f"  动作序列: {actions[0]}")
        print(f"  奖励序列: {rewards[0]}")
        print(f"  最终状态: {next_states[0]}")
        print(f"  done序列: {dones[0]}")
        print(f"  实际步数: {actual_steps[0]}")
        print(f"  早期终止: {early_term[0]}")

def test_data_utilization_comparison():
    """对比不同方法的数据利用率"""
    print("\n" + "="*50)
    print("数据利用率对比测试")
    print("="*50)
    
    episode_lengths = [5, 8, 10]
    n_steps = [3, 4, 5]
    
    for ep_len in episode_lengths:
        for n_step in n_steps:
            if n_step < ep_len:
                # 原始方法：每个episode只生成1个样本
                original_samples = 1
                
                # 滑动窗口方法：理论上应该生成的样本数
                # 从位置0到位置(ep_len-1)，每个位置都可以作为起始点
                # 但需要考虑是否有足够的步数
                sliding_samples = 0
                for start_pos in range(ep_len):
                    remaining_steps = ep_len - start_pos
                    if remaining_steps > 0:  # 至少有1步可以收集
                        sliding_samples += 1
                
                improvement = ((sliding_samples - original_samples) / original_samples) * 100
                
                print(f"Episode长度={ep_len}, n_step={n_step}:")
                print(f"  原始方法: {original_samples} 样本")
                print(f"  滑动窗口: {sliding_samples} 样本")
                print(f"  提升率: {improvement:.1f}%")
                print()

def test_actual_implementation():
    """测试实际实现的数据利用率"""
    print("="*50)
    print("测试实际实现")
    print("="*50)
    
    # 测试5步episode，n_step=3
    buffer = MultiStepReplayBuffer(capacity=100, device='cpu', n_step=3)
    
    # 添加5步数据
    for i in range(5):
        state = np.array([i, i])
        action = np.array([i*0.1, i*0.1])
        reward = i
        next_state = np.array([i+1, i+1])
        done = (i == 4)
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"实际生成的样本数: {len(buffer)}")
    print(f"期望的样本数: 5 (每个起始位置一个)")
    
    if len(buffer) == 5:
        print("✓ 滑动窗口实现正确！")
    else:
        print("✗ 滑动窗口实现有问题")

if __name__ == "__main__":
    test_sliding_window_concept()
    test_data_utilization_comparison()
    test_actual_implementation()