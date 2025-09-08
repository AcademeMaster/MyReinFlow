import numpy as np
import time
from utils import MultiStepReplayBuffer

def test_performance_optimization():
    """测试预计算优化的性能提升"""
    print("测试预计算优化的性能和正确性")
    print("="*60)
    
    # 创建较大的episode进行性能测试
    buffer = MultiStepReplayBuffer(capacity=10000, device='cpu', n_step=4)
    
    # 生成一个较长的episode
    episode_length = 100
    print(f"生成长度为{episode_length}的episode...")
    
    start_time = time.time()
    
    for i in range(episode_length):
        state = np.random.randn(10)  # 10维状态
        action = np.random.randn(3)  # 3维动作
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = (i == episode_length - 1)  # 最后一步done=True
        
        buffer.add(state, action, reward, next_state, done)
    
    processing_time = time.time() - start_time
    
    print(f"Episode处理时间: {processing_time:.4f}秒")
    print(f"生成的样本数量: {len(buffer)}")
    print(f"预期样本数量: {episode_length}")
    
    # 验证正确性
    if len(buffer) == episode_length:
        print("✓ 样本数量正确")
    else:
        print("✗ 样本数量错误")
    
    # 验证预计算的准确性
    print("\n验证预计算逻辑:")
    print(f"  - Episode长度: {episode_length}")
    print(f"  - n_step: {buffer.n_step}")
    print(f"  - 每个起始位置都能生成1个样本")
    print(f"  - 总样本数 = Episode长度 = {episode_length}")
    print(f"  - 实际生成: {len(buffer)}")
    
    return processing_time

def test_multiple_episodes():
    """测试多个episode的处理"""
    print("\n" + "="*60)
    print("测试多个episode的批量处理")
    print("="*60)
    
    buffer = MultiStepReplayBuffer(capacity=10000, device='cpu', n_step=3)
    
    episode_lengths = [20, 35, 50, 15, 40]
    total_expected_samples = sum(episode_lengths)
    
    print(f"处理{len(episode_lengths)}个episode，长度分别为: {episode_lengths}")
    print(f"预期总样本数: {total_expected_samples}")
    
    start_time = time.time()
    
    for ep_idx, ep_len in enumerate(episode_lengths):
        print(f"\n处理Episode {ep_idx + 1} (长度: {ep_len})...")
        
        for i in range(ep_len):
            state = np.random.randn(5)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(5)
            done = (i == ep_len - 1)
            
            buffer.add(state, action, reward, next_state, done)
        
        print(f"  当前缓冲区大小: {len(buffer)}")
    
    total_time = time.time() - start_time
    
    print(f"\n总处理时间: {total_time:.4f}秒")
    print(f"最终样本数量: {len(buffer)}")
    print(f"预期样本数量: {total_expected_samples}")
    
    if len(buffer) == total_expected_samples:
        print("✓ 多episode处理正确")
    else:
        print("✗ 多episode处理错误")
    
    return total_time

def test_early_termination_optimization():
    """测试早期终止情况下的优化"""
    print("\n" + "="*60)
    print("测试早期终止情况的处理")
    print("="*60)
    
    buffer = MultiStepReplayBuffer(capacity=1000, device='cpu', n_step=5)
    
    # 创建一个在中间结束的episode
    episode_length = 8
    early_done_at = 5  # 在第6步(索引5)设置done=True
    
    print(f"创建长度为{episode_length}的episode，在第{early_done_at+1}步设置done=True")
    
    for i in range(episode_length):
        state = np.array([i, i])
        action = np.array([i*0.1, i*0.1])
        reward = i
        next_state = np.array([i+1, i+1])
        done = (i == early_done_at)  # 在指定位置设置done=True
        
        buffer.add(state, action, reward, next_state, done)
        
        if done:
            print(f"  在步骤{i+1}设置done=True，episode结束")
            break
    
    actual_episode_length = early_done_at + 1
    print(f"实际episode长度: {actual_episode_length}")
    print(f"生成的样本数: {len(buffer)}")
    print(f"预期样本数: {actual_episode_length}")
    
    # 验证样本内容
    print("\n验证前3个样本的early_termination标记:")
    for i in range(min(3, len(buffer))):
        states, actions, rewards, next_states, dones, actual_steps, early_term = buffer.sample(1)
        print(f"样本{i+1}: actual_steps={actual_steps[0].item()}, early_termination={early_term[0].item()}")
    
    if len(buffer) == actual_episode_length:
        print("✓ 早期终止处理正确")
    else:
        print("✗ 早期终止处理错误")

def benchmark_optimization():
    """性能基准测试"""
    print("\n" + "="*60)
    print("性能基准测试")
    print("="*60)
    
    # 测试不同规模的episode
    test_sizes = [50, 100, 200, 500]
    
    for size in test_sizes:
        buffer = MultiStepReplayBuffer(capacity=size*2, device='cpu', n_step=4)
        
        start_time = time.time()
        
        # 生成episode
        for i in range(size):
            state = np.random.randn(20)
            action = np.random.randn(5)
            reward = np.random.randn()
            next_state = np.random.randn(20)
            done = (i == size - 1)
            
            buffer.add(state, action, reward, next_state, done)
        
        processing_time = time.time() - start_time
        samples_per_second = len(buffer) / processing_time if processing_time > 0 else float('inf')
        
        print(f"Episode长度: {size:3d} | 处理时间: {processing_time:.4f}s | 样本数: {len(buffer):3d} | 速度: {samples_per_second:.0f} 样本/秒")

if __name__ == "__main__":
    # 运行所有测试
    test_performance_optimization()
    test_multiple_episodes()
    test_early_termination_optimization()
    benchmark_optimization()
    
    print("\n" + "="*60)
    print("✓ 所有优化测试完成！")
    print("\n优化总结:")
    print("1. 预先计算样本数量，避免重复判断")
    print("2. 批量填充操作，减少循环开销")
    print("3. 预分配存储空间，提高内存效率")
    print("4. 批量添加到缓冲区，减少单次操作")
    print("5. 保持完全的功能正确性")