import numpy as np
import time
import torch
from utils import MultiStepReplayBuffer

def performance_test():
    """性能测试：比较优化前后的处理速度"""
    print("=== MultiStepReplayBuffer 性能测试 ===")
    
    # 测试参数
    capacity = 10000
    n_step = 4
    device = torch.device('cpu')
    num_episodes = 100
    episode_length = 50
    
    # 创建缓冲区
    buffer = MultiStepReplayBuffer(capacity, device, n_step)
    
    print(f"测试配置:")
    print(f"  容量: {capacity}")
    print(f"  n_step: {n_step}")
    print(f"  episode数量: {num_episodes}")
    print(f"  每个episode长度: {episode_length}")
    print(f"  预期生成样本数: {num_episodes * episode_length}")
    
    # 测试数据添加性能
    print("\n--- 数据添加性能测试 ---")
    start_time = time.time()
    
    for episode in range(num_episodes):
        for step in range(episode_length):
            state = np.random.randn(4)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = (step == episode_length - 1)  # 最后一步为done
            
            buffer.add(state, action, reward, next_state, done)
    
    add_time = time.time() - start_time
    print(f"数据添加耗时: {add_time:.4f}秒")
    print(f"缓冲区大小: {len(buffer)}")
    print(f"episode数量: {buffer.get_episode_count()}")
    print(f"平均每个episode处理时间: {add_time/num_episodes:.6f}秒")
    
    # 测试采样性能
    print("\n--- 采样性能测试 ---")
    batch_sizes = [32, 64, 128, 256]
    
    for batch_size in batch_sizes:
        if len(buffer) >= batch_size:
            start_time = time.time()
            
            # 进行多次采样测试
            num_samples = 100
            for _ in range(num_samples):
                states, actions, rewards, next_states, dones, actual_steps, early_termination = buffer.sample(batch_size)
            
            sample_time = time.time() - start_time
            avg_sample_time = sample_time / num_samples
            
            print(f"  批次大小 {batch_size}: 平均采样时间 {avg_sample_time:.6f}秒")
    
    # 测试内存效率
    print("\n--- 内存效率分析 ---")
    print(f"  多步样本数量: {len(buffer.multi_step_data)}")
    print(f"  完成的episode数量: {len(buffer.completed_episodes)}")
    print(f"  内存利用率: {len(buffer.multi_step_data)/capacity*100:.2f}%")
    
    # 验证数据质量
    print("\n--- 数据质量验证 ---")
    if len(buffer) >= 10:
        states, actions, rewards, next_states, dones, actual_steps, early_termination = buffer.sample(10)
        
        print(f"  采样样本数: {len(states)}")
        print(f"  actual_steps范围: {actual_steps.min().item()} - {actual_steps.max().item()}")
        print(f"  early_termination比例: {early_termination.float().mean().item():.2f}")
        
        # 检查变长序列
        action_lengths = [len(action_seq) for action_seq in actions]
        reward_lengths = [len(reward_seq) for reward_seq in rewards]
        done_lengths = [len(done_seq) for done_seq in dones]
        
        print(f"  action序列长度范围: {min(action_lengths)} - {max(action_lengths)}")
        print(f"  reward序列长度范围: {min(reward_lengths)} - {max(reward_lengths)}")
        print(f"  done序列长度范围: {min(done_lengths)} - {max(done_lengths)}")
        
        # 验证长度一致性
        length_consistent = all(
            len(actions[i]) == len(rewards[i]) == len(dones[i]) == actual_steps[i].item()
            for i in range(len(actions))
        )
        print(f"  长度一致性: {'✓ 通过' if length_consistent else '✗ 失败'}")
    
    print("\n=== 性能测试完成 ===")

def stress_test():
    """压力测试：测试大容量下的性能"""
    print("\n=== 压力测试 ===")
    
    capacity = 50000
    n_step = 8
    device = torch.device('cpu')
    
    buffer = MultiStepReplayBuffer(capacity, device, n_step)
    
    print(f"压力测试配置: 容量={capacity}, n_step={n_step}")
    
    start_time = time.time()
    
    # 添加大量数据
    for episode in range(500):
        episode_len = np.random.randint(10, 100)  # 随机episode长度
        
        for step in range(episode_len):
            state = np.random.randn(8)
            action = np.random.randn(4)
            reward = np.random.randn()
            next_state = np.random.randn(8)
            done = (step == episode_len - 1)
            
            buffer.add(state, action, reward, next_state, done)
        
        if episode % 100 == 0:
            print(f"  已处理 {episode} episodes, 缓冲区大小: {len(buffer)}")
    
    total_time = time.time() - start_time
    print(f"压力测试完成: {total_time:.4f}秒")
    print(f"最终缓冲区大小: {len(buffer)}")
    print(f"episode数量: {buffer.get_episode_count()}")
    
if __name__ == "__main__":
    performance_test()
    stress_test()