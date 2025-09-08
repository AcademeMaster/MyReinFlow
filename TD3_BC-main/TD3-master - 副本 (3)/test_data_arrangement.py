import numpy as np
import torch
from utils import MultiStepReplayBuffer

def test_data_arrangement():
    """测试数据排列的正确性"""
    print("=== 测试数据排列正确性 ===")
    
    # 创建缓冲区
    capacity = 1000
    device = torch.device('cpu')
    n_step = 4
    buffer = MultiStepReplayBuffer(capacity, device, n_step)
    
    print(f"配置: n_step={n_step}")
    print("\n原始episode数据:")
    
    # 按照用户提供的数据格式添加数据
    episode_data = [
        ([1], [1], 1, [2], False),   # s1=[1], a1=[1], r1=1, next_s1=[2], done=0
        ([2], [2], 2, [3], False),   # s2=[2], a2=[2], r2=1, next_s2=[3], done=0
        ([3], [3], 3, [4], False),   # s3=[3], a3=[3], r3=1, next_s3=[4], done=0
        ([4], [4], 4, [5], False),   # s4=[4], a4=[4], r4=1, next_s4=[5], done=0
        ([5], [5], 5, [6], False),   # s5=[5], a5=[5], r5=1, next_s5=[6], done=0
        ([6], [6], 6, [7], False),   # s6=[6], a6=[6], r6=1, next_s6=[7], done=0
        ([7], [7], 7, [8], False),   # s7=[7], a7=[7], r7=1, next_s7=[8], done=0
        ([8], [8], 8, [9], False),   # s8=[8], a8=[8], r8=1, next_s8=[9], done=0
        ([9], [9], 9, [10], False),  # s9=[9], a9=[9], r9=1, next_s9=[10], done=0
        ([10], [10], 10, [11], True)  # s10=[10], a10=[10], r10=1, next_s10=[11], done=1
    ]
    
    # 打印原始数据
    for i, (state, action, reward, next_state, done) in enumerate(episode_data, 1):
        print(f"  步骤{i}: s{i}={state}, a{i}={action}, r{i}={reward}, next_s{i}={next_state}, done={int(done)}")
    
    # 添加数据到缓冲区
    for state, action, reward, next_state, done in episode_data:
        buffer.add(np.array(state), np.array(action), reward, np.array(next_state), done)
    
    print(f"\nEpisode处理完成，缓冲区大小: {len(buffer)}")
    
    # 理论分析：应该生成的多步样本
    print("\n理论上应该生成的多步样本 (n_step=4):")
    episode_len = len(episode_data)
    
    # 找到done=True的位置
    done_position = None
    for idx, (_, _, _, _, done) in enumerate(episode_data):
        if done:
            done_position = idx
            break
    
    # 只生成到done=True位置的样本
    max_start_idx = done_position if done_position is not None else episode_len - 1
    
    for i in range(max_start_idx + 1):  # 包含done=True的那一步
        remaining_steps = episode_len - i
        max_steps = min(n_step, remaining_steps)
        
        # 检查是否有early termination
        actual_steps = max_steps
        early_term = False
        for j in range(max_steps):
            if episode_data[i + j][4]:  # done=True
                actual_steps = j + 1
                early_term = True
                break
        
        start_idx = i + 1  # 从1开始编号
        
        actions = [episode_data[i + j][1][0] for j in range(actual_steps)]
        rewards = [episode_data[i + j][2] for j in range(actual_steps)]
        final_done = episode_data[i + actual_steps - 1][4]  # 只有最后一个done值
        
        print(f"  样本{i}: state=s{start_idx}={episode_data[i][0]}, actions={actions}, rewards={rewards}, next_state={episode_data[i + actual_steps - 1][3]}, done={final_done}")
        print(f"         actual_steps={actual_steps}, early_termination={early_term}")
        
        # 如果当前样本包含done=True，则停止生成更多样本
        if final_done:
            print("\n  >>> Episode结束 (done=True)，停止生成更多样本")
            break
    
    # 实际采样验证
    print("\n实际采样结果:")
    
    # 采样所有数据进行验证
    sample_size = min(len(buffer), 10)  # 最多采样10个
    if sample_size > 0:
        states, actions, rewards, next_states, dones = buffer.sample(sample_size)
        
        for i in range(sample_size):
            print(f"\n样本{i}:")
            print(f"  state: {states[i].cpu().numpy()}")
            print(f"  actions: {[a.item() if hasattr(a, 'item') else float(a) for a in actions[i]]} (长度: {len(actions[i])})")
            print(f"  rewards: {rewards[i]} (长度: {len(rewards[i])})")
            print(f"  next_state: {next_states[i].cpu().numpy()}")
            print(f"  done: {dones[i]}")

    
    # 验证数据一致性
    print("\n=== 数据一致性验证 ===")
    
    if sample_size > 0:
        all_consistent = True
        
        for i in range(sample_size):

            action_len = len(actions[i])
            reward_len = len(rewards[i])
            # dones现在是单个值，不需要检查长度
            consistent = (action_len == reward_len)
            
            if not consistent:
                print(f"  ✗ 样本{i}: 长度不一致 - actions:{action_len}, rewards:{reward_len}")
                all_consistent = False
            else:
                print(f"  ✓ 样本{i}: 长度一致 ({action_len})")
        
        if all_consistent:
            print("\n✓ 所有样本数据排列正确！")
        else:
            print("\n✗ 发现数据排列问题！")
    
    print("\n=== 测试完成 ===")

def test_multiple_episodes():
    """测试多个episode的数据排列"""
    print("\n=== 测试多个Episode ===")
    
    capacity = 1000
    device = torch.device('cpu')
    n_step = 3
    buffer = MultiStepReplayBuffer(capacity, device, n_step)
    
    print(f"配置: n_step={n_step}")
    
    # Episode 1: 短episode (3步)
    print("\nEpisode 1 (3步):")
    episode1 = [
        ([1], [1], 10, [2], False),
        ([2], [2], 20, [3], False),
        ([3], [3], 30, [4], True)
    ]
    
    for i, (state, action, reward, next_state, done) in enumerate(episode1, 1):
        print(f"  步骤{i}: s={state}, a={action}, r={reward}, next_s={next_state}, done={int(done)}")
        buffer.add(np.array(state), np.array(action), reward, np.array(next_state), done)
    
    # Episode 2: 长episode (5步)
    print("\nEpisode 2 (5步):")
    episode2 = [
        ([10], [10], 100, [20], False),
        ([20], [20], 200, [30], False),
        ([30], [30], 300, [40], False),
        ([40], [40], 400, [50], False),
        ([50], [50], 500, [60], True)
    ]
    
    for i, (state, action, reward, next_state, done) in enumerate(episode2, 1):
        print(f"  步骤{i}: s={state}, a={action}, r={reward}, next_s={next_state}, done={int(done)}")
        buffer.add(np.array(state), np.array(action), reward, np.array(next_state), done)
    
    print(f"\n两个episode处理完成，缓冲区大小: {len(buffer)}")
    print(f"Episode数量: {buffer.get_episode_count()}")
    
    # 采样验证
    if len(buffer) >= 5:
        print("\n采样验证 (5个样本):")
        states, actions, rewards, next_states, dones = buffer.sample(5)
        
        for i in range(5):
            print(f"\n样本{i}:")
            print(f"  state: {states[i].cpu().numpy()}")
            print(f"  actions: {[a.item() if hasattr(a, 'item') else float(a) for a in actions[i]]}")
            print(f"  rewards: {rewards[i]}")
            print(f"  next_state: {next_states[i].cpu().numpy()}")
            print(f"  done: {dones[i]}")


if __name__ == "__main__":
    test_data_arrangement()
    test_multiple_episodes()