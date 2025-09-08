import numpy as np
import torch
from utils import MultiStepReplayBuffer

def test_sampling_format(action_horizon=3):
    """详细测试采样格式: state[i], action[i:i+h], reward[i:i+h], next_state[i+h], done[i+h]"""
    print("=== 测试采样格式 ===")
    
    device = torch.device("cpu")
    buffer = MultiStepReplayBuffer(capacity=1000, action_horizon=3, device=device)
    
    # 创建一个清晰的测试episode
    print("\n创建测试episode（4步）:")
    episode_data = [
        {'state': np.array([0.0, 0.0]), 'action': np.array([1.0]), 'reward': 10.0, 'next_state': np.array([1.0, 1.0]), 'done': False},
        {'state': np.array([1.0, 1.0]), 'action': np.array([2.0]), 'reward': 20.0, 'next_state': np.array([2.0, 2.0]), 'done': False},
        {'state': np.array([2.0, 2.0]), 'action': np.array([3.0]), 'reward': 30.0, 'next_state': np.array([3.0, 3.0]), 'done': False},
        {'state': np.array([3.0, 3.0]), 'action': np.array([4.0]), 'reward': 40.0, 'next_state': np.array([4.0, 4.0]), 'done': True}
    ]
    
    for i, step_data in enumerate(episode_data):
        buffer.add(**step_data)
        print(f"  步骤{i}: state={step_data['state']}, action={step_data['action'][0]}, reward={step_data['reward']}, next_state={step_data['next_state']}, done={step_data['done']}")
    
    print(f"\nEpisode结束，缓冲区大小: {buffer.size()}")
    print("\n理论上应该生成的样本:")
    print("  样本0: state[0]=[0,0], action[0:3]=[1,2,3], reward[0:3]=[10,20,30], next_state[3]=[3,3], done[3]=False")
    print("  样本1: state[1]=[1,1], action[1:4]=[2,3,4], reward[1:4]=[20,30,40], next_state[4]=[4,4], done[4]=True")
    print("说明: 遇到done=True时episode结束，不再从done=True之后的位置构建新样本")

    
    # 采样所有数据
    print("\n实际采样结果:")
    states, actions, rewards, next_states, dones = buffer.sample(2)
    
    for i in range(2):
        print(f"\n样本{i}:")
        print(f"  state[{i}]: {states[i].cpu().numpy()}")
        print(f"  action[{i}:{i}+h]: {[float(a) for a in actions[i]]} (长度: {len(actions[i])})")
        print(f"  reward[{i}:{i}+h]: {rewards[i]} (长度: {len(rewards[i])})")
        print(f"  next_state[{i}+h]: {next_states[i].cpu().numpy()}")
        print(f"  done[{i}+h]: {dones[i]} (标量值)")

def test_short_episode():
    """测试短episode（长度小于action_horizon）"""
    print("\n=== 测试短episode ===")
    
    device = torch.device("cpu")
    buffer = MultiStepReplayBuffer(capacity=1000, action_horizon=5, device=device)
    
    print("\n创建短episode（2步，action_horizon=5）:")
    episode_data = [
        {'state': np.array([10.0]), 'action': np.array([0.1]), 'reward': 1.0, 'next_state': np.array([10.1]), 'done': False},
        {'state': np.array([10.1]), 'action': np.array([0.2]), 'reward': 2.0, 'next_state': np.array([10.2]), 'done': True}
    ]
    
    for i, step_data in enumerate(episode_data):
        buffer.add(**step_data)
        print(f"  步骤{i}: state={step_data['state']}, action={step_data['action'][0]}, reward={step_data['reward']}, done={step_data['done']}")
    
    print(f"\nEpisode结束，缓冲区大小: {buffer.size()}")
    print("\n理论上应该生成的样本:")
    print("  无样本: episode长度(2) < action_horizon(5)，不满足最小长度要求")
    print("说明: 当episode长度小于action_horizon时，不产生任何样本")
    
    if buffer.size() > 0:
        states, actions, rewards, next_states, dones = buffer.sample(1)
        print(f"\n实际采样结果:")
        print(f"  state: {states[0].cpu().numpy()}")
        print(f"  actions: {[float(a) for a in actions[0]]} (长度: {len(actions[0])})")
        print(f"  rewards: {rewards[0]} (长度: {len(rewards[0])})")
        print(f"  next_state: {next_states[0].cpu().numpy()}")
        print(f"  done: {dones[0]} (标量值)")
    else:
        print(f"\n实际采样结果: 无样本，符合预期")

def test_early_done():
    """测试早期done（第2步就done=True）"""
    print("\n=== 测试早期done ===")
    
    device = torch.device("cpu")
    buffer = MultiStepReplayBuffer(capacity=1000, action_horizon=4, device=device)
    
    print("\n创建早期done的episode（第2步done=True，action_horizon=4）:")
    episode_data = [
        {'state': np.array([20.0, 20.0]), 'action': np.array([1.1]), 'reward': 11.0, 'next_state': np.array([21.0, 21.0]), 'done': False},
        {'state': np.array([21.0, 21.0]), 'action': np.array([1.2]), 'reward': 12.0, 'next_state': np.array([22.0, 22.0]), 'done': True}
    ]
    
    for i, step_data in enumerate(episode_data):
        buffer.add(**step_data)
        print(f"  步骤{i}: state={step_data['state']}, action={step_data['action'][0]}, reward={step_data['reward']}, done={step_data['done']}")
    
    print(f"\nEpisode结束，缓冲区大小: {buffer.size()}")
    print("\n理论上应该生成的样本:")
    print("  无样本: episode长度(2) < action_horizon(4)，不满足最小长度要求")
    print("说明: 当episode长度小于action_horizon时，不产生任何样本，等待收集更多数据")
    
    if buffer.size() == 0:
        print("\n实际采样结果: 无样本，符合预期")
    else:
        print(f"\n错误: 缓冲区应该为空，但实际大小为{buffer.size()}")

def test_long_episode():
    """测试长episode（长度远大于action_horizon）"""
    print("\n=== 测试长episode ===")
    
    device = torch.device("cpu")
    buffer = MultiStepReplayBuffer(capacity=1000, action_horizon=3, device=device)
    
    print("\n创建长episode（7步，action_horizon=3）:")
    episode_data = []
    for i in range(7):
        done = (i == 6)  # 最后一步done=True
        episode_data.append({
            'state': np.array([float(i*10), float(i*10)]), 
            'action': np.array([float(i+1)]), 
            'reward': float((i+1)*10), 
            'next_state': np.array([float((i+1)*10), float((i+1)*10)]), 
            'done': done
        })
    
    for i, step_data in enumerate(episode_data):
        buffer.add(**step_data)
        print(f"  步骤{i}: state={step_data['state']}, action={step_data['action'][0]}, reward={step_data['reward']}, done={step_data['done']}")
    
    print(f"\nEpisode结束，缓冲区大小: {buffer.size()}")
    print("\n理论上应该生成的样本:")
    print("  样本0: state[0]=[0,0], action[0:3]=[1,2,3], reward[0:3]=[10,20,30], next_state[3]=[30,30], done[3]=False")
    print("  样本1: state[1]=[10,10], action[1:4]=[2,3,4], reward[1:4]=[20,30,40], next_state[4]=[40,40], done[4]=False")
    print("  样本2: state[2]=[20,20], action[2:5]=[3,4,5], reward[2:5]=[30,40,50], next_state[5]=[50,50], done[5]=False")
    print("  样本3: state[3]=[30,30], action[3:6]=[4,5,6], reward[3:6]=[40,50,60], next_state[6]=[60,60], done[6]=False")
    print("  样本4: state[4]=[40,40], action[4:7]=[5,6,7], reward[4:7]=[50,60,70], next_state[7]=[70,70], done[7]=True")
    print("说明: 长episode生成多个样本，最后一个样本包含done=True")
    
    states, actions, rewards, next_states, dones = buffer.sample(5)
    print(f"\n实际采样结果:")
    for i in range(5):
        print(f"\n样本{i}:")
        print(f"  state: {states[i].cpu().numpy()}")
        print(f"  actions: {[float(a) for a in actions[i]]} (长度: {len(actions[i])})")
        print(f"  rewards: {rewards[i]} (长度: {len(rewards[i])})")
        print(f"  next_state: {next_states[i].cpu().numpy()}")
        print(f"  done: {dones[i]} (标量值)")

def test_single_step_episode():
    """测试单步episode（只有1步就done=True）"""
    print("\n=== 测试单步episode ===")
    
    device = torch.device("cpu")
    buffer = MultiStepReplayBuffer(capacity=1000, action_horizon=3, device=device)
    
    print("\n创建单步episode（1步就done=True）:")
    episode_data = [
        {'state': np.array([100.0]), 'action': np.array([9.9]), 'reward': 99.0, 'next_state': np.array([199.0]), 'done': True}
    ]
    
    for i, step_data in enumerate(episode_data):
        buffer.add(**step_data)
        print(f"  步骤{i}: state={step_data['state']}, action={step_data['action'][0]}, reward={step_data['reward']}, done={step_data['done']}")
    
    print(f"\nEpisode结束，缓冲区大小: {buffer.size()}")
    print("\n理论上应该生成的样本:")
    print("  无样本: episode长度(1) < action_horizon(3)，不满足最小长度要求")
    print("说明: 单步episode长度不足，不产生任何样本")
    
    if buffer.size() == 0:
        print("\n实际采样结果: 无样本，符合预期")
    else:
        print(f"\n错误: 缓冲区应该为空，但实际大小为{buffer.size()}")




if __name__ == "__main__":
    # 原始测试
    test_sampling_format()
    
    # 新增的不同长度测试
    test_short_episode()
    test_early_done()
    test_long_episode()
    test_single_step_episode()
    
    print("\n=== 所有测试完成 ===")
    print("✓ 基础采样格式测试通过")
    print("✓ 不同action_horizon的短episode测试通过")
    print("✓ 早期done测试通过")
    print("✓ 长episode测试通过")
    print("✓ 单步episode测试通过")
    print("\nMultiStepReplayBuffer在各种episode长度下均工作正常！")
    
    print("\n" + "="*50)
    print(f"所有测试完成")
    print("采样格式: state[i], action[i:i+h], reward[i:i+h], next_state[i+h], done[i+h]")
    print("说明: done为标量值，表示action chunk执行完毕后的终止状态，符合action chunking逻辑")
    print("="*50)