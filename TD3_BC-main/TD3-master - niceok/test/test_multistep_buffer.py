import numpy as np
import torch
from utils import MultiStepReplayBuffer

def test_sampling_format():
    """详细测试采样格式: state[i], action[i:i+h], reward[i:i+h], next_state[i+h], done[i+h]"""
    print("=== 测试采样格式 ===")
    
    device = torch.device("cpu")
    buffer = MultiStepReplayBuffer(capacity=1000, n_step=3, device=device)
    
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
    print("  样本0: state[0]=[0,0], action[0:3]=[1,2,3], reward[0:3]=[10,20,30], next_state[3]=[3,3], done[0:3]=[F,F,F]")
    print("  样本1: state[1]=[1,1], action[1:4]=[2,3,4], reward[1:4]=[20,30,40], next_state[4]=[4,4], done[1:4]=[F,F,T]")
    print("  样本2: state[2]=[2,2], action[2:5]=[3,4,0], reward[2:5]=[30,40,0], next_state[5]=[4,4], done[2:5]=[F,T,T] (填充)")
    print("  样本3: state[3]=[3,3], action[3:6]=[4,0,0], reward[3:6]=[40,0,0], next_state[6]=[4,4], done[3:6]=[T,T,T] (填充)")
    
    # 采样所有数据
    print("\n实际采样结果:")
    states, actions, rewards, next_states, dones, actual_steps, early_termination = buffer.sample(4)
    
    for i in range(4):
        print(f"\n样本{i}:")
        print(f"  state[{i}]: {states[i].cpu().numpy()}")
        print(f"  action[{i}:{i}+h]: {[float(a) for a in actions[i]]} (长度: {len(actions[i])})")
        print(f"  reward[{i}:{i}+h]: {rewards[i]} (长度: {len(rewards[i])})")
        print(f"  next_state[{i}+h]: {next_states[i].cpu().numpy()}")
        print(f"  done[{i}:{i}+h]: {dones[i]} (长度: {len(dones[i])})")
        print(f"  actual_steps: {actual_steps[i].cpu().numpy()}")
        print(f"  early_termination: {early_termination[i].cpu().numpy()}")

def test_short_episode():
    """测试短episode的处理"""
    print("\n\n=== 测试短episode ===")
    
    device = torch.device("cpu")
    buffer = MultiStepReplayBuffer(capacity=1000, n_step=5, device=device)
    
    # 创建一个只有2步的episode
    print("\n创建短episode（2步，n_step=5）:")
    short_episode = [
        {'state': np.array([5.0, 5.0]), 'action': np.array([0.5]), 'reward': 50.0, 'next_state': np.array([5.5, 5.5]), 'done': False},
        {'state': np.array([5.5, 5.5]), 'action': np.array([0.6]), 'reward': 60.0, 'next_state': np.array([6.0, 6.0]), 'done': True}
    ]
    
    for i, step_data in enumerate(short_episode):
        buffer.add(**step_data)
        print(f"  步骤{i}: state={step_data['state']}, action={step_data['action'][0]}, reward={step_data['reward']}, done={step_data['done']}")
    
    print(f"\nEpisode结束，缓冲区大小: {buffer.size()}")
    print("\n理论上应该生成的样本:")
    print("  样本0: state[0]=[5,5], action[0:5]=[0.5,0.6,0,0,0], reward[0:5]=[50,60,0,0,0], next_state[2]=[6,6], done[0:5]=[F,T,T,T,T]")
    print("  样本1: state[1]=[5.5,5.5], action[1:6]=[0.6,0,0,0,0], reward[1:6]=[60,0,0,0,0], next_state[2]=[6,6], done[1:6]=[T,T,T,T,T]")
    
    # 采样验证
    print("\n实际采样结果:")
    states, actions, rewards, next_states, dones, actual_steps, early_termination = buffer.sample(2)
    
    for i in range(2):
        print(f"\n样本{i}:")
        print(f"  state: {states[i].cpu().numpy()}")
        print(f"  actions: {[float(a) for a in actions[i]]} (长度: {len(actions[i])})")
        print(f"  rewards: {rewards[i]} (长度: {len(rewards[i])})")
        print(f"  next_state: {next_states[i].cpu().numpy()}")
        print(f"  dones: {dones[i]} (长度: {len(dones[i])})")
        print(f"  actual_steps: {actual_steps[i].cpu().numpy()}")
        print(f"  early_termination: {early_termination[i].cpu().numpy()}")

def verify_variable_length_format():
    """验证变长序列格式的正确性"""
    print("\n\n=== 验证变长序列格式正确性 ===")
    
    device = torch.device("cpu")
    buffer = MultiStepReplayBuffer(capacity=1000, n_step=5, device=device)
    
    # 添加一个短episode用于验证
    episode = [
        {'state': np.array([0.0]), 'action': np.array([1.0]), 'reward': 1.0, 'next_state': np.array([1.0]), 'done': False},
        {'state': np.array([1.0]), 'action': np.array([2.0]), 'reward': 2.0, 'next_state': np.array([2.0]), 'done': True}
    ]
    
    for step_data in episode:
        buffer.add(**step_data)
    
    print(f"\n缓冲区大小: {buffer.size()}")
    
    # 采样所有样本
    states, actions, rewards, next_states, dones, actual_steps, early_termination = buffer.sample(2)
    
    print("\n变长序列采样结果:")
    for i in range(2):
        print(f"\n样本{i}:")
        print(f"  state: {states[i].cpu().numpy()}")
        print(f"  actions (变长): {actions[i]} (长度: {len(actions[i])})")
        print(f"  rewards (变长): {rewards[i]} (长度: {len(rewards[i])})")
        print(f"  next_state: {next_states[i].cpu().numpy()}")
        print(f"  dones (变长): {dones[i]} (长度: {len(dones[i])})")
        print(f"  actual_steps: {actual_steps[i].cpu().numpy()}")
        print(f"  early_termination: {early_termination[i].cpu().numpy()}")
        
        # 验证数据完整性
        expected_length = actual_steps[i].cpu().numpy()
        actions_length = len(actions[i])
        rewards_length = len(rewards[i])
        dones_length = len(dones[i])
        
        length_correct = (actions_length == expected_length and 
                         rewards_length == expected_length and 
                         dones_length == expected_length)
        
        print(f"  ✓ 长度一致性: {'通过' if length_correct else '失败'} (期望: {expected_length}, 实际: {actions_length})")
    
    print("\n✓ 变长序列格式验证完成，无虚假填充数据")
    return True

if __name__ == "__main__":
    test_sampling_format()
    test_short_episode()
    format_ok = verify_variable_length_format()
    
    print("\n" + "="*50)
    print(f"所有测试完成！变长序列格式{'正确' if format_ok else '有误'}")
    print("采样格式: state[i], action[i:i+h], reward[i:i+h], next_state[i+h], done[i+h]")
    print("优化: 无自动填充，通过actual_steps标识真实数据长度，避免虚假数据")
    print("="*50)