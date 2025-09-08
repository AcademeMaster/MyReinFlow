# MultiStepReplayBuffer 多步采样经验回放缓冲区

## 概述

`MultiStepReplayBuffer` 是一个专门为多步强化学习设计的经验回放缓冲区，支持采样格式：
```
state[i], action[i:i+h], reward[i:i+h], next_state[i+h], done[i:i+h]
```

其中 `h` 为步数长度，用于 n 步 TD 学习。

## 核心特性

### 1. 多步序列采样
- **输入格式**: 单步转换 `(state, action, reward, next_state, done)`
- **输出格式**: 多步序列 `(state[i], actions[i:i+h], rewards[i:i+h], next_state[i+h], dones[i:i+h])`
- **自动构建**: 从单步数据自动构建多步序列

### 2. 边界条件处理
- **短 Episode 处理**: 当 episode 长度小于 n_step 时，自动用零填充
- **Episode 结束处理**: 正确处理 episode 边界，避免跨 episode 采样
- **Done 序列**: 提供完整的 done 标记序列，便于计算 n 步回报

### 3. 轨迹顺序保持
- **顺序存储**: 确保多步动作来自连续的时间步
- **策略一致性**: 保证 n 步序列中的动作都来自同一策略

## 使用方法

### 基本使用

```python
from utils import MultiStepReplayBuffer
import torch

# 创建缓冲区
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
buffer = MultiStepReplayBuffer(
    capacity=100000,  # 缓冲区容量
    device=device,    # 设备
    n_step=4         # 多步长度
)

# 数据收集（与普通 ReplayBuffer 相同）
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        # 添加单步转换
        buffer.add(state, action, reward, next_state, done)
        
        state = next_state

# 多步采样
if buffer.size() > batch_size:
    states, actions, rewards, next_states, dones, actual_steps = buffer.sample(batch_size)
    
    # states: [batch_size, state_dim] - 起始状态
    # actions: [batch_size, n_step, action_dim] - 动作序列
    # rewards: [batch_size, n_step] - 奖励序列  
    # next_states: [batch_size, state_dim] - 最终状态
    # dones: [batch_size, n_step] - done 标记序列
    # actual_steps: [batch_size] - 实际有效步数
```

### 与 Actor 多步输出配合

```python
# Actor 输出多步动作序列
class MultiStepActor(nn.Module):
    def __init__(self, state_dim, action_dim, n_step):
        super().__init__()
        self.n_step = n_step
        # ... 网络定义
    
    def forward(self, state):
        # 输出 n_step 个动作
        actions = self.network(state)  # [batch_size, n_step * action_dim]
        return actions.view(-1, self.n_step, action_dim)

# 训练循环
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Actor 输出多步动作序列
        multi_actions = actor(state)  # [1, n_step, action_dim]
        
        # 依次执行每个动作
        for i in range(n_step):
            action = multi_actions[0, i].cpu().numpy()
            next_state, reward, done, _ = env.step(action)
            
            # 存储单步转换
            buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            if done:
                break
```

## 数据格式说明

### 输入数据（单步）
```python
state: np.array([state_dim])          # 当前状态
action: np.array([action_dim])        # 执行的动作
reward: float                         # 获得的奖励
next_state: np.array([state_dim])     # 下一个状态
done: bool                            # 是否结束
```

### 输出数据（多步）
```python
states: torch.Tensor([batch_size, state_dim])           # 起始状态
actions: torch.Tensor([batch_size, n_step, action_dim]) # 动作序列
rewards: torch.Tensor([batch_size, n_step])             # 奖励序列
next_states: torch.Tensor([batch_size, state_dim])      # 最终状态
dones: torch.Tensor([batch_size, n_step])               # done 序列
actual_steps: torch.Tensor([batch_size])                # 实际步数
```

## 边界条件示例

### 情况 1: Episode 长度 ≥ n_step
```
Episode: [s0, a0, r0, s1] -> [s1, a1, r1, s2] -> [s2, a2, r2, s3] -> [s3, a3, r3, s4] -> [s4, a4, r4, s5(done)]
n_step = 4

可生成的多步样本:
1. state=s0, actions=[a0,a1,a2,a3], rewards=[r0,r1,r2,r3], next_state=s4, dones=[F,F,F,F]
2. state=s1, actions=[a1,a2,a3,a4], rewards=[r1,r2,r3,r4], next_state=s5, dones=[F,F,F,T]
```

### 情况 2: Episode 长度 < n_step
```
Episode: [s0, a0, r0, s1] -> [s1, a1, r1, s2(done)]
n_step = 4

生成的多步样本:
1. state=s0, actions=[a0,a1,0,0], rewards=[r0,r1,0,0], next_state=s2, dones=[F,T,T,T]
2. state=s1, actions=[a1,0,0,0], rewards=[r1,0,0,0], next_state=s2, dones=[T,T,T,T]
```

## n 步 TD 学习应用

```python
def compute_n_step_td_target(rewards, next_states, dones, actual_steps, gamma=0.99):
    """
    计算 n 步 TD 目标值
    
    Args:
        rewards: [batch_size, n_step] - 奖励序列
        next_states: [batch_size, state_dim] - 最终状态
        dones: [batch_size, n_step] - done 序列
        actual_steps: [batch_size] - 实际步数
        gamma: 折扣因子
    
    Returns:
        targets: [batch_size] - n 步 TD 目标值
    """
    batch_size = rewards.shape[0]
    targets = torch.zeros(batch_size, device=rewards.device)
    
    for i in range(batch_size):
        n = actual_steps[i].item()
        
        # 计算 n 步累积奖励
        cumulative_reward = 0
        discount = 1
        
        for j in range(n):
            cumulative_reward += discount * rewards[i, j]
            discount *= gamma
            
            # 如果遇到 done，提前结束
            if dones[i, j]:
                break
        
        # 如果没有提前结束，加上最终状态的价值
        if not dones[i, n-1]:
            final_value = critic(next_states[i:i+1])  # 评估最终状态价值
            cumulative_reward += discount * final_value
        
        targets[i] = cumulative_reward
    
    return targets
```

## 优势

1. **策略一致性**: 确保 n 步序列中的动作都来自同一策略，提高学习稳定性
2. **高效采样**: 预处理多步数据，支持快速随机采样
3. **自动边界处理**: 无需手动处理 episode 边界条件
4. **灵活配置**: 可调节 n_step 参数适应不同算法需求
5. **兼容性**: 与现有 TD3 等算法框架兼容

## 注意事项

1. **内存使用**: 多步缓冲区会使用更多内存，因为需要存储序列数据
2. **Episode 完整性**: 只有在 episode 结束后才会生成多步样本
3. **采样延迟**: 相比单步采样，需要等待 episode 结束才能采样
4. **填充处理**: 短 episode 会用零填充，需要在计算中正确处理

## 测试

运行测试示例：
```bash
python multi_step_example.py
```

这将演示 MultiStepReplayBuffer 的基本使用、边界条件处理和与智能体的配合使用。