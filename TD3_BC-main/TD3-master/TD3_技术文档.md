# TD3算法与多步学习技术文档

## 目录
1. [项目概述](#项目概述)
2. [算法原理](#算法原理)
3. [网络架构](#网络架构)
4. [数据处理](#数据处理)
5. [核心算法实现](#核心算法实现)
6. [训练流程](#训练流程)
7. [代码详解](#代码详解)
8. [参数配置](#参数配置)

## 项目概述

本项目实现了基于TD3（Twin Delayed Deep Deterministic Policy Gradient）算法的强化学习智能体，并集成了多步学习（Multi-Step Learning）和动作分块（Action Chunking）技术。主要特点包括：

- **TD3算法**：解决了DDPG算法中的过估计问题
- **多步学习**：通过n步TD学习提高样本效率
- **动作分块**：一次生成多个连续动作，提高动作一致性
- **经验回放**：支持多步采样的经验回放缓冲区

## 算法原理

### TD3算法核心思想

TD3算法通过三个关键技术改进DDPG：

1. **双Q网络（Twin Q-Networks）**：使用两个Q网络，取最小值作为目标
2. **延迟策略更新（Delayed Policy Updates）**：降低策略更新频率
3. **目标策略平滑（Target Policy Smoothing）**：为目标动作添加噪声

### 数学公式

#### 1. Q值更新公式

对于标准TD3：
```
y = r + γ * min(Q₁'(s', a'), Q₂'(s', a'))
```

对于多步TD3（本项目实现）：
```
G = Σ(t=0 to H-1) γᵗ * rₜ₊₁
y = G + γᴴ * min(Q₁'(s', a'), Q₂'(s', a'))
```

其中：
- `G`：多步折扣奖励
- `H`：动作序列长度（action_horizon）
- `γ`：折扣因子
- `Q₁', Q₂'`：目标Q网络

#### 2. 策略损失函数
```
L_π = -E[Q₁(s, π(s))]
```

#### 3. Q网络损失函数
```
L_Q = E[(Q₁(s,a) - y)²] + E[(Q₂(s,a) - y)²]
```

#### 4. 软更新公式
```
θ' ← τθ + (1-τ)θ'
```

## 网络架构

### Actor网络

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),      # 输入层
            nn.ReLU(),
            nn.Linear(256, 256),            # 隐藏层
            nn.ReLU(),
            nn.Linear(256, action_dim)      # 输出层
        )
        
        self.max_action = max_action
```

**架构特点**：
- 输入维度：状态空间维度
- 输出维度：`action_dim * action_horizon`（支持动作分块）
- 激活函数：ReLU + Tanh（输出层）
- 权重初始化：Xavier均匀分布

### Critic网络

```python
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # 双Q网络架构
        self.Q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
```

**架构特点**：
- 输入：状态和动作的拼接
- 输出：Q值（标量）
- 双网络设计：Q1和Q2独立计算

## 数据处理

### 多步经验回放缓冲区

#### 核心设计思想

```python
class MultiStepReplayBuffer:
    """
    支持采样格式: state[i], action[i:i+h], reward[i:i+h], next_state[i+h], done[i+h]
    其中h为步数长度，用于n步TD学习
    """
```

#### 滑动窗口机制

给定原始轨迹长度`n`和窗口大小`h`，通过滑动窗口生成新轨迹：

```
新轨迹长度 l = max(n - h + 1, 0)
```

**示例**：
- 原始轨迹：`[s₀, s₁, s₂, s₃, s₄, s₅]`（长度=6）
- 窗口大小：`h=4`
- 生成样本：
  - 样本1：`state[0], actions[0:4], rewards[0:4], next_state[3], done[3]`
  - 样本2：`state[1], actions[1:5], rewards[1:5], next_state[4], done[4]`
  - 样本3：`state[2], actions[2:6], rewards[2:6], next_state[5], done[5]`

#### 数据处理流程

```python
def _process_episode(self):
    """处理完整的episode，生成多步采样数据"""
    if len(self.current_episode) < self.action_horizon:
        return
        
    episode_len = len(self.current_episode)
    new_episode_lin = episode_len - self.action_horizon + 1
    
    # 批量生成所有可能的多步样本
    for i in range(new_episode_lin):
        self.multi_step_data.append({
            'state': [self.current_episode[i]['state']],
            'actions': [self.current_episode[i+j]['action'] for j in range(self.action_horizon)],
            'rewards': [self.current_episode[i+j]['reward'] for j in range(self.action_horizon)],
            'next_state': [self.current_episode[i+self.action_horizon-1]['next_state']],
            'dones': [self.current_episode[i+self.action_horizon-1]['done']],
        })
```

## 核心算法实现

### 动作分块机制

#### 动作生成

```python
@torch.no_grad()
def take_action(self, state, add_noise=True):
    # 如果还有动作，就每次单步取出动作用于执行
    if len(self.action_deque) > 0:
        action = self.action_deque.popleft()
    else:
        # 执行actor推理action chunking，并保存
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action_chunk = self.actor(state)  # 生成action_horizon个动作
        
        if add_noise:
            noise = torch.normal(0, self.sigma, size=action_chunk.shape).to(self.device)
            action_chunk = action_chunk + noise
            action_chunk = torch.clamp(action_chunk, -self.max_action, self.max_action)
        
        # 将action chunk分解为单个动作并存入deque
        action_chunk_np = action_chunk.cpu().data.numpy().flatten()
        for i in range(self.action_horizon):
            start_idx = i * self.action_dim
            end_idx = (i + 1) * self.action_dim
            single_action = action_chunk_np[start_idx:end_idx]
            self.action_deque.append(torch.tensor(single_action, dtype=torch.float).to(self.device))
        
        # 取出第一个动作
        action = self.action_deque.popleft()
        
    return action.cpu().data.numpy().flatten()
```

**关键特点**：
1. **一次生成多个动作**：Actor网络输出`action_horizon`个连续动作
2. **动作队列管理**：使用deque存储生成的动作序列
3. **逐步执行**：每次环境交互只使用一个动作

### 多步折扣奖励计算

```python
def compute_discounted_reward(self, rewards_tensor):
    """
    计算多步折扣奖励：G = Σ(t=0 to H-1) γᵗ * rₜ
    
    Args:
        rewards_tensor: [batch_size, action_horizon] - 奖励序列
        
    Returns:
        torch.Tensor: [batch_size, 1] - 折扣奖励
    """
    # 预计算的折扣因子：[γ⁰, γ¹, γ², ..., γᴴ⁻¹]
    # self.gamma_powers = [1, γ, γ², γ³, ...]
    
    # 并行计算折扣奖励：逐元素相乘后按行求和
    discounted_matrix = rewards_tensor * self.gamma_powers
    discounted_rewards = discounted_matrix.sum(dim=1, keepdim=True)
    return discounted_rewards
```

**数学原理**：
```
G = r₁ + γ*r₂ + γ²*r₃ + ... + γᴴ⁻¹*rₕ
  = Σ(t=0 to H-1) γᵗ * rₜ₊₁
```

### TD3更新算法

```python
def update(self, batch_size):
    self.total_it += 1
    
    # 1. 采样多步数据
    states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

    with torch.no_grad():
        # 2. 目标策略平滑
        target_action = self.actor_target(next_states)
        noise = (torch.randn_like(target_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (target_action + noise).clamp(-self.max_action, self.max_action)
        
        # 3. 双Q网络目标值计算
        target_Q1, target_Q2 = self.critic_target(next_states, next_action)
        target_Q = torch.min(target_Q1, target_Q2)

        # 4. 多步折扣奖励
        discounted_rewards = self.compute_discounted_reward(rewards)
        
        # 5. 目标Q值：G + γⁿ * Q(s', a')
        target_Q = discounted_rewards + (1 - dones) * (self.discount ** self.action_horizon) * target_Q

    # 6. 当前Q值
    current_Q1, current_Q2 = self.critic(states, actions)

    # 7. Critic损失和更新
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # 8. 延迟策略更新
    if self.total_it % self.policy_freq == 0:
        # Actor损失和更新
        actor_loss = -self.critic.Q1Value(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

## 训练流程

### 基于时间步的训练循环

```python
def train_td3_timestep(env, agent, max_timesteps, replay_buffer, minimal_size, batch_size, 
                      start_timesteps=1000, expl_noise=0.1, eval_freq=5000, env_name='Pendulum-v1', seed=0):
    """
    基于时间步的TD3训练循环，参考官方实现
    """
    evaluations = []
    episode_rewards = []
    
    # 评估未训练的策略
    evaluations.append(eval_policy(agent, env_name, seed))
    
    # 初始化环境
    state, _ = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    with tqdm(total=max_timesteps, desc='Training') as pbar:
        for t in range(max_timesteps):
            # 选择动作：初期随机探索，后期策略+噪声
            if t < start_timesteps:
                action = env.action_space.sample()  # 随机动作
            else:
                action = agent.take_action(state, add_noise=True)  # 策略动作+噪声

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_timesteps += 1
            
            # 训练智能体（收集足够数据后每步都训练）
            if t >= start_timesteps and len(replay_buffer.multi_step_data) > minimal_size:
                agent.update(batch_size)
            
            # Episode结束处理
            if done:
                episode_rewards.append(episode_reward)
                state, _ = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                done = False
            
            # 定期评估
            if (t + 1) % eval_freq == 0:
                eval_reward = eval_policy(agent, env_name, seed)
                evaluations.append(eval_reward)
                print(f"\nTimestep {t+1}: 评估分数 = {eval_reward:.3f}")
            
            pbar.update(1)
    
    return episode_rewards, evaluations
```

### 训练阶段划分

1. **探索阶段**（0 ~ start_timesteps）：
   - 使用随机动作探索环境
   - 收集初始经验数据
   - 不进行网络更新

2. **学习阶段**（start_timesteps ~ max_timesteps）：
   - 使用策略网络生成动作（加噪声）
   - 每步进行网络更新
   - 定期评估性能

## 代码详解

### 配置管理

```python
@dataclass
class TD3Config:
    # 学习率配置
    actor_lr: float = 3e-4          # Actor学习率
    critic_lr: float = 3e-4         # Critic学习率
    
    # 训练配置
    max_timesteps: int = int(2e4)   # 最大训练步数
    start_timesteps: int = 1000     # 随机探索步数
    batch_size: int = 256           # 批次大小
    
    # 网络配置
    hidden_dim: int = 256           # 隐藏层维度
    gamma: float = 0.99             # 折扣因子
    tau: float = 0.005              # 软更新参数
    
    # TD3特有参数
    policy_noise: float = 0.5       # 目标策略噪声
    noise_clip: float = 1.0         # 噪声裁剪
    policy_freq: int = 2            # 策略更新频率
    
    # 多步学习参数
    action_horizon: int = 4         # 动作序列长度
    sigma: float = 0.2              # 探索噪声标准差
```

### 环境初始化

```python
def setup_environment(config: TD3Config) -> EnvState:
    # 设置设备
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 创建环境
    env = gym.make(config.env_name)
    
    # 设置随机种子（确保可重现性）
    random.seed(config.seed)
    np.random.seed(config.seed)
    env.reset(seed=config.seed)
    torch.manual_seed(config.seed)
    
    # 获取环境参数
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    
    # 创建多步回放缓冲区
    replay_buffer = MultiStepReplayBuffer(config.buffer_size, device, config.action_horizon)
    
    return EnvState(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        device=device,
        replay_buffer=replay_buffer
    )
```

### 智能体创建

```python
# 创建TD3智能体
agent = TD3(
    env_state.state_dim,           # 状态维度
    config.hidden_dim,             # 隐藏层维度
    env_state.action_dim,          # 动作维度
    env_state.action_bound,        # 动作边界
    config.sigma,                  # 噪声标准差
    config.actor_lr,               # Actor学习率
    config.critic_lr,              # Critic学习率
    config.tau,                    # 软更新参数
    config.gamma,                  # 折扣因子
    env_state.device,              # 计算设备
    config.policy_noise,           # 策略噪声
    config.noise_clip,             # 噪声裁剪
    config.policy_freq,            # 策略更新频率
    env_state.replay_buffer,       # 回放缓冲区
    action_horizon=config.action_horizon  # 动作序列长度
)
```

## 参数配置

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `action_horizon` | 4 | 动作序列长度，影响多步学习和动作分块 |
| `gamma` | 0.99 | 折扣因子，控制未来奖励的重要性 |
| `tau` | 0.005 | 软更新参数，控制目标网络更新速度 |
| `policy_freq` | 2 | 策略更新频率，TD3的延迟更新机制 |
| `policy_noise` | 0.5 | 目标策略噪声，TD3的平滑机制 |
| `noise_clip` | 1.0 | 噪声裁剪范围 |
| `sigma` | 0.2 | 探索噪声标准差 |
| `batch_size` | 256 | 训练批次大小 |
| `buffer_size` | 1000000 | 回放缓冲区容量 |

### 参数调优建议

1. **action_horizon**：
   - 较小值（1-2）：接近标准TD3
   - 较大值（4-8）：更强的动作一致性，但可能降低灵活性

2. **gamma**：
   - 接近1：重视长期奖励
   - 较小值：重视即时奖励

3. **tau**：
   - 较小值：目标网络更新更稳定
   - 较大值：更快的学习速度

4. **policy_freq**：
   - TD3推荐值：2
   - 可根据环境复杂度调整

## 性能优化

### 计算优化

1. **预计算折扣因子**：
```python
self.gamma_powers = torch.pow(self.discount, torch.arange(self.action_horizon, dtype=torch.float).to(self.device))
```

2. **批量处理**：
```python
# 并行计算折扣奖励
discounted_matrix = rewards_tensor * self.gamma_powers
discounted_rewards = discounted_matrix.sum(dim=1, keepdim=True)
```

3. **内存管理**：
```python
# 使用deque自动管理容量
self.multi_step_data = collections.deque(maxlen=capacity)
```

### 训练稳定性

1. **梯度裁剪**（可选）：
```python
torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
```

2. **学习率调度**（可选）：
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
```

## 总结

本项目成功实现了TD3算法与多步学习的结合，主要创新点包括：

1. **多步TD学习**：通过n步奖励累积提高样本效率
2. **动作分块**：一次生成多个连续动作，提高动作一致性
3. **优化的数据结构**：高效的多步经验回放缓冲区
4. **稳定的训练流程**：基于时间步的训练循环

该实现在保持TD3算法核心优势的同时，通过多步学习技术进一步提升了学习效率和性能表现。