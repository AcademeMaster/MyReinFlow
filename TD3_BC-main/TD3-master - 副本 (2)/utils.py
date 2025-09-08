from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import gymnasium as gym

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device


        self.use_tensor_storage = False
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):

        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        states = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(action), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(reward), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(done), dtype=torch.float).view(-1, 1).to(self.device)
        return states, actions, rewards, next_states, dones

    def size(self):
        return  len(self.buffer)


class MultiStepReplayBuffer:
    """多步采样经验回放缓冲区
    
    支持采样格式: state[i], action[i:i+h], reward[i:i+h], next_state[i+h], done[i+h]
    其中h为步数长度，用于n步TD学习
    
    优化设计：
    1. 使用deque自动管理容量
    2. 延迟处理episode，避免实时计算开销
    3. 批量生成多步样本，提升效率
    """
    
    def __init__(self, capacity, device, action_horizon=4):
        self.capacity = capacity
        self.device = device
        self.action_horizon = action_horizon
        
        # 当前正在构建的episode
        self.current_episode = []
        
        # 已完成的episodes队列
        self.completed_episodes = collections.deque(maxlen=capacity//10)
        
        # 预处理的多步样本数据
        self.multi_step_data = collections.deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        """添加单步转换到当前episode"""
        self.current_episode.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
        # episode结束时处理并生成多步样本
        if done:
            self._process_episode()
            self.current_episode = []
            
    def _process_episode(self):
        """处理完整的episode，生成多步采样数据
        
        采样格式: state[i], action[i:i+h], reward[i:i+h], next_state[i+h], done[i+h]
        
        优化策略：
        1. 预先计算episode信息，避免重复计算
        2. 批量生成样本，提升处理效率
        3. 智能处理done状态，减少条件判断
        """
        if len(self.current_episode) == 0:
            return
            
        episode_len = len(self.current_episode)
        
        # 预先计算episode的done位置，用于优化处理
        done_positions = []
        for idx, step in enumerate(self.current_episode):
            if step['done']:
                done_positions.append(idx)
        
        # 批量生成所有可能的多步样本
        for start_idx in range(episode_len):
            # 计算从当前位置开始的最大可收集步数
            remaining_steps = episode_len - start_idx
            max_possible_steps = min(self.action_horizon, remaining_steps)
            
            # 查找在max_possible_steps范围内的第一个done位置
            actual_steps = max_possible_steps
            early_termination = False
            
            for done_pos in done_positions:
                if start_idx <= done_pos < start_idx + max_possible_steps:
                    actual_steps = done_pos - start_idx + 1
                    early_termination = True
                    break
            
            # 批量提取数据（避免逐个append）
            end_idx = start_idx + actual_steps
            episode_slice = self.current_episode[start_idx:end_idx]
            
            # 构建多步样本
            # 根据预计算方法，dones只包含最后一个done值：done[i+h]
            sample = {
                'state': episode_slice[0]['state'],
                'actions': [step['action'] for step in episode_slice],
                'rewards': [step['reward'] for step in episode_slice],
                'next_state': episode_slice[-1]['next_state'],
                'dones': episode_slice[-1]['done'],  # 只有最后一个done值
                'actual_steps': actual_steps,
                'early_termination': early_termination
            }
            
            self.multi_step_data.append(sample)
        
        # 存储完整episode供后续分析（可选）
        self.completed_episodes.append(self.current_episode.copy())
            


    def sample(self, batch_size):
        """采样多步数据
        
        采样格式: state[i], action[i:i+h], reward[i:i+h], next_state[i+h], done[i+h]
        注意：不进行自动填充，保持原始序列长度，通过actual_steps标识有效数据长度
        
        返回:
            states: [batch_size, state_dim]
            actions: List[List] - 变长序列，每个样本的实际长度由actual_steps决定
            rewards: List[List] - 变长序列，每个样本的实际长度由actual_steps决定
            next_states: [batch_size, state_dim]
            dones: List[bool] - 每个样本的最终done状态（done[i+h]）
            actual_steps: [batch_size] - 每个样本的实际有效步数
            early_termination: [batch_size] - 是否因done而早期终止
        """
        if len(self.multi_step_data) < batch_size:
            raise ValueError(f"缓冲区数据不足，需要{batch_size}个样本，但只有{len(self.multi_step_data)}个")
        
        # 从deque中随机采样
        sample_indices = random.sample(range(len(self.multi_step_data)), batch_size)
        samples = [self.multi_step_data[i] for i in sample_indices]
        
        # 分离数据，保持原始长度（不填充）
        states = [sample['state'] for sample in samples]
        actions = [sample['actions'] for sample in samples]  # 变长列表
        rewards = [sample['rewards'] for sample in samples]  # 变长列表
        next_states = [sample['next_state'] for sample in samples]
        dones = [sample['dones'] for sample in samples]  # 单个done值列表
        actual_steps = [sample['actual_steps'] for sample in samples]
        early_termination = [sample['early_termination'] for sample in samples]
        
        # 转换为tensor（states和next_states可以直接转换）
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
        actual_steps = torch.tensor(np.array(actual_steps), dtype=torch.long).to(self.device)
        early_termination = torch.tensor(np.array(early_termination), dtype=torch.bool).to(self.device)
        
        # actions, rewards保持为变长列表，dones为单个值列表，由使用者根据actual_steps处理
        # 如果需要tensor格式，使用者可以根据actual_steps进行适当的填充或截断
        
        return states, actions, rewards, next_states, dones, actual_steps, early_termination
    
    def size(self):
        """返回可用的多步样本数量"""
        return len(self.multi_step_data)
    
    def __len__(self):
        """返回可用的多步样本数量"""
        return len(self.multi_step_data)
    
    def get_episode_count(self):
        """返回存储的episode数量"""
        return len(self.completed_episodes)


def eval_policy(agent, env_name, seed, eval_episodes=10):
    """评估策略性能"""
    eval_env = gym.make(env_name)

    eval_env.reset(seed=seed + 100) 
    return_list = []
    
    for _ in range(eval_episodes):
        state, info = eval_env.reset()
        done = False
        episode_return = 0.
        while not done:
            action = agent.take_action(state, add_noise=False)  # 评估时不添加噪声
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_return += reward
        return_list.append(episode_return)
    eval_env.close()
    avg_reward = np.mean(return_list)
    print("----------------------------------------------------------------------------")
    print(f"Evaluation over {eval_episodes} eval_episodes, avg_reward: {avg_reward:.3f}")
    print("----------------------------------------------------------------------------")
    return avg_reward


def train_td3_timestep(env, agent, max_timesteps, replay_buffer, minimal_size, batch_size, start_timesteps=1000, expl_noise=0.1, eval_freq=5000, env_name='Pendulum-v1', seed=0):
    """基于时间步的TD3训练循环，参考官方实现"""
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
    
    # 获取动作空间信息
    max_action = float(env.action_space.high[0])
    action_dim = env.action_space.shape[0]
    
    with tqdm(total=max_timesteps, desc='Training') as pbar:
        for t in range(max_timesteps):
            # 选择动作：初期随机探索，后期策略+噪声
            if t < start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.take_action(state, add_noise=True)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_timesteps += 1
            
            # 训练智能体（收集足够数据后每步都训练）
            if t >= start_timesteps and replay_buffer.size() > minimal_size:
                agent.update(batch_size)
            
            # Episode结束处理
            if done:
                # 记录episode奖励
                episode_rewards.append(episode_reward)
                # 重置环境和变量
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
            
            # 更新进度条
            if len(episode_rewards) > 0:
                pbar.set_postfix({
                    'Episode': episode_num,
                    'Timesteps': episode_timesteps,
                    'Avg_Reward': f'{np.mean(episode_rewards[-10:]):.3f}' if len(episode_rewards) >= 10 else f'{np.mean(episode_rewards):.3f}',
                    'Last_Eval': f'{evaluations[-1]:.3f}' if len(evaluations) > 1 else 'N/A'
                })
            pbar.update(1)
    
    print(f"\n训练完成！")
    print(f"总Episodes: {episode_num}")
    print(f"最终评估分数: {evaluations[-1]:.3f}")
    print(f"平均Episode奖励: {np.mean(episode_rewards):.3f}")
    return episode_rewards, evaluations


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, update_freq=4, gradient_steps=2):
    """保留原有的基于episode的训练函数以兼容性"""
    return_list = []
    step_count = 0
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    step_count += 1
                    # 每update_freq步进行批量更新，而不是每步都更新
                    if step_count % update_freq == 0 and replay_buffer.size() > minimal_size:
                        # 批量更新：一次采样进行多次梯度更新
                        for _ in range(gradient_steps):
                            agent.update(batch_size)
                return_list.append(episode_return)
                pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


