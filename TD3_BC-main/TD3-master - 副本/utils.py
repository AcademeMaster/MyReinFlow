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


