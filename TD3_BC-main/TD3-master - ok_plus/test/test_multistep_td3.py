import gymnasium as gym
import numpy as np
import torch
from TD3 import TD3
from utils import MultiStepReplayBuffer, eval_policy
from dataclasses import dataclass
import random

@dataclass
class MultiStepTD3Config:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    max_timesteps: int = 50000
    start_timesteps: int = 1000
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 100000
    minimal_size: int = 1000
    batch_size: int = 256
    expl_noise: float = 0.1
    env_name: str = 'Pendulum-v1'
    seed: int = 0
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    eval_freq: int = 5000
    # 多步参数
    n_step: int = 4  # 多步长度
    action_horizon: int = 4  # 动作序列长度
    sigma: float = 0.1

def setup_multistep_environment(config: MultiStepTD3Config):
    # 设置设备
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 创建并设置环境
    env = gym.make(config.env_name)
    
    # 设置随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    env.reset(seed=config.seed)
    torch.manual_seed(config.seed)
    
    # 初始化回放缓冲区和环境参数
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    
    # 使用MultiStepReplayBuffer
    replay_buffer = MultiStepReplayBuffer(config.buffer_size, device, config.n_step)
    
    return env, state_dim, action_dim, action_bound, device, replay_buffer

def train_multistep_td3(env, agent, config, replay_buffer):
    """多步TD3训练循环"""
    evaluations = []
    episode_rewards = []
    
    # 评估未训练的策略
    evaluations.append(eval_policy(agent, config.env_name, config.seed))
    
    # 初始化环境
    state, _ = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    print(f"开始多步TD3训练 (n_step={config.n_step}, action_horizon={config.action_horizon})...")
    
    for t in range(config.max_timesteps):
        # 选择动作：初期随机探索，后期策略+噪声
        if t < config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.take_action(state, add_noise=True)

        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 存储经验到多步缓冲区
        replay_buffer.add(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        episode_timesteps += 1
        
        # 训练智能体（收集足够数据后每步都训练）
        if t >= config.start_timesteps and replay_buffer.size() > config.minimal_size:
            agent.update(config.batch_size)
        
        # Episode结束处理
        if done:
            # 记录episode奖励
            episode_rewards.append(episode_reward)
            print(f"Episode {episode_num + 1}: Reward = {episode_reward:.3f}, Steps = {episode_timesteps}")
            
            # 重置环境和变量
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            done = False
        
        # 定期评估
        if (t + 1) % config.eval_freq == 0:
            eval_reward = eval_policy(agent, config.env_name, config.seed)
            evaluations.append(eval_reward)
            print(f"\nTimestep {t+1}: 评估分数 = {eval_reward:.3f}")
            print(f"缓冲区大小: {replay_buffer.size()}, Episodes: {replay_buffer.get_episode_count()}")
    
    print(f"\n训练完成！")
    print(f"总Episodes: {episode_num}")
    print(f"最终评估分数: {evaluations[-1]:.3f}")
    print(f"平均Episode奖励: {np.mean(episode_rewards):.3f}")
    return episode_rewards, evaluations

if __name__ == "__main__":
    # 初始化配置
    config = MultiStepTD3Config()
    
    # 设置环境
    env, state_dim, action_dim, action_bound, device, replay_buffer = setup_multistep_environment(config)
    
    # 创建多步TD3智能体
    agent = TD3(
        state_dim,
        config.hidden_dim,
        action_dim,
        action_bound,
        config.sigma,
        config.actor_lr,
        config.critic_lr,
        config.tau,
        config.gamma,
        device,
        config.policy_noise,
        config.noise_clip,
        config.policy_freq,
        replay_buffer,
        action_horion=config.action_horizon  # 设置动作序列长度
    )
    
    # 训练智能体
    episode_rewards, evaluations = train_multistep_td3(env, agent, config, replay_buffer)
    
    print(f"\n=== 训练结果总结 ===")
    print(f"配置: n_step={config.n_step}, action_horizon={config.action_horizon}")
    print(f"最佳评估分数: {max(evaluations):.3f}")
    print(f"最后10个episode平均奖励: {np.mean(episode_rewards[-10:]):.3f}")
    print(f"总共完成 {len(episode_rewards)} 个episode")
    print(f"评估历史: {[f'{eval_score:.3f}' for eval_score in evaluations]}")