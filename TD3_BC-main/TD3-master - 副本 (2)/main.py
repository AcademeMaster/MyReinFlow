from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import ReplayBuffer, MultiStepReplayBuffer, train_off_policy_agent, train_td3_timestep
import random
from TD3 import TD3

@dataclass
class TD3Config:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    # 官方TD3参数设置
    max_timesteps: int = 100000  # 最大时间步数（官方默认1e6，这里设小一点测试）
    start_timesteps: int = 1000  # 初始随机探索步数（官方默认25e3）
    hidden_dim: int = 256  # 网络维度
    gamma: float = 0.99  # 折扣因子
    tau: float = 0.005  # 软更新参数
    buffer_size: int = 1000000  # 缓冲区大小
    minimal_size: int = 1000
    batch_size: int = 256  # 批次大小（官方默认256）
    expl_noise: float = 0.1  # 探索噪声标准差
    env_name: str = 'Pendulum-v1'
    seed: int = 0
    policy_noise: float = 0.2  # TD3特有参数
    noise_clip: float = 0.5
    policy_freq: int = 2
    eval_freq: int = 5000  # 评估频率
    # 兼容旧版本的参数
    num_episodes: int = 300
    sigma: float = 0.1
    update_freq: int = 4
    gradient_steps: int = 2
    # 多步学习参数
    use_multistep: bool = False  # 是否使用多步学习
    action_horizon: int = 1  # 动作序列长度（action chunking和多步学习统一参数）

@dataclass
class EnvState:
    state_dim: int
    action_dim: int
    action_bound: float
    device: torch.device
    replay_buffer: ReplayBuffer

def setup_environment(config: TD3Config) -> EnvState:
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
    action_bound = env.action_space.high[0]  # 动作最大值
    
    # 根据配置选择缓冲区类型
    if config.use_multistep:
        replay_buffer = MultiStepReplayBuffer(config.buffer_size, device, config.action_horizon)
        print(f"使用多步缓冲区: action_horizon={config.action_horizon}")
    else:
        replay_buffer = ReplayBuffer(config.buffer_size, device)
        print("使用单步缓冲区")
    
    return EnvState(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        device=device,
        replay_buffer=replay_buffer
    )


if __name__ == "__main__":
    # 初始化配置
    config = TD3Config()

    # 设置环境
    env_state = setup_environment(config)
    env = gym.make(config.env_name)

    # 创建智能体，注入replay_buffer
    agent = TD3(
        env_state.state_dim,
        config.hidden_dim,
        env_state.action_dim,
        env_state.action_bound,
        config.sigma,
        config.actor_lr,
        config.critic_lr,
        config.tau,
        config.gamma,
        env_state.device,
        config.policy_noise,
        config.noise_clip,
        config.policy_freq,
        env_state.replay_buffer,
        action_horizon=config.action_horizon  # 设置动作序列长度
    )
    
    print(f"创建TD3智能体: action_horizon={config.action_horizon}")
    if config.use_multistep:
        print(f"多步学习配置: action_horizon={config.action_horizon}")
    else:
        print("单步学习配置")

    # 训练智能体
    # 使用官方TD3的基于时间步训练方式
    print("开始使用官方TD3训练方式...")
    episode_rewards, evaluations = train_td3_timestep(
        env,
        agent,
        config.max_timesteps,
        env_state.replay_buffer,
        config.minimal_size,
        config.batch_size,
        config.start_timesteps,
        config.expl_noise,
        config.eval_freq,
        config.env_name,
        config.seed
    )
    
    print(f"训练完成，平均回报: {np.mean(episode_rewards[-10:]):.3f}")
    print(f"总共完成 {len(episode_rewards)} 个episode")
    print(f"评估历史: {[f'{eval_score:.3f}' for eval_score in evaluations]}")
    print(f"最佳评估分数: {max(evaluations):.3f}")
    
    # 可选：也可以使用原来的训练方式进行对比
    # return_list = train_off_policy_agent(
    #     env, 
    #     agent, 
    #     config.num_episodes, 
    #     env_state.replay_buffer, 
    #     config.minimal_size, 
    #     config.batch_size,
    #     config.update_freq,
    #     config.gradient_steps
    # )