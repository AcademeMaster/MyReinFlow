from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import ReplayBuffer, MultiStepReplayBuffer, train_off_policy_agent, train_td3_timestep, eval_policy
import random
from TD3 import TD3
import argparse
import os
import json
from datetime import datetime

@dataclass
class TD3Config:
    # 学习率参数
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    
    # 训练参数
    max_timesteps: int = int(2e5)  # 最大时间步数
    start_timesteps: int = 1000  # 初始随机探索步数
    hidden_dim: int = 256  # 网络维度
    gamma: float = 0.99  # 折扣因子
    tau: float = 0.005  # 软更新参数
    
    # 缓冲区参数
    buffer_size: int = 1000000  # 缓冲区大小
    minimal_size: int = 1000
    batch_size: int = 256  # 批次大小
    
    # 探索参数
    expl_noise: float = 0.1  # 探索噪声标准差
    policy_noise: float = 0.5  # TD3策略噪声
    noise_clip: float = 1.0  # 噪声裁剪
    
    # 环境参数
    env_name: str = 'Ant-v5'
    seed: int = 0
    
    # TD3特有参数
    policy_freq: int = 2  # 策略更新频率
    eval_freq: int = 5000  # 评估频率
    sigma: float = 0.2  # 动作噪声
    action_horizon: int = 4  # 动作序列长度
    
    # 实验参数
    save_model: bool = True  # 是否保存模型
    save_freq: int = 50000  # 模型保存频率
    log_freq: int = 1000  # 日志记录频率
    render_eval: bool = False  # 评估时是否渲染
    exp_name: str = "td3_experiment"  # 实验名称
    
    def save_config(self, save_path: str):
        """保存配置到JSON文件"""
        config_dict = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, config_path: str):
        """从JSON文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

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
    replay_buffer = MultiStepReplayBuffer(config.buffer_size, device, config.action_horizon)

    
    return EnvState(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        device=device,
        replay_buffer=replay_buffer
    )

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TD3 强化学习实验')
    
    # 基本参数
    parser.add_argument('--env_name', type=str, default='Ant-v5', help='环境名称')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--max_timesteps', type=int, default=200000, help='最大训练步数')
    parser.add_argument('--exp_name', type=str, default='td3_experiment', help='实验名称')
    
    # 网络参数
    parser.add_argument('--actor_lr', type=float, default=3e-4, help='Actor学习率')
    parser.add_argument('--critic_lr', type=float, default=3e-4, help='Critic学习率')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--tau', type=float, default=0.005, help='软更新参数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='经验回放缓冲区大小')
    parser.add_argument('--start_timesteps', type=int, default=1000, help='随机探索步数')
    parser.add_argument('--action_horizon', type=int, default=4, help='动作序列长度')
    
    # TD3特有参数
    parser.add_argument('--policy_noise', type=float, default=0.5, help='策略噪声')
    parser.add_argument('--noise_clip', type=float, default=1.0, help='噪声裁剪')
    parser.add_argument('--policy_freq', type=int, default=2, help='策略更新频率')
    parser.add_argument('--expl_noise', type=float, default=0.1, help='探索噪声')
    
    # 评估和保存参数
    parser.add_argument('--eval_freq', type=int, default=5000, help='评估频率')
    parser.add_argument('--save_freq', type=int, default=50000, help='模型保存频率')
    parser.add_argument('--log_freq', type=int, default=1000, help='日志记录频率')
    parser.add_argument('--render_eval', action='store_true', help='评估时是否渲染')
    parser.add_argument('--save_model', action='store_true', default=True, help='是否保存模型')
    
    # 配置文件
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--save_config', action='store_true', help='保存当前配置')
    
    return parser.parse_args()

def create_experiment_dir(exp_name: str) -> str:
    """创建实验目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/{exp_name}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)
    return exp_dir


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 初始化配置
    if args.config:
        # 从配置文件加载
        config = TD3Config.load_config(args.config)
        print(f"从配置文件加载: {args.config}")
    else:
        # 使用命令行参数创建配置
        config = TD3Config(
            env_name=args.env_name,
            seed=args.seed,
            max_timesteps=args.max_timesteps,
            exp_name=args.exp_name,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            tau=args.tau,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            start_timesteps=args.start_timesteps,
            action_horizon=args.action_horizon,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
            expl_noise=args.expl_noise,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            log_freq=args.log_freq,
            render_eval=args.render_eval,
            save_model=args.save_model
        )
    
    # 创建实验目录
    exp_dir = create_experiment_dir(config.exp_name)
    print(f"实验目录: {exp_dir}")
    
    # 保存配置
    config.save_config(f"{exp_dir}/config.json")
    if args.save_config:
        config.save_config("config.json")
        print("配置已保存到 config.json")
    
    # 打印实验配置
    print("\n=== 实验配置 ===")
    print(f"环境: {config.env_name}")
    print(f"随机种子: {config.seed}")
    print(f"最大训练步数: {config.max_timesteps}")
    print(f"动作序列长度: {config.action_horizon}")
    print(f"批次大小: {config.batch_size}")
    print(f"Actor学习率: {config.actor_lr}")
    print(f"Critic学习率: {config.critic_lr}")
    print("================\n")

    # 设置环境
    env_state = setup_environment(config)
    env = gym.make(config.env_name)

    # 创建智能体
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
        action_horizon=config.action_horizon
    )
    
    print(f"创建TD3智能体: action_horizon={config.action_horizon}")
    print(f"多步学习配置: action_horizon={config.action_horizon}")

    # 训练智能体
    print("开始训练...")
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
    
    # 保存训练结果
    results = {
        "episode_rewards": episode_rewards,
        "evaluations": evaluations,
        "final_avg_reward": float(np.mean(episode_rewards[-10:])),
        "best_eval_score": float(max(evaluations)) if evaluations else 0.0,
        "total_episodes": len(episode_rewards)
    }
    
    with open(f"{exp_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== 训练结果 ===")
    print(f"训练完成，平均回报: {results['final_avg_reward']:.3f}")
    print(f"总共完成 {results['total_episodes']} 个episode")
    print(f"最佳评估分数: {results['best_eval_score']:.3f}")
    print(f"结果已保存到: {exp_dir}")
    print("================\n")

    # 最终评估，打开渲染
    if config.render_eval:
        print("开始最终评估（带渲染）...")
        eval_policy(agent, config.env_name, config.seed, render=True)
    
    # 保存最终模型
    if config.save_model:
        model_path = f"{exp_dir}/models/final_model.pth"
        torch.save({
            'actor_state_dict': agent.actor.state_dict(),
            'critic_1_state_dict': agent.critic_1.state_dict(),
            'critic_2_state_dict': agent.critic_2.state_dict(),
            'config': config.__dict__
        }, model_path)
        print(f"模型已保存到: {model_path}")
    
