"""多步TD3配置示例

这个文件展示了如何配置和使用多步TD3算法。
包含了不同的配置示例，用户可以根据需要选择合适的配置。
"""

from main import TD3Config, setup_environment
from TD3 import TD3
from utils import train_td3_timestep
import gymnasium as gym

# 示例1: 基础多步配置
def get_basic_multistep_config():
    """基础多步学习配置"""
    config = TD3Config(
        # 基础参数
        max_timesteps=50000,
        start_timesteps=1000,
        batch_size=256,
        eval_freq=5000,
        
        # 多步学习参数
        use_multistep=True,
        n_step=4,  # 4步TD学习
        action_horizon=1,  # 单步动作（标准TD3）
        
        # 环境参数
        env_name='Pendulum-v1',
        seed=42
    )
    return config

# 示例2: 动作序列配置
def get_action_chunking_config():
    """动作序列（Action Chunking）配置"""
    config = TD3Config(
        # 基础参数
        max_timesteps=50000,
        start_timesteps=1000,
        batch_size=256,
        eval_freq=5000,
        
        # 动作序列参数
        use_multistep=False,  # 不使用多步TD，只使用动作序列
        n_step=1,
        action_horizon=4,  # 4步动作序列
        
        # 环境参数
        env_name='Pendulum-v1',
        seed=42
    )
    return config

# 示例3: 完整多步+动作序列配置
def get_full_multistep_config():
    """完整多步学习+动作序列配置"""
    config = TD3Config(
        # 基础参数
        max_timesteps=50000,
        start_timesteps=1000,
        batch_size=256,
        eval_freq=5000,
        
        # 完整多步参数
        use_multistep=True,
        n_step=4,  # 4步TD学习
        action_horizon=4,  # 4步动作序列
        
        # 环境参数
        env_name='Pendulum-v1',
        seed=42
    )
    return config

# 示例4: 保守配置（适合初学者）
def get_conservative_config():
    """保守配置，适合初学者和调试"""
    config = TD3Config(
        # 基础参数
        max_timesteps=20000,  # 较少的训练步数
        start_timesteps=500,
        batch_size=128,  # 较小的批次
        eval_freq=2000,
        
        # 保守的多步参数
        use_multistep=True,
        n_step=2,  # 较短的步数
        action_horizon=2,  # 较短的动作序列
        
        # 环境参数
        env_name='Pendulum-v1',
        seed=42
    )
    return config

def run_example(config_name="basic"):
    """运行指定配置的示例
    
    Args:
        config_name: 配置名称，可选 'basic', 'action_chunking', 'full', 'conservative'
    """
    
    # 选择配置
    if config_name == "basic":
        config = get_basic_multistep_config()
        print("=== 运行基础多步学习示例 ===")
    elif config_name == "action_chunking":
        config = get_action_chunking_config()
        print("=== 运行动作序列示例 ===")
    elif config_name == "full":
        config = get_full_multistep_config()
        print("=== 运行完整多步+动作序列示例 ===")
    elif config_name == "conservative":
        config = get_conservative_config()
        print("=== 运行保守配置示例 ===")
    else:
        raise ValueError(f"未知配置: {config_name}")
    
    print(f"配置详情:")
    print(f"  - 多步学习: {config.use_multistep}")
    print(f"  - n_step: {config.n_step}")
    print(f"  - action_horizon: {config.action_horizon}")
    print(f"  - max_timesteps: {config.max_timesteps}")
    print(f"  - batch_size: {config.batch_size}")
    print()
    
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
        action_horion=config.action_horizon
    )
    
    # 训练智能体
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
    
    print(f"\n=== 训练完成 ===")
    print(f"最佳评估分数: {max(evaluations):.3f}")
    print(f"最后10个episode平均奖励: {np.mean(episode_rewards[-10:]):.3f}")
    
    return episode_rewards, evaluations

if __name__ == "__main__":
    import numpy as np
    
    # 运行不同配置的示例
    print("多步TD3配置示例")
    print("可用配置: basic, action_chunking, full, conservative")
    print()
    
    # 默认运行基础配置
    run_example("basic")
    
    # 如果想运行其他配置，取消下面的注释
    # run_example("action_chunking")
    # run_example("full")
    # run_example("conservative")