#!/usr/bin/env python3
"""
MyReinFlow - 统一命令行界面
支持训练和运行不同类型的流模型与强化学习算法
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# 动态导入，避免在导入时出错
def import_modules():
    """动态导入训练模块"""
    try:
        from scripts.train_behavioral_cloning import main as bc_main
        from scripts.train_fql import main as fql_main
        from scripts.train_ppo_flow import main as ppo_main
        from examples.mean_flow_2d_example import main as meanflow_2d_main
        return bc_main, fql_main, ppo_main, meanflow_2d_main
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖都已正确安装")
        sys.exit(1)


def create_parser():
    """创建主命令行解析器"""
    parser = argparse.ArgumentParser(
        description="MyReinFlow - 流模型与强化学习训练工具包",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s bc --dataset mujoco/pusher/expert-v0 --epochs 50
  %(prog)s fql --dataset mujoco/pusher/expert-v0 --epochs 20 
  %(prog)s mean-flow-2d --epochs 10000
  %(prog)s --help                # 显示帮助信息
        """
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='可用的训练命令',
        metavar='COMMAND'
    )
    
    # 行为克隆子命令
    bc_parser = subparsers.add_parser(
        'bc', 
        help='训练行为克隆模型',
        description='使用Minari数据集训练行为克隆模型'
    )
    add_bc_args(bc_parser)
    
    # FQL子命令
    fql_parser = subparsers.add_parser(
        'fql',
        help='训练FQL模型', 
        description='训练Flow Q-Learning模型'
    )
    add_fql_args(fql_parser)
    
    # PPO Flow 子命令
    ppo_parser = subparsers.add_parser(
        'ppo',
        help='训练PPO Flow模型',
        description='使用PPO优化Flow策略的在线训练'
    )
    add_ppo_args(ppo_parser)
    
    # 2D MeanFlow示例子命令
    meanflow_parser = subparsers.add_parser(
        'mean-flow-2d',
        help='运行2D MeanFlow可视化示例',
        description='在2D数据上演示MeanFlow算法'
    )
    add_meanflow_2d_args(meanflow_parser)
    
    return parser


def add_bc_args(parser):
    """添加行为克隆相关参数"""
    parser.add_argument('--dataset', type=str, default='mujoco/pusher/expert-v0',
                       help='Minari数据集名称')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--eval_freq', type=int, default=5,
                       help='评估频率(轮数)')
    parser.add_argument('--eval_episodes', type=int, default=5,
                       help='评估轮次数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--no_wandb', action='store_true',
                       help='禁用WandB日志记录')
    parser.add_argument('--render', type=str, default='none',
                       choices=['none', 'human', 'rgb_array'],
                       help='渲染模式')
    parser.add_argument('--record_video_dir', type=str, default='',
                       help='视频录制目录')


def add_fql_args(parser):
    """添加FQL相关参数"""
    parser.add_argument('--dataset', type=str, default='mujoco/pusher/expert-v0',
                       help='Minari数据集名称')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批大小')
    parser.add_argument('--cond_steps', type=int, default=1,
                       help='条件观测步数')
    parser.add_argument('--horizon_steps', type=int, default=4,
                       help='预测动作步数')
    parser.add_argument('--lr_flow', type=float, default=3e-4,
                       help='流模型学习率')
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                       help='Actor学习率')
    parser.add_argument('--lr_critic', type=float, default=3e-4,
                       help='Critic学习率')
    parser.add_argument('--eval_freq', type=int, default=5,
                       help='评估频率')
    parser.add_argument('--eval_episodes', type=int, default=5,
                       help='评估轮次数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--render', type=str, default='none',
                       choices=['none', 'human', 'rgb_array'],
                       help='渲染模式')
    parser.add_argument('--record_video_dir', type=str, default='',
                       help='视频录制目录')
    parser.add_argument('--no_wandb', action='store_true',
                       help='禁用WandB日志记录')
    parser.add_argument('--only_optimize_bc_flow', action='store_true',
                       help='仅优化行为克隆流模型')
    parser.add_argument('--target_q_agg', type=str, default='min',
                       choices=['min', 'mean'],
                       help='目标Q值聚合方式')


def add_ppo_args(parser):
    """添加PPO Flow相关参数"""
    parser.add_argument('--env', type=str, default='Pendulum-v1', help='Gymnasium环境ID')
    parser.add_argument('--total_timesteps', type=int, default=50000, help='总采样步数')
    parser.add_argument('--rollout_steps', type=int, default=2048, help='每次rollout步数')
    parser.add_argument('--update_epochs', type=int, default=10, help='每次更新的epoch数')
    parser.add_argument('--minibatch_size', type=int, default=256, help='小批量大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--inference_steps', type=int, default=32, help='流采样步数')
    parser.add_argument('--horizon_steps', type=int, default=1, help='动作预测步数(通常为1)')
    parser.add_argument('--cond_steps', type=int, default=1, help='条件观测步数')
    parser.add_argument('--actor_policy_path', type=str, default=None, help='预训练Actor策略权重路径（可选）')


def add_meanflow_2d_args(parser):
    """添加2D MeanFlow示例参数"""
    parser.add_argument('--epochs', type=int, default=10000,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='计算设备')


def main():
    """主入口函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    print(f"🚀 启动 MyReinFlow - {args.command.upper()} 模式")
    print("-" * 50)
    
    # 动态导入模块
    bc_main, fql_main, ppo_main, meanflow_2d_main = import_modules()
    
    try:
        if args.command == 'bc':
            bc_main(args)
        elif args.command == 'fql':
            fql_main(args)
        elif args.command == 'ppo':
            ppo_main([
                '--env', args.env,
                '--total_timesteps', str(args.total_timesteps),
                '--rollout_steps', str(args.rollout_steps),
                '--update_epochs', str(args.update_epochs),
                '--minibatch_size', str(args.minibatch_size),
                '--seed', str(args.seed),
                '--inference_steps', str(args.inference_steps),
                '--horizon_steps', str(args.horizon_steps),
                '--cond_steps', str(args.cond_steps),
                *(["--actor_policy_path", args.actor_policy_path] if getattr(args, 'actor_policy_path', None) else []),
            ])
        elif args.command == 'mean-flow-2d':
            meanflow_2d_main(args)
        else:
            print(f"未知命令: {args.command}")
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断训练")
        return 1
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    print("✅ 任务完成!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
