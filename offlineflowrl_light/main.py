#!/usr/bin/env python3
import argparse
import torch
import numpy as np

from config import Config
from trainer import FlowModelTrainer, OfflineFlowRLTrainer
from tester import FlowModelTester, OfflineFlowRLTester


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于流匹配的机器人操作训练")
    parser.add_argument("mode", choices=["train", "test", "train_offline_rl", "test_offline_rl"], help="运行模式: train, test, train_offline_rl 或 test_offline_rl")
    parser.add_argument("--dataset", default="mujoco/humanoid/expert-v0", help="Minari数据集名称")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批量大小")
    parser.add_argument("--checkpoint", help="测试时使用的模型路径")
    parser.add_argument("--test-episodes", type=int, default=20, help="测试轮数")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--normalize", action="store_true", default=True, help="启用数据归一化")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="禁用数据归一化")
    parser.add_argument("--render", choices=["none", "human", "rgb_array"], default="human", 
                       help="测试时的渲染模式 (默认: human)")
    # Accelerator相关参数
    parser.add_argument("--mixed-precision", type=str, choices=["no", "fp16", "bf16"], default="fp16", 
                        help="混合精度训练 (no, fp16 或 bf16)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, 
                        help="梯度累积步数")
    args = parser.parse_args()
    
    # 初始化配置
    config = Config(
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        test_episodes=args.test_episodes,
        learning_rate=args.learning_rate
    )
    
    # 添加Accelerator相关配置属性
    config.mixed_precision = args.mixed_precision
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # 覆盖数据归一化设置（如果指定了）
    if args.normalize is not None:
        config.normalize_data = args.normalize
    
    print("="*50)
    print("配置参数:")
    print(config)
    print("="*50)
    
    if args.mode == "train":
        trainer = FlowModelTrainer(config)
        trainer.train()
    
    elif args.mode == "test":
        tester = FlowModelTester(config, args.checkpoint, render_mode=args.render)
        tester.test()
        
    elif args.mode == "train_offline_rl":
        trainer = OfflineFlowRLTrainer(config)
        trainer.train()
        
    elif args.mode == "test_offline_rl":
        tester = OfflineFlowRLTester(config, args.checkpoint, render_mode=args.render)
        tester.test()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()