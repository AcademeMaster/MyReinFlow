# Updated main.py (integrated Minari loading, DataModule, training, and online evaluation)
# !/usr/bin/env python3
import argparse

# 第三方库导入
import collections
import lightning as L
import minari
import numpy as np
import torch
# Lightning callbacks
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# 本地模块导入
from config import Config
from dataset import MinariDataModule, SlidingWindowDataset
from meanflow_ql import LitMeanFQL


# PyTorch imports

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于流匹配的机器人操作训练")
    parser.add_argument("mode", nargs='?', default="train", choices=["train", "test"],
                        help="运行模式: train 或 test")
    parser.add_argument("--dataset", default="mujoco/pusher/expert-v0", help="Minari数据集名称")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2048, help="批量大小")
    parser.add_argument("--checkpoint", help="测试时使用的模型路径")
    parser.add_argument("--test-episodes", type=int, default=20, help="测试轮数")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--normalize", action="store_true", default=False, help="启用数据归一化")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="禁用数据归一化")
    # Accelerator相关参数
    parser.add_argument("--mixed-precision", type=str, choices=["32-true", "16-mixed", "bf16-mixed"], default="32-true",
                        help="混合精度训练 (32-true, 16-mixed 或 bf16-mixed)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="梯度累积步数")
    # 在线评估开关
    parser.add_argument("--skip-online-eval", action="store_true", default=False,
                        help="跳过在线评估")
    args = parser.parse_args()

    # 初始化配置
    config = Config(
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        test_episodes=args.test_episodes,
        learning_rate=args.learning_rate,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # 覆盖数据归一化设置（如果指定了）
    if hasattr(args, 'normalize'):
        config.normalize_data = args.normalize

    print("=" * 50)
    print("配置参数:")
    print(config)
    print("=" * 50)

    # Infer obs_dim and action_dim from dataset
    minari_dataset = minari.load_dataset(config.dataset_name)
    sample_episode = next(minari_dataset.iterate_episodes())
    obs_dim = sample_episode.observations.shape[-1]
    action_dim = sample_episode.actions.shape[-1]
    config.action_dim = action_dim
    config.observation_dim=obs_dim

    dm = MinariDataModule(config)
    # 确保在测试模式下也调用setup方法
    if args.mode == "test":
        dm.setup()

    model = LitMeanFQL(obs_dim, action_dim, config)

    # 创建回调函数
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath='checkpoint_t',
        filename='meanflow_ql-epoch{epoch:02d}-loss{val/loss:.2f}',
        save_top_k=3,
        save_last=True,
        mode='min',
    )
    early_stop = EarlyStopping(monitor="val/loss", patience=20, mode="min")

    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, early_stop],
        precision=config.mixed_precision,
        accumulate_grad_batches=config.gradient_accumulation_steps,
    )

    if args.mode == "train":
        # 训练模型
        trainer.fit(model=model, datamodule=dm)

    elif args.mode == "test":
        if args.checkpoint:
            model = LitMeanFQL.load_from_checkpoint(args.checkpoint, obs_dim=obs_dim, action_dim=action_dim,
                                                    cfg=config)
        else:
            print("Warning: No checkpoint provided, using untrained model for test.")

        # Offline test (action MSE on val data)
        print("\nOffline Testing (Action MSE):")
        trainer.test(model, dataloaders=dm.test_dataloader())




if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision('high')
    main()