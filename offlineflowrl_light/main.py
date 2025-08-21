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
from meanflow_ql import LitConservativeMeanFQL


# PyTorch imports


def evaluate_online(model: LitConservativeMeanFQL, config: Config, render_mode: str = "human"):
    """Online evaluation in the environment"""
    # 使用Minari数据集的恢复环境功能
    minari_dataset = minari.load_dataset(config.dataset_name)
    
    # 尝试使用指定的渲染模式恢复环境
    eval_env = None
    if render_mode is not None and render_mode != "none":
        try:
            eval_env = minari_dataset.recover_environment(eval_env=True, render_mode=render_mode)
        except TypeError:
            # 兼容旧版本不支持 render_mode 参数
            try:
                eval_env = minari_dataset.recover_environment(eval_env=True)
            except Exception:
                eval_env = minari_dataset.recover_environment()
    else:
        eval_env = minari_dataset.recover_environment(eval_env=True)

    total_rewards = []
    for ep in range(config.test_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        step = 0

        # 重置模型中的观测历史
        model.net.actor.reset_obs_history()

        while not done and step < config.max_steps:
            # Prepare observation tensor
            obs_tensor = torch.tensor(obs).float()  # [obs_dim]
            
            # 使用select_action方法获取单个动作，内部维护观测历史
            action = model.net.actor.select_action(obs_tensor, n_steps=config.inference_steps)  # [1, action_dim]
            action = action[0].cpu().numpy()  # [action_dim]

            # Denormalize if needed
            if config.normalize_data:
                # 直接使用minari数据集计算统计信息，避免创建SlidingWindowDataset
                all_actions = []
                for episode in minari_dataset.iterate_episodes():
                    all_actions.append(episode.actions)
                all_actions = np.concatenate(all_actions, axis=0)
                action_stats = {
                    "mean": all_actions.mean(axis=0),
                    "std": all_actions.std(axis=0) + 1e-8
                }
                # 反归一化动作
                action = action * action_stats["std"] + action_stats["mean"]

            next_obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            obs = next_obs
            step += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        print(f"Episode {ep + 1}: Reward = {episode_reward}")

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards) if len(total_rewards) > 1 else 0.0
    print(f"Average Reward over {config.test_episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")
    eval_env.close()
    return avg_reward


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于流匹配的机器人操作训练")
    parser.add_argument("mode", nargs='?', default="train", choices=["train", "test"],
                        help="运行模式: train 或 test")
    parser.add_argument("--dataset", default="mujoco/pusher/expert-v0", help="Minari数据集名称")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2048, help="批量大小")
    parser.add_argument("--checkpoint", help="测试时使用的模型路径")
    parser.add_argument("--test-episodes", type=int, default=5, help="测试轮数")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--normalize", action="store_true", default=True, help="启用数据归一化")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="禁用数据归一化")
    parser.add_argument("--render", choices=["none", "human", "rgb_array"], default="none",
                        help="测试时的渲染模式 (默认: none)")
    # Accelerator相关参数
    parser.add_argument("--mixed-precision", type=str, choices=["32-true", "16-mixed", "bf16-mixed"], default="32",
                        help="混合精度训练 (32-true, 16-mixed 或 bf16-mixed)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="梯度累积步数")
    # 在线评估开关
    parser.add_argument("--skip-online-eval", action="store_true", default=True,
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
    if args.normalize is not None:
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

    dm = MinariDataModule(config)
    # 确保在测试模式下也调用setup方法
    if args.mode == "test":
        dm.setup()

    model = LitConservativeMeanFQL(obs_dim, action_dim, config)

    # 创建回调函数
    checkpoint_callback = ModelCheckpoint(
        monitor='val/critic_loss',
        dirpath='checkpoints/meanflow_ql',
        filename='meanflow_ql-epoch{epoch:02d}-val_critic_loss{val/critic_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    early_stop = EarlyStopping(monitor="val/critic_loss", patience=5, mode="min")

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
            model = LitConservativeMeanFQL.load_from_checkpoint(args.checkpoint, obs_dim=obs_dim, action_dim=action_dim,
                                                                cfg=config)
        else:
            print("Warning: No checkpoint provided, using untrained model for test.")

        # Offline test (action MSE on val data)
        print("\nOffline Testing (Action MSE):")
        trainer.test(model, dataloaders=dm.val_dataloader())

        # Online evaluation
        if not args.skip_online_eval:
            print("\nOnline Evaluation:")
            evaluate_online(model, config, render_mode=args.render)
        else:
            print("\nSkipping online evaluation as requested.")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision('high')
    main()