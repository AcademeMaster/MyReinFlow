import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import minari
import time
import wandb

from reinflow.models.flow_mlp import FlowMLP
from reinflow.models.reflow import ReFlow
from reinflow.models.mean_flow import MeanFlow
from reinflow.data.minari_dataset import MinariDataset
from reinflow.trainers.trainers import train_reflow, train_mean_flow
from reinflow.trainers.evaluators import evaluate_reflow, evaluate_mean_flow, compare_methods
from reinflow.utils.helpers import set_seed, create_output_dir, init_wandb, log_model_params, log_final_metrics

from torch.utils.data import DataLoader


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='Train ReFlow and MeanFlow models')
    parser.add_argument('--dataset', type=str, default='hopper-medium-expert-v2', help='Minari dataset name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--horizon_steps', type=int, default=10, help='Horizon steps')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--eval_freq', type=int, default=5, help='Evaluation frequency')
    parser.add_argument('--inference_steps', type=int, default=20, help='Inference steps')
    parser.add_argument('--noise_schedule', type=str, default='prioritized', help='Noise schedule')
    parser.add_argument('--early_stop_patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--normalize', action='store_true', help='Normalize data')
    parser.add_argument('--method', type=str, default='reflow', choices=['reflow', 'meanflow', 'both'], help='Training method')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb_project', type=str, default='reinflow', help='WandB project name')
    parser.add_argument('--wandb_run', type=str, default=None, help='WandB run name')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples')
    parser.add_argument('--render', action='store_true', help='Render evaluation')
    parser.add_argument('--record_video', action='store_true', help='Record evaluation video')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 初始化WandB
    if args.wandb_run is None:
        args.wandb_run = f"{args.method}_{args.dataset}_{int(time.time())}"
    
    wandb_config = vars(args)
    init_wandb(args.wandb_project, args.wandb_run, wandb_config)
    
    # 加载Minari数据集
    try:
        print(f"Loading Minari dataset: {args.dataset}")
        minari_dataset = minari.load_dataset(args.dataset)
        print(f"Dataset loaded successfully: {args.dataset}")
    except Exception as e:
        print(f"Error loading dataset {args.dataset}: {e}")
        print("Trying to load a fallback dataset...")
        minari_dataset = None
        # 尝试加载备用数据集
        fallback_datasets = ['hopper-medium-v2', 'halfcheetah-medium-v2', 'walker2d-medium-v2', 'mujoco/pusher/medium-v0']
        for fallback in fallback_datasets:
            try:
                print(f"Trying fallback dataset: {fallback}")
                minari_dataset = minari.load_dataset(fallback)
                args.dataset = fallback
                print(f"Successfully loaded fallback dataset: {fallback}")
                break
            except Exception as fallback_e:
                print(f"Failed to load {fallback}: {fallback_e}")
                continue
        
        if minari_dataset is None:
            raise Exception("Could not load any dataset")
    
    # 恢复环境
    print(f"Recovering environment from dataset...")
    env = minari_dataset.recover_environment()
    eval_env = minari_dataset.recover_environment(eval_env=True)
    
    # 获取环境参数
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    act_min = float(env.action_space.low.min())
    act_max = float(env.action_space.high.max())
    
    # 获取环境ID
    env_id = env.spec.id if hasattr(env.spec, 'id') else str(env.spec)
    
    print(f"Environment: {env_id}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Action range: [{act_min}, {act_max}]")
    
    wandb.log({
        'env/id': env_id,
        'env/obs_dim': obs_dim,
        'env/action_dim': action_dim,
        'env/act_min': act_min,
        'env/act_max': act_max,
    })
    
    # 创建数据集
    dataset = MinariDataset(
        dataset_name=args.dataset,
        horizon_steps=args.horizon_steps,
        device=device,
        normalize=args.normalize,
        max_samples=args.max_samples,
        seed=args.seed
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 避免多进程问题
        drop_last=True
    )
    
    # 创建输出目录
    output_dir = create_output_dir(args.output_dir, args.wandb_run)
    print(f"Output directory: {output_dir}")
    
    # 训练ReFlow
    if args.method in ['reflow', 'both']:
        print("\n" + "=" * 50)
        print("Training ReFlow model")
        print("=" * 50)
        
        # 创建网络和模型
        reflow_network = FlowMLP(
            horizon_steps=args.horizon_steps,
            action_dim=action_dim,
            cond_dim=obs_dim,
            time_dim=32,
            mlp_dims=[512, 512, 256],
            activation_type="SiLU",
            dropout_rate=0.1
        )
        
        reflow_model = ReFlow(
            network=reflow_network,
            device=device,
            horizon_steps=args.horizon_steps,
            action_dim=action_dim,
            act_min=act_min,
            act_max=act_max,
            obs_dim=obs_dim,
            max_denoising_steps=args.inference_steps,
            seed=args.seed,
            noise_schedule=args.noise_schedule
        )
        
        # 记录模型参数
        log_model_params(reflow_model, prefix='reflow')
        
        # 创建优化器和学习率调度器
        optimizer = optim.AdamW(reflow_model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        
        # 训练模型
        reflow_results = train_reflow(
            model=reflow_model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=args.epochs,
            eval_env=env,
            eval_freq=args.eval_freq,
            output_dir=output_dir,
            early_stop_patience=args.early_stop_patience,
            inference_steps=args.inference_steps,
            wandb_log=True
        )
        
        # 记录最终指标
        log_final_metrics(reflow_results, prefix='reflow')
        
        # 保存模型
        torch.save(reflow_model.state_dict(), os.path.join(output_dir, 'reflow_model.pt'))
        print(f"ReFlow model saved to {os.path.join(output_dir, 'reflow_model.pt')}")
    
    # 训练MeanFlow
    if args.method in ['meanflow', 'both']:
        print("\n" + "=" * 50)
        print("Training MeanFlow model")
        print("=" * 50)
        
        # 创建网络和模型
        meanflow_network = FlowMLP(
            horizon_steps=args.horizon_steps,
            action_dim=action_dim,
            cond_dim=obs_dim,
            time_dim=32,
            mlp_dims=[512, 512, 256],
            activation_type="SiLU",
            dropout_rate=0.1
        )
        
        meanflow_model = MeanFlow(
            network=meanflow_network,
            device=device,
            horizon_steps=args.horizon_steps,
            action_dim=action_dim,
            act_min=act_min,
            act_max=act_max,
            obs_dim=obs_dim,
            max_denoising_steps=1,  # MeanFlow通常只需要1步
            seed=args.seed,
            noise_schedule=args.noise_schedule
        )
        
        # 记录模型参数
        log_model_params(meanflow_model, prefix='meanflow')
        
        # 创建优化器和学习率调度器
        optimizer = optim.AdamW(meanflow_model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        
        # 训练模型
        meanflow_results = train_mean_flow(
            model=meanflow_model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=args.epochs,
            eval_env=env,
            eval_freq=args.eval_freq,
            output_dir=output_dir,
            early_stop_patience=args.early_stop_patience,
            inference_steps=1,  # MeanFlow通常只需要1步
            wandb_log=True
        )
        
        # 记录最终指标
        log_final_metrics(meanflow_results, prefix='meanflow')
        
        # 保存模型
        torch.save(meanflow_model.state_dict(), os.path.join(output_dir, 'meanflow_model.pt'))
        print(f"MeanFlow model saved to {os.path.join(output_dir, 'meanflow_model.pt')}")
    
    # 比较两种方法
    if args.method == 'both':
        print("\n" + "=" * 50)
        print("Comparing ReFlow and MeanFlow methods")
        print("=" * 50)
        
        # 比较不同推理步数下的性能
        inference_steps_list = [1, 5, 10, 20, 50]
        comparison_results = compare_methods(
            reflow_model=reflow_model,
            meanflow_model=meanflow_model,
            env=env,
            device=device,
            inference_steps_list=inference_steps_list,
            num_episodes=10,
            wandb_log=True
        )
        
        print("\nComparison results:")
        for i, steps in enumerate(inference_steps_list):
            print(f"Inference steps: {steps}")
            print(f"ReFlow reward: {comparison_results['reflow_rewards'][i]:.2f}")
            print(f"MeanFlow reward: {comparison_results['meanflow_rewards'][i]:.2f}")
            print(f"ReFlow time: {comparison_results['reflow_times'][i]:.2f}s")
            print(f"MeanFlow time: {comparison_results['meanflow_times'][i]:.2f}s")
            print()
    
    # 关闭环境
    env.close()
    
    # 关闭WandB
    wandb.finish()
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()