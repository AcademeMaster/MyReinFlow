import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import minari
import time
import wandb

from reinflow.models.flow_mlp import FlowMLP
from reinflow.models.reflow import ReFlow
from reinflow.models.mean_flow import MeanFlow
from reinflow.data.minari_dataset import MinariDataset
from reinflow.trainers.evaluators import evaluate_reflow, evaluate_mean_flow, compare_methods, create_generation_visualization
from reinflow.utils.helpers import set_seed, init_wandb


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='Evaluate ReFlow and MeanFlow models')
    parser.add_argument('--dataset', type=str, default='hopper-medium-expert-v2', help='Minari dataset name')
    parser.add_argument('--horizon_steps', type=int, default=10, help='Horizon steps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--inference_steps', type=int, default=20, help='Inference steps')
    parser.add_argument('--noise_schedule', type=str, default='prioritized', help='Noise schedule')
    parser.add_argument('--normalize', action='store_true', help='Normalize data')
    parser.add_argument('--method', type=str, default='both', choices=['reflow', 'meanflow', 'both'], help='Evaluation method')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb_project', type=str, default='reinflow_eval', help='WandB project name')
    parser.add_argument('--wandb_run', type=str, default=None, help='WandB run name')
    parser.add_argument('--model_dir', type=str, default='./outputs', help='Model directory')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render evaluation')
    parser.add_argument('--record_video', action='store_true', help='Record evaluation video')
    parser.add_argument('--compare', action='store_true', help='Compare methods')
    parser.add_argument('--visualize', action='store_true', help='Create generation visualization')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 初始化WandB
    if args.wandb_run is None:
        args.wandb_run = f"eval_{args.method}_{args.dataset}_{int(time.time())}"
    
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
    if args.record_video:
        # 如果需要录制视频，重新创建环境并设置render_mode
        env_spec = env.spec
        env.close()  # 关闭当前环境
        env = gym.make(env_spec.id, render_mode='rgb_array')
    
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
    
    # 创建数据集（用于可视化）
    dataset = MinariDataset(
        dataset_name=args.dataset,
        horizon_steps=args.horizon_steps,
        device=device,
        normalize=args.normalize,
        max_samples=100,  # 只需要少量样本用于可视化
        seed=args.seed
    )
    
    # 加载ReFlow模型
    reflow_model = None
    if args.method in ['reflow', 'both']:
        print("\n" + "=" * 50)
        print("Loading ReFlow model")
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
        
        # 加载模型权重
        model_path = os.path.join(args.model_dir, 'reflow_best_model.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.model_dir, 'reflow_final_model.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.model_dir, 'reflow_model.pt')
        
        if os.path.exists(model_path):
            print(f"Loading ReFlow model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                reflow_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                reflow_model.load_state_dict(checkpoint)
        else:
            print(f"Warning: ReFlow model not found at {model_path}. Using untrained model.")
        
        # 评估ReFlow模型
        print("\nEvaluating ReFlow model...")
        reflow_reward = evaluate_reflow(
            model=reflow_model,
            env=env,
            device=device,
            inference_steps=args.inference_steps,
            num_episodes=args.num_episodes,
            render=args.render,
            record_video=args.record_video,
            wandb_log=True
        )
        
        print(f"ReFlow evaluation result: {reflow_reward:.2f}")
    
    # 加载MeanFlow模型
    meanflow_model = None
    if args.method in ['meanflow', 'both']:
        print("\n" + "=" * 50)
        print("Loading MeanFlow model")
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
        
        # 加载模型权重
        model_path = os.path.join(args.model_dir, 'meanflow_best_model.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.model_dir, 'meanflow_final_model.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.model_dir, 'meanflow_model.pt')
        
        if os.path.exists(model_path):
            print(f"Loading MeanFlow model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                meanflow_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                meanflow_model.load_state_dict(checkpoint)
        else:
            print(f"Warning: MeanFlow model not found at {model_path}. Using untrained model.")
        
        # 评估MeanFlow模型
        print("\nEvaluating MeanFlow model...")
        meanflow_reward = evaluate_mean_flow(
            model=meanflow_model,
            env=env,
            device=device,
            inference_steps=1,  # MeanFlow通常只需要1步
            num_episodes=args.num_episodes,
            render=args.render,
            record_video=args.record_video,
            wandb_log=True
        )
        
        print(f"MeanFlow evaluation result: {meanflow_reward:.2f}")
    
    # 比较两种方法
    if args.method == 'both' and args.compare:
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
            num_episodes=args.num_episodes,
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
    
    # 创建生成可视化
    if args.method == 'both' and args.visualize:
        print("\n" + "=" * 50)
        print("Creating generation visualization")
        print("=" * 50)
        
        visualization_results = create_generation_visualization(
            reflow_model=reflow_model,
            meanflow_model=meanflow_model,
            dataset=dataset,
            device=device,
            num_samples=5,
            wandb_log=True
        )
        
        print("Generation visualization created and logged to WandB")
    
    # 关闭环境
    env.close()
    
    # 关闭WandB
    wandb.finish()
    
    print("\nEvaluation completed successfully!")


if __name__ == '__main__':
    main()