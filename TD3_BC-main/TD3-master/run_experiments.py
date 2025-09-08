#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验运行脚本示例
展示如何使用不同的参数配置运行多个实验
"""

import subprocess
import sys
import os

def run_experiment(args_dict, exp_description=""):
    """运行单个实验"""
    print(f"\n{'='*50}")
    print(f"开始实验: {exp_description}")
    print(f"{'='*50}")
    
    # 构建命令行参数
    cmd = [sys.executable, "main.py"]
    for key, value in args_dict.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd.extend([f"--{key}", str(value)])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"实验完成: {exp_description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"实验失败: {exp_description}, 错误: {e}")
        return False

def main():
    """运行多个实验配置"""
    
    # 实验1: 基础配置
    exp1_args = {
        "env_name": "Ant-v5",
        "seed": 0,
        "max_timesteps": 100000,
        "exp_name": "baseline_experiment",
        "action_horizon": 1,  # 单步动作
        "batch_size": 256,
        "eval_freq": 10000
    }
    
    # 实验2: 多步动作配置
    exp2_args = {
        "env_name": "Ant-v5",
        "seed": 0,
        "max_timesteps": 100000,
        "exp_name": "multistep_experiment",
        "action_horizon": 4,  # 4步动作序列
        "batch_size": 256,
        "eval_freq": 10000
    }
    
    # 实验3: 不同学习率配置
    exp3_args = {
        "env_name": "Ant-v5",
        "seed": 0,
        "max_timesteps": 100000,
        "exp_name": "high_lr_experiment",
        "action_horizon": 4,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "batch_size": 256,
        "eval_freq": 10000
    }
    
    # 实验4: 使用配置文件
    exp4_args = {
        "config": "example_config.json",
        "exp_name": "config_file_experiment"
    }
    
    experiments = [
        (exp1_args, "基础单步实验"),
        (exp2_args, "多步动作实验"),
        (exp3_args, "高学习率实验"),
        (exp4_args, "配置文件实验")
    ]
    
    successful_experiments = 0
    total_experiments = len(experiments)
    
    for args, description in experiments:
        if run_experiment(args, description):
            successful_experiments += 1
        print(f"\n当前进度: {successful_experiments}/{total_experiments} 实验完成")
    
    print(f"\n{'='*50}")
    print(f"所有实验完成!")
    print(f"成功: {successful_experiments}/{total_experiments}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()