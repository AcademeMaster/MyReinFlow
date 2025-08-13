import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import wandb


def set_seed(seed):
    """设置随机种子以确保可重复性
    
    参数:
        seed (int): 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_cond_to_device(cond, device):
    """将条件字典移动到指定设备
    
    参数:
        cond (dict): 条件字典
        device (torch.device): 目标设备
        
    返回:
        dict: 移动到目标设备的条件字典
    """
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cond.items()}


def create_output_dir(base_dir='./outputs', experiment_name=None):
    """创建输出目录
    
    参数:
        base_dir (str): 基础目录
        experiment_name (str, optional): 实验名称
        
    返回:
        str: 输出目录路径
    """
    if experiment_name is None:
        experiment_name = f"experiment_{int(time.time())}"
    
    output_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def plot_training_curves(train_losses, eval_rewards=None, save_path=None):
    """绘制训练曲线
    
    参数:
        train_losses (list): 训练损失列表
        eval_rewards (list, optional): 评估奖励列表
        save_path (str, optional): 保存路径
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制训练损失
    plt.subplot(1, 2 if eval_rewards is not None else 1, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 绘制评估奖励
    if eval_rewards is not None:
        plt.subplot(1, 2, 2)
        plt.plot(eval_rewards)
        plt.title('Evaluation Reward')
        plt.xlabel('Evaluation')
        plt.ylabel('Reward')
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def init_wandb(project_name, run_name, config):
    """初始化WandB
    
    参数:
        project_name (str): 项目名称
        run_name (str): 运行名称
        config (dict): 配置字典
        
    返回:
        wandb.Run: WandB运行实例
    """
    return wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        reinit=True
    )


def log_model_params(model, prefix='model'):
    """记录模型参数
    
    参数:
        model (nn.Module): 模型
        prefix (str): 前缀
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{prefix} parameters: {param_count:,} (trainable: {trainable_param_count:,})")
    
    if wandb.run is not None:
        wandb.log({
            f"{prefix}/param_count": param_count,
            f"{prefix}/trainable_param_count": trainable_param_count,
        })


def log_final_metrics(results, prefix='model'):
    """记录最终指标
    
    参数:
        results (dict): 结果字典
        prefix (str): 前缀
    """
    if wandb.run is not None:
        metrics = {
            f"{prefix}/best_epoch": results.get('best_epoch', 0),
            f"{prefix}/best_loss": results.get('best_loss', float('inf')),
            f"{prefix}/best_reward": results.get('best_reward', float('-inf')),
        }
        
        wandb.log(metrics)
        
        # 创建摘要指标
        for key, value in metrics.items():
            wandb.run.summary[key] = value