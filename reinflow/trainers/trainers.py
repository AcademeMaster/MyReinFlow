import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from torch.utils.data import DataLoader
import gymnasium as gym
import wandb

from .evaluators import evaluate_reflow, evaluate_mean_flow


def train_reflow(model, dataloader, optimizer, scheduler, device, epochs, 
                eval_env=None, eval_freq=10, output_dir='./outputs', 
                early_stop_patience=20, inference_steps=20, wandb_log=True):
    """训练ReFlow模型
    
    参数:
        model (nn.Module): ReFlow模型
        dataloader (DataLoader): 数据加载器
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器
        device (torch.device): 计算设备
        epochs (int): 训练轮次
        eval_env (gym.Env, optional): 评估环境
        eval_freq (int): 评估频率
        output_dir (str): 输出目录
        early_stop_patience (int): 早停耐心
        inference_steps (int): 推理步数
        wandb_log (bool): 是否使用WandB记录
        
    返回:
        dict: 训练结果
    """
    print(f"Training ReFlow model for {epochs} epochs...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练统计
    best_loss = float('inf')
    best_reward = float('-inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    eval_rewards = []
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        start_time = time.time()
        for batch_idx, batch in enumerate(dataloader):
            # 移动数据到设备
            cond = {k: v.to(device) for k, v in batch.items() if k != 'actions'}
            actions = batch['actions'].to(device)
            
            # 生成训练目标
            (xt, t), v = model.generate_target(actions)
            
            # 计算损失
            optimizer.zero_grad()
            loss = model.loss(xt, t, cond, v)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_losses.append(loss.item())
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.6f}")
        
        # 计算平均损失
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        # 更新学习率
        scheduler.step()
        
        # 记录到WandB
        if wandb_log:
            wandb.log({
                'reflow/train_loss': avg_loss,
                'reflow/learning_rate': optimizer.param_groups[0]['lr'],
                'reflow/epoch': epoch + 1,
            })
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{epochs} completed in {time.time() - start_time:.2f}s | "
              f"Avg Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 定期评估
        if eval_env is not None and (epoch + 1) % eval_freq == 0:
            eval_reward = evaluate_reflow(model, eval_env, device, inference_steps=inference_steps)
            eval_rewards.append(eval_reward)
            
            print(f"Evaluation at epoch {epoch+1}: Reward = {eval_reward:.2f}")
            
            if wandb_log:
                wandb.log({
                    'reflow/eval_reward': eval_reward,
                    'reflow/epoch': epoch + 1,
                })
            
            # 保存最佳模型（基于奖励）
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_epoch = epoch + 1
                patience_counter = 0
                
                # 保存模型
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'reward': eval_reward,
                }, os.path.join(output_dir, 'reflow_best_model.pt'))
                
                print(f"New best model saved at epoch {epoch+1} with reward {best_reward:.2f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} evaluations")
                
                # 早停
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            # 如果没有评估环境，则基于损失保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                # 保存模型
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(output_dir, 'reflow_best_model.pt'))
                
                print(f"New best model saved at epoch {epoch+1} with loss {best_loss:.6f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")
                
                # 早停
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    # 保存最终模型
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
    }, os.path.join(output_dir, 'reflow_final_model.pt'))
    
    print(f"Training completed. Best model at epoch {best_epoch}")
    
    return {
        'train_losses': train_losses,
        'eval_rewards': eval_rewards,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'best_reward': best_reward
    }


def train_mean_flow(model, dataloader, optimizer, scheduler, device, epochs, 
                   eval_env=None, eval_freq=10, output_dir='./outputs', 
                   early_stop_patience=20, inference_steps=1, wandb_log=True):
    """训练MeanFlow模型
    
    参数:
        model (nn.Module): MeanFlow模型
        dataloader (DataLoader): 数据加载器
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器
        device (torch.device): 计算设备
        epochs (int): 训练轮次
        eval_env (gym.Env, optional): 评估环境
        eval_freq (int): 评估频率
        output_dir (str): 输出目录
        early_stop_patience (int): 早停耐心
        inference_steps (int): 推理步数
        wandb_log (bool): 是否使用WandB记录
        
    返回:
        dict: 训练结果
    """
    print(f"Training MeanFlow model for {epochs} epochs...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练统计
    best_loss = float('inf')
    best_reward = float('-inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    eval_rewards = []
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        start_time = time.time()
        for batch_idx, batch in enumerate(dataloader):
            # 移动数据到设备
            cond = {k: v.to(device) for k, v in batch.items() if k != 'actions'}
            actions = batch['actions'].to(device)
            
            # 生成训练目标
            (xt0, xt1, t0, t1), v_avg = model.generate_mean_flow_target(actions)
            
            # 计算损失
            optimizer.zero_grad()
            loss = model.loss(xt0, t0, xt1, t1, cond, v_avg)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_losses.append(loss.item())
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.6f}")
        
        # 计算平均损失
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        # 更新学习率
        scheduler.step()
        
        # 记录到WandB
        if wandb_log:
            wandb.log({
                'meanflow/train_loss': avg_loss,
                'meanflow/learning_rate': optimizer.param_groups[0]['lr'],
                'meanflow/epoch': epoch + 1,
            })
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{epochs} completed in {time.time() - start_time:.2f}s | "
              f"Avg Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 定期评估
        if eval_env is not None and (epoch + 1) % eval_freq == 0:
            eval_reward = evaluate_mean_flow(model, eval_env, device, inference_steps=inference_steps)
            eval_rewards.append(eval_reward)
            
            print(f"Evaluation at epoch {epoch+1}: Reward = {eval_reward:.2f}")
            
            if wandb_log:
                wandb.log({
                    'meanflow/eval_reward': eval_reward,
                    'meanflow/epoch': epoch + 1,
                })
            
            # 保存最佳模型（基于奖励）
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_epoch = epoch + 1
                patience_counter = 0
                
                # 保存模型
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'reward': eval_reward,
                }, os.path.join(output_dir, 'meanflow_best_model.pt'))
                
                print(f"New best model saved at epoch {epoch+1} with reward {best_reward:.2f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} evaluations")
                
                # 早停
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            # 如果没有评估环境，则基于损失保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                # 保存模型
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(output_dir, 'meanflow_best_model.pt'))
                
                print(f"New best model saved at epoch {epoch+1} with loss {best_loss:.6f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")
                
                # 早停
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    # 保存最终模型
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
    }, os.path.join(output_dir, 'meanflow_final_model.pt'))
    
    print(f"Training completed. Best model at epoch {best_epoch}")
    
    return {
        'train_losses': train_losses,
        'eval_rewards': eval_rewards,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'best_reward': best_reward
    }