import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from tqdm import tqdm
import minari
# 创建模型代理

from meanflow_ql import ConservativeMeanFQLModel, Config as MeanFlowQLConfig

class FlowModelTrainer:
    """流匹配模型训练器"""
    
    def __init__(self, config: "Config"):
        from config import Config  # 避免循环导入
        self.config: Config = config
        self._init_paths()
        
        # 初始化Accelerator，支持多GPU训练和混合精度
        self.accelerator = Accelerator(
            mixed_precision=getattr(config, 'mixed_precision', 'no'),
            gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 1)
        )
        
        # 加载数据集
        self.minari_dataset = minari.load_dataset(config.dataset_name)
        from dataset import MinariFlowDataset
        self.flow_dataset = MinariFlowDataset(self.minari_dataset, config)
        
        # 获取模型维度
        obs_dim = self.flow_dataset[0]["observations"].shape[-1]
        action_dim = self.flow_dataset[0]["actions"].shape[-1]
        self.config.action_dim = action_dim  # 添加到配置中以便测试时使用
        

        # 创建离线强化学习模型配置
        meanflow_ql_config = MeanFlowQLConfig(
            hidden_dim=config.hidden_dim,
            time_dim=config.time_dim,
            pred_horizon=config.pred_horizon,
            learning_rate=config.learning_rate,
            grad_clip_value=config.grad_clip_value,
            cql_alpha=config.cql_alpha,
            cql_temp=config.cql_temp,
            tau=config.tau,
            gamma=config.gamma,
            inference_steps=config.inference_steps,
            normalize_q_loss=config.normalize_q_loss,
            device=config.device
        )
        
        self.agent = ConservativeMeanFQLModel(obs_dim, action_dim, meanflow_ql_config)
        
        # 创建数据加载器
        from dataset import collate_fn_fixed
        self.train_loader = DataLoader(
            self.flow_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn_fixed,
            num_workers=0,  # 设置为0以避免Windows上的多进程问题
            pin_memory=True,
            persistent_workers=False
        )
        
        # 学习率调度器
        self.actor_lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.agent.actor_optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.train_loader) * config.num_epochs
        )
        
        self.critic_lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.agent.critic_optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.train_loader) * config.num_epochs
        )
        
        # 准备Accelerator
        prepared_components = self.accelerator.prepare(
            self.agent.actor.model, 
            self.agent.actor_optimizer,
            self.agent.critic,
            self.agent.critic_optimizer,
            self.train_loader, 
            self.actor_lr_scheduler,
            self.critic_lr_scheduler
        )
        
        self.agent.actor.model = prepared_components[0]
        self.agent.actor_optimizer = prepared_components[1]
        self.agent.critic = prepared_components[2]
        self.agent.critic_optimizer = prepared_components[3]
        self.train_loader = prepared_components[4]
        self.actor_lr_scheduler = prepared_components[5]
        self.critic_lr_scheduler = prepared_components[6]

        # EMA模型也需要准备
        self.ema = EMAModel(self.agent.actor.model.parameters(), power=0.75)
        
        print("模型初始化完成")
        print(f"观测维度: {obs_dim}, 动作维度: {action_dim}")
        print(f"训练数据: {len(self.flow_dataset)} 个序列")
    
    def _init_paths(self):
        """初始化必要目录"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        print(f"检查点将保存到: {self.config.checkpoint_dir}")
    
    def train(self) -> None:
        """训练模型的主循环"""
        print("开始训练...")
        start_time = time.time()
        total_steps = 0
        avg_loss = 0.0  # 初始化默认值
        
        for epoch in range(self.config.num_epochs):
            # 训练阶段
            self.agent.actor.model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs}")
            for batch in pbar:
                # 使用代理计算损失
                loss = self.agent.actor(batch)
                epoch_loss += loss.item()
                epoch_steps += 1
                total_steps += 1
                
                # 反向传播
                self.agent.actor_optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.agent.actor.model.parameters(), self.config.grad_clip_value)
                self.agent.actor_optimizer.step()
                self.actor_lr_scheduler.step()
                
                # 更新EMA
                self.ema.step(self.agent.actor.model.parameters())
                
                # 更新进度条
                pbar.set_postfix(loss=loss.item(), 
                                actor_lr=self.actor_lr_scheduler.get_last_lr()[0])
            
            # 计算平均loss
            if epoch_steps > 0:
                avg_loss = epoch_loss / epoch_steps
            print(f"Epoch {epoch}: loss = {avg_loss:.6f}")
            
            # 定期评估和保存模型
            if epoch % self.config.eval_interval == 0 or epoch == self.config.num_epochs - 1:
                self._save_checkpoint(epoch, avg_loss)
                self._eval_model()
        
        # 最终保存（使用最后一个epoch的平均损失）
        self._save_checkpoint(self.config.num_epochs, avg_loss)
        print(f"训练完成，耗时: {time.time()-start_time:.2f}秒")
    
    def _save_checkpoint(self, epoch, loss):
        """保存模型检查点"""
        # 应用EMA权重
        self.ema.copy_to(self.agent.actor.model.parameters())
        
        # 获取原始模型用于保存
        unwrapped_model = self.accelerator.unwrap_model(self.agent.actor.model)
        
        # 创建检查点
        checkpoint = {
            "model_state": unwrapped_model.state_dict(),
            "optimizer_state": self.agent.actor_optimizer.state_dict(),
            "ema_state": self.ema.state_dict(),  # 保存EMA状态
            "epoch": epoch,
            "loss": loss,
            "stats": self.flow_dataset.stats,
            "config": self.config.__dict__  # 保存配置字典以确保兼容性
        }
        
        # 保存文件
        filename = f"flow_ema_{epoch:04d}.pth"
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"模型已保存: {filepath}")
    
    def _eval_model(self) -> None:
        """在训练过程中进行简单评估"""
        # 使用训练数据的前几个样本进行评估
        self.agent.actor.model.eval()
        with torch.no_grad():
            sample = next(iter(self.train_loader))
            # 使用代理的predict_action_chunk方法进行预测
            pred_actions = self.agent.actor.predict_action_chunk(sample)
            # 获取真实动作并保持完整维度
            actions = sample["actions"]
            
            # 确保在相同的设备上进行评估
            device = self.accelerator.device
            actions = actions.to(device)
            
            # 仅评估一个批次样本
            batch_size = actions.shape[0]
            if batch_size > 1:
                # 如果批次大小大于1，只取第一个样本进行评估
                pred_actions = pred_actions[:1]
                actions = actions[:1]
            
            # 计算MSE损失
            mse = F.mse_loss(pred_actions, actions)
            print(f"评估误差: {mse.item():.6f}")
        # 恢复训练模式
        self.agent.actor.model.train()


class OfflineFlowRLTrainer:
    """离线强化学习训练器"""
    
    def __init__(self, config: "Config"):
        from config import Config  # 避免循环导入
        self.config: Config = config
        self._init_paths()
        
        # 初始化Accelerator，支持多GPU训练和混合精度
        self.accelerator = Accelerator(
            mixed_precision=getattr(config, 'mixed_precision', 'no'),
            gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 1)
        )
        
        # 加载数据集
        self.minari_dataset = minari.load_dataset(config.dataset_name)
        from dataset import MinariFlowDataset
        self.flow_dataset = MinariFlowDataset(self.minari_dataset, config)
        
        # 获取模型维度
        obs_dim = self.flow_dataset[0]["observations"].shape[-1]
        action_dim = self.flow_dataset[0]["actions"].shape[-1]
        self.config.action_dim = action_dim  # 添加到配置中以便测试时使用
        
        # 创建离线强化学习模型
        meanflow_ql_config = MeanFlowQLConfig(
            hidden_dim=config.hidden_dim,
            time_dim=config.time_dim,
            pred_horizon=config.pred_horizon,
            learning_rate=config.learning_rate,
            grad_clip_value=config.grad_clip_value,
            cql_alpha=config.cql_alpha,
            cql_temp=config.cql_temp,
            tau=config.tau,
            gamma=config.gamma,
            inference_steps=config.inference_steps,
            normalize_q_loss=config.normalize_q_loss,
            device=config.device
        )
        
        self.model = ConservativeMeanFQLModel(obs_dim, action_dim, meanflow_ql_config)
        
        # 创建数据加载器
        from dataset import collate_fn_fixed
        self.train_loader = DataLoader(
            self.flow_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn_fixed,
            num_workers=0,  # 设置为0以避免Windows上的多进程问题
            pin_memory=True,
            persistent_workers=False
        )
        
        # 学习率调度器
        self.critic_lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.model.critic_optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.train_loader) * config.num_epochs
        )
        
        self.actor_lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.model.actor_optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.train_loader) * config.num_epochs
        )
        
        # 准备Accelerator
        prepared_components = self.accelerator.prepare(
            self.model, 
            self.model.critic_optimizer,
            self.model.actor_optimizer,
            self.train_loader, 
            self.critic_lr_scheduler,
            self.actor_lr_scheduler
        )
        
        self.model = prepared_components[0]
        self.model.critic_optimizer = prepared_components[1]
        self.model.actor_optimizer = prepared_components[2]
        self.train_loader = prepared_components[3]
        self.critic_lr_scheduler = prepared_components[4]
        self.actor_lr_scheduler = prepared_components[5]
        
        print("离线强化学习模型初始化完成")
        print(f"观测维度: {obs_dim}, 动作维度: {action_dim}")
        print(f"训练数据: {len(self.flow_dataset)} 个序列")
    
    def _init_paths(self):
        """初始化必要目录"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
    
    def train(self) -> None:
        """训练离线强化学习模型的主循环"""
        print("开始离线强化学习训练...")
        start_time = time.time()
        total_steps = 0
        
        for epoch in range(self.config.num_epochs):
            # 训练阶段
            self.model.train()
            epoch_critic_loss = 0.0
            epoch_actor_loss = 0.0
            epoch_steps = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs}")
            for batch in pbar:
                obs = batch["observations"]
                actions = batch["actions"]
                
                # 确保下一个观测值存在，如果不存在则使用当前观测值作为近似
                next_obs = batch.get("next_observations", obs)
                
                # 创建奖励和终止信号（这里使用启发式方法，实际应用中应该从数据集中获取）
                batch_size = obs.shape[0]
                rewards = torch.mean(actions ** 2, dim=[1, 2]).unsqueeze(1).repeat(1, actions.shape[1]) * -0.5
                terminated = torch.zeros(batch_size, 1)
                
                # 更新critic
                critic_loss, critic_info = self.model.loss_critic(
                    obs, actions, next_obs, rewards, terminated, self.config.gamma
                )
                
                self.model.critic_optimizer.zero_grad()
                self.accelerator.backward(critic_loss)
                self.accelerator.clip_grad_norm_(self.model.critic.parameters(), self.config.grad_clip_value)
                self.model.critic_optimizer.step()
                self.critic_lr_scheduler.step()
                
                # 更新actor (较少频率)
                actor_loss = None
                actor_info = {}
                if epoch_steps % 2 == 0:  # 每2个step更新一次actor
                    actor_loss, actor_info = self.model.loss_actor(obs, actions, alpha=1.0)
                    
                    self.model.actor_optimizer.zero_grad()
                    self.accelerator.backward(actor_loss)
                    self.accelerator.clip_grad_norm_(self.model.actor.parameters(), self.config.grad_clip_value)
                    self.model.actor_optimizer.step()
                    self.actor_lr_scheduler.step()
                
                # 更新目标网络
                self.model.update_target_critic(self.config.tau)
                
                epoch_critic_loss += critic_info['total_critic_loss']
                if actor_loss is not None:
                    epoch_actor_loss += actor_info['loss_actor']
                epoch_steps += 1
                total_steps += 1
                
                # 更新进度条
                pbar_dict = {
                    'critic_loss': critic_info['total_critic_loss'],
                    'q1_mean': critic_info['q1_mean'],
                    'q2_mean': critic_info['q2_mean']
                }
                if actor_loss is not None:
                    pbar_dict['actor_loss'] = actor_info['loss_actor']
                pbar.set_postfix(**pbar_dict)
            
            # 计算平均loss
            avg_critic_loss = epoch_critic_loss / epoch_steps if epoch_steps > 0 else 0
            avg_actor_loss = epoch_actor_loss / max(1, epoch_steps // 2) if epoch_steps > 0 else 0
            
            print(f"Epoch {epoch}: critic_loss = {avg_critic_loss:.6f}, actor_loss = {avg_actor_loss:.6f}")
            
            # 定期保存模型
            if epoch % self.config.eval_interval == 0 or epoch == self.config.num_epochs - 1:
                self._save_checkpoint(epoch, avg_critic_loss, avg_actor_loss)
        
        # 最终保存
        self._save_checkpoint(self.config.num_epochs, avg_critic_loss, avg_actor_loss)
        print(f"离线强化学习训练完成，耗时: {time.time()-start_time:.2f}秒")
    
    def _save_checkpoint(self, epoch, critic_loss, actor_loss):
        """保存离线强化学习模型检查点"""
        # 获取原始模型用于保存
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # 创建检查点
        checkpoint = {
            "model_state": unwrapped_model.state_dict(),
            "actor_optimizer_state": self.model.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.model.critic_optimizer.state_dict(),
            "epoch": epoch,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "stats": self.flow_dataset.stats,
            "config": self.config.__dict__  # 保存配置字典以确保兼容性
        }
        
        # 保存文件
        filename = f"offline_flow_rl_{epoch:04d}.pth"
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"离线强化学习模型已保存: {filepath}")