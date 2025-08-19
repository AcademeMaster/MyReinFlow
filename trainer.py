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
from normalflow import FlowPolicyAgent
from meanflow import MeanFlowPolicyAgent
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
        

        self.agent = FlowPolicyAgent(obs_dim, action_dim, config)
        
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
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.agent.optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.train_loader) * config.num_epochs
        )
        
        # 准备Accelerator
        (self.agent.model, self.agent.optimizer, self.train_loader, self.lr_scheduler) = self.accelerator.prepare(
            self.agent.model, self.agent.optimizer, self.train_loader, self.lr_scheduler
        )
        
        # EMA模型也需要准备
        self.ema = EMAModel(self.agent.model.parameters(), power=0.75)
        
        print("模型初始化完成")
        print(f"观测维度: {obs_dim}, 动作维度: {action_dim}")
        print(f"训练数据: {len(self.flow_dataset)} 个序列")
    
    def _init_paths(self):
        """初始化必要目录"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
    
    def train(self) -> None:
        """训练模型的主循环"""
        print("开始训练...")
        start_time = time.time()
        total_steps = 0
        avg_loss = 0.0  # 初始化默认值
        
        for epoch in range(self.config.num_epochs):
            # 训练阶段
            self.agent.model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs}")
            for batch in pbar:
                # 使用代理计算损失
                loss = self.agent.forward(batch)
                epoch_loss += loss.item()
                epoch_steps += 1
                total_steps += 1
                
                # 反向传播
                self.agent.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.agent.model.parameters(), 1.0)
                self.agent.optimizer.step()
                self.lr_scheduler.step()
                
                # 更新EMA
                self.ema.step(self.agent.model.parameters())
                
                # 更新进度条
                pbar.set_postfix(loss=loss.item(), lr=self.lr_scheduler.get_last_lr()[0])
            
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
        self.ema.copy_to(self.agent.model.parameters())
        
        # 获取原始模型用于保存
        unwrapped_model = self.accelerator.unwrap_model(self.agent.model)
        
        # 创建检查点
        checkpoint = {
            "model_state": unwrapped_model.state_dict(),
            "optimizer_state": self.agent.optimizer.state_dict(),
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
        self.agent.model.eval()
        with torch.no_grad():
            sample = next(iter(self.train_loader))
            # 使用代理的predict_action_chunk方法进行预测
            pred_actions = self.agent.predict_action_chunk(sample)
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