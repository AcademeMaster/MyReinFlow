import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import minari
import gymnasium as gym
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from gymnasium import spaces
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from accelerate import Accelerator


########################################################################
######## 配置类 ########################################################
########################################################################

@dataclass
class Config:
    """配置参数容器"""
    # 训练参数
    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    eval_interval: int = 10
    checkpoint_dir: str = "./checkpoint_t"
    
    # 环境参数
    dataset_name: str = "mujoco/pusher/expert-v0"
    
    # 序列参数
    obs_horizon: int = 1
    pred_horizon: int = 16
    action_horizon: int = 8
    inference_steps: int = 10
    
    # 模型参数
    time_dim: int = 32
    hidden_dim: int = 256
    sigma: float = 0.0
    
    # 归一化
    normalize_data: bool = True
    
    # 测试参数
    test_episodes: int = 5
    max_steps: int = 300
    
    # 动作维度（在初始化时设置）
    action_dim: int = 0
    
    # Accelerator相关参数
    mixed_precision: str = "no"
    gradient_accumulation_steps: int = 1
    
    def __post_init__(self):
        """初始化后处理"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def __repr__(self):
        """以可读方式显示所有配置"""
        return "\n".join(f"{k}: {v}" for k, v in asdict(self).items())


########################################################################
######## 数据集类 ######################################################
########################################################################

class MinariFlowDataset(Dataset):
    """处理Minari数据集的自定义Dataset类"""
    
    def __init__(self, dataset, config):
        """
        参数:
            dataset: Minari数据集
            config: Config对象
        """
        self.config = config
        self.episodes = []
        self.stats = self._compute_stats(dataset)
        
        # 处理每个episode
        for episode in tqdm(dataset, desc="处理数据集"):
            ep_len = len(episode.actions)
            
            # 跳过太短的序列
            if ep_len < config.pred_horizon + config.obs_horizon:
                continue
                
            # 准备数据
            obs = self._normalize(episode.observations, self.stats["observations"])
            actions = self._normalize(episode.actions, self.stats["actions"])
            
            # 提取子序列
            num_segments = (ep_len - config.obs_horizon - config.pred_horizon + 1)
            
            for i in range(num_segments):
                start_obs = i
                end_obs = start_obs + config.obs_horizon
                start_act = end_obs
                end_act = start_act + config.pred_horizon
                
                # 确保索引不越界
                if end_act <= len(actions):
                    self.episodes.append({
                        "observations": obs[start_obs:end_obs],
                        "actions": actions[start_act:end_act]
                    })
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        item = self.episodes[idx]
        return {
            "observations": torch.as_tensor(item["observations"], dtype=torch.float32),
            "actions": torch.as_tensor(item["actions"], dtype=torch.float32)
        }
    
    def _compute_stats(self, dataset):
        """计算整个数据集的统计量"""
        all_obs = []
        all_actions = []
        
        for episode in dataset:
            all_obs.append(episode.observations)
            all_actions.append(episode.actions)
        
        all_obs = np.concatenate(all_obs, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        
        return {
            "observations": {
                "min": all_obs.min(axis=0),
                "max": all_obs.max(axis=0),
                "mean": all_obs.mean(axis=0),
                "std": all_obs.std(axis=0)
            },
            "actions": {
                "min": all_actions.min(axis=0),
                "max": all_actions.max(axis=0),
                "mean": all_actions.mean(axis=0),
                "std": all_actions.std(axis=0)
            }
        }
    
    def _normalize(self, data, stats):
        """数据归一化"""
        if self.config.normalize_data:
            return (data - stats["mean"]) / (stats["std"] + 1e-8)
        return data
    
    def denormalize(self, data, stats):
        """数据反归一化"""
        if self.config.normalize_data:
            return data * stats["std"] + stats["mean"]
        return data


def collate_fn_fixed(batch):
    """固定长度序列的批处理函数"""
    return {
        "observations": torch.stack([item["observations"] for item in batch]),
        "actions": torch.stack([item["actions"] for item in batch])
    }


########################################################################
######## 模型类 ########################################################
########################################################################

class TimeConditionedFlowModel(nn.Module):
    """带时间条件约束的流匹配模型"""
    
    def __init__(self, obs_dim, action_dim, config):
        """
        参数:
            obs_dim: 观测维度
            action_dim: 动作维度
            config: Config对象
        """
        super().__init__()
        self.config = config
        
        # 时间特征嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.time_dim),
            nn.SiLU(),
            nn.Linear(config.time_dim, config.time_dim)
        )
        
        # 观测特征处理
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # 噪声特征处理
        self.noise_embed = nn.Sequential(
            nn.Linear(action_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # 联合处理模块
        self.joint_processor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 + config.time_dim, config.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, action_dim)
        )
    
    def forward(self, obs, t, noise):
        """
        前向传播
        参数:
            obs: 观测张量 [B, obs_dim]
            t: 时间步张量 [B]
            noise: 噪声张量 [B, pred_horizon, action_dim]
        
        返回:
            速度场 [B, pred_horizon, action_dim]
        """
        # 获取模型所在的设备
        device = next(self.parameters()).device
        
        # 将输入张量移动到模型设备
        obs = obs.to(device)
        t = t.to(device)
        noise = noise.to(device)
        
        # 嵌入时间
        t_emb = self.time_embed(t.unsqueeze(-1).float())  # [B, time_dim]
        
        # 嵌入观测
        obs_emb = self.obs_embed(obs)  # [B, hidden_dim]
        
        # 嵌入噪声
        # 对每个时间步独立处理噪声
        B, H, A = noise.shape
        noise_emb = self.noise_embed(noise.view(B * H, A))  # [B*H, hidden_dim]
        noise_emb = noise_emb.view(B, H, -1)  # [B, H, hidden_dim]
        
        # 合并特征
        obs_emb = obs_emb.unsqueeze(1).expand(-1, H, -1)  # [B, H, hidden_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, H, -1)  # [B, H, time_dim]
        
        combined = torch.cat([obs_emb, noise_emb, t_emb], dim=-1)  # [B, H, hidden_dim*2+time_dim]
        
        # 预测速度场
        velocity = self.joint_processor(combined)  # [B, H, action_dim]
        
        return velocity



class FlowPolicyAgent:
    """流策略代理"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Config):
        """
        初始化流策略代理
        
        参数:
            obs_dim: 观测维度
            action_dim: 动作维度
            config: 配置对象
        """
        self.config = config
        self.model = TimeConditionedFlowModel(obs_dim, action_dim, config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.flow_matcher = ConditionalFlowMatcher(sigma=config.sigma)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        预测动作块
        
        参数:
            batch: 包含观测数据的批次字典
            
        返回:
            预测的动作张量
        """
        self.model.eval()
        # 获取模型所在的设备
        device = next(self.model.parameters()).device
        
        observations = batch["observations"].to(device)  # [B, obs_horizon, obs_dim]
        
        # 使用最新的观测作为条件
        obs_cond = observations[:, -1, :]  # [B, obs_dim]
        
        # 迭代求解器生成动作
        noise = torch.randn(observations.size(0), self.config.pred_horizon, self.config.action_dim).to(device)
        
        for step in range(self.config.inference_steps):
            t_val = torch.full((observations.size(0),), step / self.config.inference_steps, device=device, dtype=torch.float32)
            update = self.model(obs_cond, t_val, noise)
            noise = noise + (1.0 / self.config.inference_steps) * update
            
        return noise

    @torch.no_grad()
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        # 目前实现仅返回预测的动作块的第一个动作
        action_chunk = self.predict_action_chunk(batch)
        return action_chunk[:, 0, :]  # 返回第一个动作

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播，计算流匹配损失
        
        参数:
            batch: 包含观测和动作的批次字典
            
        返回:
            计算的损失值
        """
        # 获取模型所在的设备
        device = next(self.model.parameters()).device
        
        # 将输入数据移动到模型设备
        observations = batch["observations"].to(device)  # [B, obs_horizon, obs_dim]
        actions = batch["actions"].to(device)  # [B, pred_horizon, action_dim]
        
        # 流匹配 - 使用设备一致的噪声生成
        noise = torch.randn_like(actions, device=device)
        flow_data = self.flow_matcher.sample_location_and_conditional_flow(noise, actions)
        t, xt, ut = flow_data[:3]

        
        # 使用最新的观测作为条件
        obs_cond = observations[:, -1, :]  # [B, obs_dim]
        
        # 预测速度场
        vt = self.model(obs_cond, t, xt)  # [B, pred_horizon, action_dim]
        
        # 计算损失
        loss = F.mse_loss(vt, ut)
        return loss

########################################################################
######## 训练工具类 ####################################################
########################################################################

class FlowModelTrainer:
    """流匹配模型训练器"""
    
    def __init__(self, config: Config):
        self.config: Config = config
        self._init_paths()
        
        # 初始化Accelerator，支持多GPU训练和混合精度
        self.accelerator = Accelerator(
            mixed_precision=getattr(config, 'mixed_precision', 'no'),
            gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 1)
        )
        
        # 加载数据集
        self.minari_dataset = minari.load_dataset(config.dataset_name)
        self.flow_dataset = MinariFlowDataset(self.minari_dataset, config)
        
        # 获取模型维度
        obs_dim = self.flow_dataset[0]["observations"].shape[-1]
        action_dim = self.flow_dataset[0]["actions"].shape[-1]
        self.config.action_dim = action_dim  # 添加到配置中以便测试时使用
        
        # 创建模型代理
        self.agent = FlowPolicyAgent(obs_dim, action_dim, config)
        
        # 创建数据加载器
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


########################################################################
######## 测试工具类 ####################################################
########################################################################

class FlowModelTester:
    """流匹配模型测试器"""
    
    def __init__(self, config: Config, checkpoint_path=None, render_mode=None):
        self.config = config
        
        # 确定检查点路径
        if checkpoint_path is None:
            # 如果没有指定检查点，尝试加载最新
            checkpoints = [f for f in os.listdir(config.checkpoint_dir) if f.startswith("flow_ema_")]
            if not checkpoints:
                raise FileNotFoundError("未找到检查点文件")
            checkpoints.sort(reverse=True)
            checkpoint_path = os.path.join(config.checkpoint_dir, checkpoints[0])
        
        print(f"加载模型权重: {checkpoint_path}")
        # 修复加载问题，添加weights_only=False参数
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 从检查点中获取模型维度信息
        obs_dim_from_ckpt = checkpoint["model_state"]["obs_embed.0.weight"].shape[1]
        action_dim_from_ckpt = checkpoint["model_state"]["noise_embed.0.weight"].shape[1]
        
        print(f"从检查点加载的维度信息: obs_dim={obs_dim_from_ckpt}, action_dim={action_dim_from_ckpt}")
        
        # 更新配置中的action_dim
        self.config.action_dim = action_dim_from_ckpt
        self.obs_dim_from_ckpt = obs_dim_from_ckpt  # 保存检查点中的观测维度
        self.action_dim_from_ckpt = action_dim_from_ckpt  # 保存检查点中的动作维度
        
        # 加载数据集以恢复环境
        minari_dataset = minari.load_dataset(config.dataset_name)
        
        # 尝试使用指定的渲染模式恢复环境
        self.eval_env = None
        if render_mode is not None and render_mode != "none":
            try:
                self.eval_env = minari_dataset.recover_environment(eval_env=True, render_mode=render_mode)
            except TypeError:
                # 兼容旧版本不支持 render_mode 参数
                try:
                    self.eval_env = minari_dataset.recover_environment(eval_env=True)
                except Exception:
                    self.eval_env = minari_dataset.recover_environment()
                print("警告: 当前Minari版本不支持render_mode参数")
            except Exception as e:
                print(f"使用render_mode恢复环境时出错: {e}")

        # 如果还没有成功创建评估环境，则使用默认方式
        if self.eval_env is None:
            try:
                self.eval_env = minari_dataset.recover_environment(eval_env=True)
            except Exception:
                # 最后的备选方案
                base_env = minari_dataset.recover_environment()
                env_spec = getattr(base_env, 'spec', None)
                env_id = getattr(env_spec, 'id', None) if env_spec else None
                if env_id:
                    render_mode_valid = render_mode if render_mode != "none" else None
                    self.eval_env = gym.make(env_id, render_mode=render_mode_valid)
                else:
                    self.eval_env = base_env
        
        # 获取环境信息并处理 None 的情况
        obs_space = self.eval_env.observation_space
        act_space = self.eval_env.action_space

        obs_shape = obs_space.shape if isinstance(obs_space, spaces.Box) else (1,)
        action_shape = act_space.shape if isinstance(act_space, spaces.Box) else (1,)

        obs_dim_env = int(np.prod(obs_shape)) if obs_shape is not None else 1
        action_dim_env = int(np.prod(action_shape)) if action_shape is not None else 1
        
        print(f"环境中的维度信息: obs_dim={obs_dim_env}, action_dim={action_dim_env}")
        
        # 检查检查点中的配置格式
        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            # 从检查点恢复配置
            saved_config = checkpoint["config"]
            for key, value in saved_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # 创建模型并加载权重（使用检查点中的维度信息）
        self.model = TimeConditionedFlowModel(obs_dim_from_ckpt, action_dim_from_ckpt, config).to(device)
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.stats = checkpoint.get("stats", None)
        self.model.eval()
        
        # 保存环境的动作维度
        self.action_dim_env = action_dim_env
        
        # 如果环境维度与检查点维度不一致，禁用归一化
        if obs_dim_env != obs_dim_from_ckpt or action_dim_env != action_dim_from_ckpt:
            print("警告: 环境维度与检查点维度不一致，禁用数据归一化")
            self.config.normalize_data = False
            # 创建一个索引来选择与模型匹配的观测维度
            if obs_dim_env > obs_dim_from_ckpt:
                self.obs_indices = torch.arange(obs_dim_from_ckpt)
                print(f"将从环境观测中选择前{obs_dim_from_ckpt}个维度")
            else:
                self.obs_indices = None
                print("环境观测维度小于模型期望维度，这可能会导致错误")
                
            # 对于动作维度，如果环境需要更多维度，则填充0
            if action_dim_env > action_dim_from_ckpt:
                self.action_padding = action_dim_env - action_dim_from_ckpt
                print(f"将为动作添加{self.action_padding}个零填充维度")
            elif action_dim_env < action_dim_from_ckpt:
                self.action_truncate = action_dim_from_ckpt - action_dim_env
                print(f"将从动作中截取前{action_dim_env}个维度")
            else:
                self.action_padding = 0
                self.action_truncate = 0
        else:
            self.obs_indices = None
            self.action_padding = 0
            self.action_truncate = 0
    
    def test(self):
        """测试模型在环境中的表现"""
        print("开始测试...")
        total_rewards = []
        
        # 定义设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for episode in range(self.config.test_episodes):
            start_time = time.time()
            obs, _ = self.eval_env.reset()
            obs_history = [obs] * self.config.obs_horizon
            episode_reward = 0.0  # 明确指定为浮点数
            step_count = 0
            done = False
            
            pbar = tqdm(total=self.config.max_steps, desc=f"Episode {episode}")
            
            while not done and step_count < self.config.max_steps:
                # 准备观测数据
                obs_arr = np.array(obs_history)
                if self.stats and self.config.normalize_data:
                    obs_arr = self._normalize(obs_arr, self.stats["observations"])
                
                # 转换为张量
                obs_tensor = torch.as_tensor(obs_arr[-1], device=device, dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
                
                # 如果需要，调整观测数据维度以匹配模型期望的维度
                if self.obs_indices is not None:
                    obs_tensor = torch.index_select(obs_tensor, 1, self.obs_indices.to(device))
                
                # 迭代求解器
                noise = torch.randn(1, self.config.pred_horizon, self.config.action_dim).to(device)
                
                for step in range(self.config.inference_steps):
                    t_val = torch.full((1,), step / self.config.inference_steps, device=device, dtype=torch.float32)
                    update = self.model(obs_tensor, t_val, noise)
                    noise = noise + (1.0 / self.config.inference_steps) * update
                
                # 转换为numpy数组
                action_seq = noise.squeeze(0).detach().cpu().numpy()
                
                # 反归一化
                if self.stats and self.config.normalize_data:
                    action_seq = self._denormalize(action_seq, self.stats["actions"])
                
                # 执行动作序列
                for i in range(min(len(action_seq), self.config.action_horizon)):
                    if done:
                        break
                    
                    action = action_seq[i]
                    
                    # 如果需要，调整动作维度以匹配环境期望的维度
                    if self.action_padding > 0:
                        # 填充0以增加动作维度
                        padded_action = np.pad(action, (0, self.action_padding), mode='constant')
                        action = padded_action
                    elif self.action_truncate > 0:
                        # 截取动作以减少维度
                        action = action[:self.action_dim_env]
                    
                    next_obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += float(reward)  # 确保类型转换
                    step_count += 1
                    
                    # 更新观测历史
                    obs_history.pop(0)
                    obs_history.append(next_obs)
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
            
            pbar.close()
            duration = time.time() - start_time
            fps = step_count / duration
            total_rewards.append(episode_reward)
            print(f"Episode {episode}: reward = {episode_reward:.2f}, steps = {step_count}, "
                  f"duration = {duration:.2f}s, FPS = {fps:.1f}")
        
        # 显示总结统计
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"\n测试完成，平均奖励: {avg_reward:.2f} (±{np.std(total_rewards):.2f})")
    
    def _normalize(self, data, stats):
        """数据归一化"""
        return (data - stats["mean"]) / (stats["std"] + 1e-8) if self.config.normalize_data else data
        
    def _denormalize(self, data, stats):
        """数据反归一化"""
        if self.config.normalize_data:
            return data * stats["std"] + stats["mean"]
        return data


########################################################################
######## 主程序 ########################################################
########################################################################

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于流匹配的机器人操作训练")
    parser.add_argument("mode", choices=["train", "test"], help="运行模式: train或test")
    parser.add_argument("--dataset", default="mujoco/ant/expert-v0", help="Minari数据集名称")
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
    parser.add_argument("--mixed-precision", type=str, choices=["no", "fp16", "bf16"], default="no", 
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
        tester.test()  # 添加缺失的test()方法调用


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()