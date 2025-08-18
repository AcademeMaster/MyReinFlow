import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import minari
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from gymnasium import spaces


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################################
######## 配置类 ########################################################
########################################################################

class Config:
    """配置参数容器"""
    def __init__(self):
        # 训练参数
        self.num_epochs = 100
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.eval_interval = 10
        self.checkpoint_dir = "./checkpoint_t"
        
        # 环境参数
        self.dataset_name = "mujoco/pusher/expert-v0"
        
        # 序列参数
        self.obs_horizon = 1
        self.pred_horizon = 16
        self.action_horizon = 8
        self.inference_steps = 10
        
        # 模型参数
        self.time_dim = 32
        self.hidden_dim = 256
        self.sigma = 0.0
        
        # 归一化
        self.normalize_data = True
        
        # 测试参数
        self.test_episodes = 5
        self.max_steps = 300
        
    def __repr__(self):
        """以可读方式显示所有配置"""
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())


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
            num_segments = (ep_len - config.obs_horizon) // config.action_horizon
            
            for i in range(num_segments):
                start_idx = i * config.action_horizon
                end_idx = start_idx + config.pred_horizon
                
                self.episodes.append({
                    "observations": obs[start_idx : start_idx + config.obs_horizon],
                    "actions": actions[start_idx : end_idx]
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


########################################################################
######## 训练工具类 ####################################################
########################################################################

class FlowModelTrainer:
    """流匹配模型训练器"""
    
    def __init__(self, config):
        self.config = config
        self._init_paths()
        
        # 加载数据集
        self.minari_dataset = minari.load_dataset(config.dataset_name)
        self.flow_dataset = MinariFlowDataset(self.minari_dataset, config)
        
        # 获取模型维度
        obs_dim = self.flow_dataset[0]["observations"].shape[-1]
        action_dim = self.flow_dataset[0]["actions"].shape[-1]
        
        # 创建模型
        self.model = TimeConditionedFlowModel(obs_dim, action_dim, config).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.flow_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn_fixed,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        # 学习率调度器和EMA
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.train_loader) * config.num_epochs
        )
        
        self.ema = EMAModel(self.model.parameters(), power=0.75)
        self.flow_matcher = ConditionalFlowMatcher(sigma=config.sigma)
        
        print("模型初始化完成")
        print(f"观测维度: {obs_dim}, 动作维度: {action_dim}")
        print(f"训练数据: {len(self.flow_dataset)} 个序列")
    
    def _init_paths(self):
        """初始化必要目录"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
    
    def train(self):
        """训练模型的主循环"""
        print("开始训练...")
        start_time = time.time()
        total_steps = 0
        
        for epoch in range(self.config.num_epochs):
            # 训练阶段
            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs}")
            for batch in pbar:
                # 准备批数据
                observations = batch["observations"].to(device)  # [B, obs_horizon, obs_dim]
                actions = batch["actions"].to(device)  # [B, pred_horizon, action_dim]
                
                # 流匹配
                noise = torch.randn_like(actions, device=device)
                t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(noise, actions)
                
                # 使用最新的观测作为条件
                obs_cond = observations[:, -1, :]  # [B, obs_dim]
                
                # 预测速度场
                vt = self.model(obs_cond, t, xt)  # [B, pred_horizon, action_dim]
                
                # 计算损失
                loss = F.mse_loss(vt, ut)
                epoch_loss += loss.item()
                epoch_steps += 1
                total_steps += 1
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                
                # 更新EMA
                self.ema.step(self.model.parameters())
                
                # 更新进度条
                pbar.set_postfix(loss=loss.item(), lr=self.lr_scheduler.get_last_lr()[0])
            
            # 计算平均loss
            avg_loss = epoch_loss / epoch_steps
            print(f"Epoch {epoch}: loss = {avg_loss:.6f}")
            
            # 定期评估和保存模型
            if epoch % self.config.eval_interval == 0 or epoch == self.config.num_epochs - 1:
                self._save_checkpoint(epoch, avg_loss)
                self._eval_model()
        
        # 最终保存
        self._save_checkpoint(self.config.num_epochs, avg_loss)
        print(f"训练完成，耗时: {time.time()-start_time:.2f}秒")
    
    def _save_checkpoint(self, epoch, loss):
        """保存模型检查点"""
        # 应用EMA权重
        self.ema.copy_to(self.model.parameters())
        
        # 创建检查点
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "stats": self.flow_dataset.stats,
            "config": self.config
        }
        
        # 保存文件
        filename = f"flow_ema_{epoch:04d}.pth"
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"模型已保存: {filepath}")
    
    def _eval_model(self):
        """在训练过程中进行简单评估"""
        # 暂时使用训练数据的前几个样本
        self.model.eval()
        with torch.no_grad():
            sample = next(iter(self.train_loader))
            observations = sample["observations"].to(device)[:1]  # [1, obs_horizon, obs_dim]
            actions = sample["actions"].to(device)[:1]  # [1, pred_horizon, action_dim]
            
            # 生成预测
            obs_cond = observations[:, -1, :]  # [1, obs_dim]
            noise = torch.randn(1, self.config.pred_horizon, self.config.action_dim).to(device)
            
            # 迭代求解
            for step in range(self.config.inference_steps):
                t_val = torch.tensor([step / self.config.inference_steps]).to(device)
                noise = noise + (1.0 / self.config.inference_steps) * self.model(obs_cond, t_val, noise)
            
            # 计算误差
            pred_actions = noise
            mse = F.mse_loss(pred_actions, actions)
            print(f"评估误差: {mse.item():.6f}")


########################################################################
######## 测试工具类 ####################################################
########################################################################

class FlowModelTester:
    """流匹配模型测试器"""
    
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        
        # 加载数据集以恢复环境
        minari_dataset = minari.load_dataset(config.dataset_name)
        self.env = minari_dataset.recover_environment()
        self.eval_env = minari_dataset.recover_environment(eval_env=True)
        
        # 获取环境信息
        obs_dim = np.prod(self.env.observation_space.shape)
        action_dim = np.prod(self.env.action_space.shape)
        
        # 创建模型并加载权重
        self.model = TimeConditionedFlowModel(obs_dim, action_dim, config).to(device)
        
        if checkpoint_path is None:
            # 如果没有指定检查点，尝试加载最新
            checkpoints = [f for f in os.listdir(config.checkpoint_dir) if f.startswith("flow_ema_")]
            if not checkpoints:
                raise FileNotFoundError("未找到检查点文件")
            checkpoints.sort(reverse=True)
            checkpoint_path = os.path.join(config.checkpoint_dir, checkpoints[0])
        
        print(f"加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.stats = checkpoint.get("stats", None)
        self.model.eval()
    
    def test(self):
        """测试模型在环境中的表现"""
        print("开始测试...")
        total_rewards = []
        
        for episode in range(self.config.test_episodes):
            start_time = time.time()
            obs, _ = self.eval_env.reset()
            obs_history = [obs] * self.config.obs_horizon
            episode_reward = 0
            step_count = 0
            done = False
            
            pbar = tqdm(total=self.config.max_steps, desc=f"Episode {episode}")
            
            while not done and step_count < self.config.max_steps:
                # 准备观测数据
                obs_arr = np.array(obs_history)
                if self.stats:
                    obs_arr = self._normalize(obs_arr, self.stats["observations"])
                
                # 转换为张量
                obs_tensor = torch.as_tensor(obs_arr[-1], device=device).unsqueeze(0)  # [1, obs_dim]
                
                # 迭代求解器
                noise = torch.randn(1, self.config.pred_horizon, self.config.action_dim).to(device)
                
                for step in range(self.config.inference_steps):
                    t_val = torch.tensor([step / self.config.inference_steps]).to(device)
                    update = self.model(obs_tensor, t_val, noise)
                    noise = noise + (1.0 / self.config.inference_steps) * update
                
                # 转换为numpy数组
                action_seq = noise.squeeze(0).detach().cpu().numpy()
                
                # 反归一化
                if self.stats:
                    action_seq = self.flow_dataset.denormalize(action_seq, self.stats["actions"])
                
                # 执行动作序列
                for i in range(min(len(action_seq), self.config.action_horizon)):
                    if done:
                        break
                    
                    action = action_seq[i]
                    next_obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
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


########################################################################
######## 主程序 ########################################################
########################################################################

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于流匹配的机器人操作训练")
    parser.add_argument("mode", choices=["train", "test"], help="运行模式: train或test")
    parser.add_argument("--dataset", default="mujoco/pusher/expert-v0", help="Minari数据集名称")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批量大小")
    parser.add_argument("--checkpoint", help="测试时使用的模型路径")
    parser.add_argument("--test-episodes", type=int, default=5, help="测试轮数")
    args = parser.parse_args()
    
    # 初始化配置
    config = Config()
    config.dataset_name = args.dataset
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.test_episodes = args.test_episodes
    
    print("="*50)
    print("配置参数:")
    print(config)
    print("="*50)
    
    if args.mode == "train":
        trainer = FlowModelTrainer(config)
        trainer.train()
    
    elif args.mode == "test":
        tester = FlowModelTester(config, args.checkpoint)
        tester.test()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()