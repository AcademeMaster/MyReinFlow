"""
使用 PyTorch 进行行为克隆 (Behavioral Cloning)
================================================

改进点：
- 更清晰的结构与类型标注
- 评估阶段可视化（human 渲染）与可选视频录制
- 支持动作裁剪（连续动作）与梯度裁剪
- WandB 可选
"""

import os
import sys
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from gymnasium.spaces import flatten_space, flatten
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False

import minari
@dataclass
class Config:
    """配置类，统一管理超参数"""
    # 数据集配置
    dataset_name: str = 'mujoco/pusher/expert-v0'

    # 训练配置
    batch_size: int = 256
    num_epochs: int = 50
    learning_rate: float = 3e-4
    hidden_dim: int = 256
    seed: int = 42
    # 性能与稳定性
    num_workers: int = 0
    max_grad_norm: float = 1.0
    use_amp: bool = True

    # wandb配置
    project_name: str = "behavioral-cloning"
    experiment_name: str = "pusher-expert-bc"

    # 评估配置
    eval_episodes: int = 10
    eval_freq: int = 5

    # 可视化/视频录制
    render_mode: str = 'human'  # 'none' | 'human' | 'rgb_array'
    record_video: bool = False
    video_dir: str = 'videos'

    # 其他
    save_model_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def set_seed(seed: int):
    """设置随机种子以确保实验可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class PolicyNetwork(nn.Module):
    """策略网络：用于行为克隆的神经网络模型"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        """简单的 MLP 策略网络"""
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)



def collate_fn(batch):
    """
    自定义批处理函数：处理不同长度的序列数据
    使用 pad_sequence 对不同长度的轨迹进行填充
    """
    lengths = torch.tensor([len(x.actions) for x in batch], dtype=torch.long)
    max_len = int(lengths.max().item()) if lengths.numel() else 0
    valid_mask = (
        torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    ) if max_len > 0 else torch.zeros((len(batch), 0), dtype=torch.bool)

    return {
        "id": torch.tensor([getattr(x, 'id', i) for i, x in enumerate(batch)], dtype=torch.long),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations, dtype=torch.float32) for x in batch],
            batch_first=True,
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions, dtype=torch.float32) for x in batch],  # 统一使用float32
            batch_first=True,
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards, dtype=torch.float32) for x in batch],
            batch_first=True,
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations, dtype=torch.bool) for x in batch],
            batch_first=True,
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations, dtype=torch.bool) for x in batch],
            batch_first=True,
        ),
        "valid_mask": valid_mask,
    }


class BehavioralCloning:
    """行为克隆训练器"""
    
    def __init__(self, config: Config):
        """
        初始化行为克隆训练器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载数据集
        print(f"正在加载数据集: {config.dataset_name}")
        self.dataset = minari.load_dataset(config.dataset_name)
        # 训练环境（无渲染）
        self.env = self.dataset.recover_environment()
        # 评估环境（根据配置启用渲染/视频录制）
        self.eval_env = self._make_eval_env()

        # 获取环境信息
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # 支持离散和连续动作空间
        if isinstance(self.action_space, spaces.Discrete):
            self.action_dim = int(self.action_space.n)
            self.is_discrete = True
        elif isinstance(self.action_space, spaces.Box):
            self.action_dim = int(np.prod(self.action_space.shape))
            self.is_discrete = False
        else:
            raise ValueError(f"不支持的动作空间类型: {type(self.action_space)}")

        # 统一展平观测空间，兼容 Dict 等复杂空间
        self.flat_obs_space = flatten_space(self.observation_space)
        flat_shape = getattr(self.flat_obs_space, 'shape', None)
        self.obs_dim = int(np.prod(flat_shape if flat_shape else (1,)))

        # 创建数据加载器
        loader_kwargs = dict(
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        if config.num_workers > 0:
            loader_kwargs.update(
                num_workers=config.num_workers,
                pin_memory=(self.device.type == 'cuda'),
                persistent_workers=True,
                prefetch_factor=2,
            )
        else:
            loader_kwargs.update(
                num_workers=0,
                pin_memory=(self.device.type == 'cuda'),
            )
        self.dataloader = DataLoader(self.dataset, **loader_kwargs)  # type: ignore

        # 初始化网络
        self.policy_net = PolicyNetwork(
            input_dim=self.obs_dim,
            output_dim=self.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=config.learning_rate,
        )

        # AMP 混合精度
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda' and self.config.use_amp))

        # 初始化损失函数
        self.loss_fn = nn.CrossEntropyLoss() if self.is_discrete else nn.SmoothL1Loss()

    def _flatten_obs(self, obs) -> np.ndarray:
        """将任意结构的观测展平成一维向量(np.float32)。"""
        try:
            flat = flatten(self.observation_space, obs)
        except Exception:
            flat = np.asarray(obs, dtype=np.float32)
        return np.asarray(flat, dtype=np.float32)

    # --------------------------
    # 环境构造与工具函数
    # --------------------------
    def _make_eval_env(self):
        """构建评估环境，尽可能使用数据集的环境规格，并根据配置启用渲染或视频录制。"""
        render_mode = None if self.config.render_mode == 'none' else self.config.render_mode
        env = None
        # 优先尝试从 Minari 恢复（如果支持 render_mode / eval_env 参数）
        try:
            env = self.dataset.recover_environment(eval_env=True, render_mode=render_mode)  # type: ignore[arg-type]
        except TypeError:
            # 兼容旧版本不支持 render_mode 参数
            try:
                env = self.dataset.recover_environment(eval_env=True)  # type: ignore
            except Exception:
                env = None

        # 无法恢复评估环境，退回用 gym.make 重新创建
        if env is None:
            try:
                env_spec = getattr(self.env, 'spec', None)
                env_id = getattr(env_spec, 'id', None)
                if env_id is None:
                    raise RuntimeError("无法从数据集恢复评估环境，也无法推断 env_id。")
                env = gym.make(env_id, render_mode=render_mode)
            except Exception as e:
                raise RuntimeError(f"创建评估环境失败: {e}")

        # 如果需要录制视频但环境不支持 human 渲染，则使用 rgb_array 并包裹 RecordVideo
        if self.config.record_video:
            # RecordVideo 要求 env.render_mode 为 'rgb_array' 或支持渲染帧
            if getattr(env, 'render_mode', None) != 'rgb_array':
                env.close()
                env_spec = getattr(env, 'spec', None)
                env_id = getattr(env_spec, 'id', None)
                if env_id is None:
                    raise RuntimeError("评估环境缺少 spec.id，无法启用视频录制。")
                env = gym.make(env_id, render_mode='rgb_array')
            try:
                os.makedirs(self.config.video_dir, exist_ok=True)
                env = gym.wrappers.RecordVideo(
                    env,
                    video_folder=self.config.video_dir,
                    episode_trigger=lambda i: True,
                    name_prefix="eval"
                )
            except Exception as e:
                print(f"警告: 无法启用视频录制: {e}")
        return env

    def _policy(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """给定观测，输出策略：
        - 离散：返回 logits
        - 连续：返回经 tanh 缩放到动作空间范围内的动作
        """
        logits = self.policy_net(obs_tensor)
        if self.is_discrete:
            return logits
        # 连续动作：仅当为 Box 空间时做 tanh 缩放
        if isinstance(self.action_space, spaces.Box):
            low = torch.as_tensor(self.action_space.low, device=logits.device, dtype=logits.dtype)
            high = torch.as_tensor(self.action_space.high, device=logits.device, dtype=logits.dtype)
            act = torch.tanh(logits)
            center = (high + low) / 2
            half_range = (high - low) / 2
            return center + act * half_range
        # 兜底：不缩放直接返回
        return logits

    def _postprocess_action(self, action: np.ndarray) -> np.ndarray:
        """对连续动作进行裁剪，确保位于 action_space 的界内。"""
        if not self.is_discrete and isinstance(self.action_space, spaces.Box):
            low = np.asarray(self.action_space.low, dtype=np.float32)
            high = np.asarray(self.action_space.high, dtype=np.float32)
            action = np.clip(action, low, high)
        return action
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        单个训练步骤：仅在有效时间步上计算损失，支持 AMP 与梯度裁剪。
        """
        # 移动到设备
        observations = batch['observations'].to(self.device)  # [B, T_obs, obs_dim]
        actions = batch['actions'].to(self.device)            # [B, T_act, action_dim]
        valid_mask = batch.get('valid_mask', None)
        if valid_mask is not None:
            valid_mask = valid_mask.to(self.device)  # [B, T_act]

        # 在 Minari 中，observations 通常比 actions 多 1 个时间步（最终状态）
        batch_size, obs_seq_len, obs_dim = observations.shape
        _, action_seq_len, action_dim = actions.shape

        # 对齐观测与动作序列长度
        if obs_seq_len == action_seq_len + 1:
            obs_for_pred = observations[:, :-1, :]  # [B, T_act, obs_dim]
        elif obs_seq_len == action_seq_len:
            obs_for_pred = observations
        else:
            min_len = min(obs_seq_len, action_seq_len)
            obs_for_pred = observations[:, :min_len, :]
            actions = actions[:, :min_len]
            if valid_mask is not None:
                valid_mask = valid_mask[:, :min_len]

        # 展平时间维度
        obs_flat = obs_for_pred.reshape(-1, obs_dim)        # [B*T, obs_dim]
        if self.is_discrete:
            actions_flat_all = actions.reshape(-1).long()    # [B*T]
        else:
            actions_flat_all = actions.reshape(-1, action_dim)  # [B*T, A]

        # 仅对有效步计算损失
        if valid_mask is not None:
            valid_mask_flat = valid_mask.reshape(-1)
            obs_flat = obs_flat[valid_mask_flat]
            actions_flat = actions_flat_all[valid_mask_flat]
        else:
            actions_flat = actions_flat_all

        # 前向与损失
        with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda' and self.config.use_amp)):
            action_pred = self._policy(obs_flat)
            loss = self.loss_fn(action_pred, actions_flat)

        # 反向传播
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.config.max_grad_norm)
            self.optimizer.step()

        return float(loss.item())
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估策略性能
        
        Args:
            num_episodes: 评估轮数
            
        Returns:
            评估结果字典
        """
        self.policy_net.eval()
        
        episode_returns = []
        episode_lengths = []
        
        with torch.no_grad():
            for ep_idx in range(num_episodes):
                obs, _ = self.eval_env.reset()
                episode_return = 0.0
                episode_length = 0
                done = False
                
                while not done:
                    # 将观测展平为 [1, obs_dim]
                    obs_arr = self._flatten_obs(obs).reshape(1, -1)
                    obs_tensor = torch.from_numpy(obs_arr).to(self.device)
                    
                    if self.is_discrete:
                        action_logits = self._policy(obs_tensor)
                        action = torch.argmax(action_logits, dim=-1).cpu().numpy()[0]
                    else:
                        action = self._policy(obs_tensor).detach().cpu().numpy()[0]
                        action = self._postprocess_action(action)
                    
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    episode_return += float(reward)
                    episode_length += 1
                    done = terminated or truncated

                    # 可视化（human 渲染模式下）
                    try:
                        if getattr(self.eval_env, 'render_mode', None) == 'human':
                            self.eval_env.render()
                    except Exception:
                        pass
                
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
        
        self.policy_net.train()
        
        return {
            'mean_return': float(np.mean(episode_returns)),
            'std_return': float(np.std(episode_returns)),
            'mean_length': float(np.mean(episode_lengths)),
            'std_length': float(np.std(episode_lengths))
        }
    
    def train(self):
        """主训练循环"""
        self.policy_net.train()
        
        print("开始训练...")
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            
            # 训练一个epoch
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                loss = self.train_step(batch)
                epoch_losses.append(loss)
            
            # 计算平均损失
            avg_loss = np.mean(epoch_losses)
            
            # 记录训练损失
            if WANDB_AVAILABLE and wandb is not None:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': avg_loss,
                })
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_loss:.6f}")
            
            # 定期评估
            if (epoch + 1) % self.config.eval_freq == 0:
                eval_results = self.evaluate(self.config.eval_episodes)
                
                # 记录评估结果
                if WANDB_AVAILABLE and wandb is not None:
                    wandb.log({
                        'eval/mean_return': eval_results['mean_return'],
                        'eval/std_return': eval_results['std_return'],
                        'eval/mean_length': eval_results['mean_length'],
                        'eval/std_length': eval_results['std_length'],
                    })
                
                print(f"评估结果 - 平均回报: {eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f}")
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }, path)
        print(f"模型已保存到: {path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='行为克隆训练')
    parser.add_argument('--dataset', type=str, default='mujoco/pusher/expert-v0', 
                        help='数据集名称')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='批大小')
    parser.add_argument('--lr', type=float, default=3e-4, 
                        help='学习率')
    parser.add_argument('--eval_freq', type=int, default=5, 
                        help='评估频率')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--no_wandb', action='store_true', 
                        help='禁用wandb记录')
    parser.add_argument('--render', type=str, default='human', choices=['none', 'human', 'rgb_array'],
                        help='评估时的渲染模式')
    parser.add_argument('--record_video_dir', type=str, default='',
                        help='如需录制评估视频，设置输出目录（将自动启用记录）')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config(
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_freq=args.eval_freq,
        seed=args.seed,
        render_mode=args.render,
        record_video=bool(args.record_video_dir),
        video_dir=args.record_video_dir if args.record_video_dir else 'videos'
    )
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 初始化wandb
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb and wandb is not None:
        wandb.init(
            project=config.project_name,
            name=config.experiment_name,
            config=config.to_dict()
        )
    else:
        print("跳过wandb初始化")
    
    # 创建训练器并训练
    trainer = BehavioralCloning(config)
    trainer.train()
    
    # 保存模型
    model_path = config.save_model_path or f"bc_model_{config.dataset_name.replace('/', '_')}.pth"
    trainer.save_model(model_path)
    
    # 关闭wandb
    if use_wandb and wandb is not None:
        wandb.finish()
    
    print("训练完成！")


if __name__ == "__main__":
    main()
