import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Tuple, Union
import lightning as L
from torch import Tensor

from config import Config
from meanflow_ql import MeanFQL


class LitMeanFQL(L.LightningModule):
    """基于Mean Flow的强化学习模型Lightning实现"""

    def __init__(self, obs_dim: int, action_dim: int, cfg: Config):
        """
        初始化LitMeanFQL模型

        Args:
            obs_dim: 观测维度
            action_dim: 动作维度
            cfg: 配置参数对象
        """
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.net = MeanFQL(obs_dim, action_dim, cfg)
        # 禁用自动优化，因为我们使用多个优化器和频率控制
        self.automatic_optimization = False
        # 将 step_idx 注册为 buffer，使其可以被 checkpoint 保存
        self.register_buffer("step_idx", torch.tensor(0))

    def forward(self, obs: Tensor) -> Tensor:
        """
        前向传播：使用优化的Best-of-N采样预测动作

        Args:
            obs: 观测张量，形状为 [B, obs_dim]

        Returns:
            预测的动作序列，形状为 [B, pred_horizon, action_dim]
        """
        return self.net.best_of_n_sampling(obs, self.cfg.N)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        训练步骤，分别更新critic、value function和policy

        Args:
            batch: 包含训练数据的字典
            batch_idx: 批次索引

        Returns:
            包含损失值的字典
        """
        device = self.device

        # 数据预处理：使用单个观测而不是观测序列
        observations = batch["observations"].float().to(device)  # [B, obs_dim]
        # 确保action_chunks的维度正确
        action_chunks = batch["action_chunks"].float().to(device)  # [B, H, A]
        next_observations = batch["next_observations"].float().to(device)  # [B, obs_dim]
        # 移除不必要的squeeze操作
        rewards = batch["rewards"].float().to(device)  # [B, H]
        dones = batch["dones"].float().to(device)  # [B, 1]

        # 获取优化器
        critic_optimizer, policy_optimizer = self.optimizers()
        # 更新 step_idx
        self.step_idx += 1

        # 初始化损失值和信息字典，确保在所有条件下都有定义
        critic_loss = torch.tensor(0.0, device=device)
        policy_loss = torch.tensor(0.0, device=device)
        critic_info: Dict[str, float] = {}
        policy_info: Dict[str, float] = {}

        # 根据更新频率分别更新各个网络组件
        if (int(self.step_idx) % self.cfg.q_update_period) == 0:
            # Critic网络更新步骤
            self.toggle_optimizer(critic_optimizer)
            critic_loss, critic_info = self.net.loss_critic(observations, action_chunks, next_observations, rewards,
                                                            dones)
            critic_optimizer.zero_grad()
            self.manual_backward(critic_loss)
            if self.cfg.grad_clip_value:
                self.clip_gradients(critic_optimizer, gradient_clip_val=self.cfg.grad_clip_value, gradient_clip_algorithm="norm")
            critic_optimizer.step()
            self.untoggle_optimizer(critic_optimizer)

        if (int(self.step_idx) % self.cfg.policy_update_period) == 0:
            # Policy更新步骤
            self.toggle_optimizer(policy_optimizer)
            policy_loss, policy_info = self.net.loss_policy(observations, action_chunks)
            policy_optimizer.zero_grad()
            self.manual_backward(policy_loss)
            if self.cfg.grad_clip_value:
                self.clip_gradients(policy_optimizer, gradient_clip_val=self.cfg.grad_clip_value, gradient_clip_algorithm="norm")
            policy_optimizer.step()
            self.untoggle_optimizer(policy_optimizer)


        # 记录详细信息，避免重复记录
        info = {**critic_info, **policy_info}
        for key, value in info.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=False, prog_bar=True)

        return info

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        验证步骤

        Args:
            batch: 包含验证数据的字典
            batch_idx: 批次索引

        Returns:
            验证损失值
        """
        device = self.device

        # 数据预处理
        observations = batch["observations"].float().to(device)
        action_chunks = batch["action_chunks"].float().to(device)
        next_observations = batch["next_observations"].float().to(device)
        rewards = batch["rewards"].float().to(device)
        # 确保奖励维度为[B, H, 1]，即使原始数据是[B, H]也适用
        if rewards.dim() == 2:
            rewards = rewards.unsqueeze(-1)
        dones = batch["dones"].float().to(device)

        # 计算各个组件的损失
        critic_loss, critic_info = self.net.loss_critic(observations, action_chunks, next_observations, rewards,
                                                        dones)

        policy_loss, policy_info = self.net.loss_policy(observations, action_chunks)

        # 计算总损失
        validation_loss = critic_loss + policy_loss

        # 记录奖励和详细信息
        self.log("val/reward", rewards.mean(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", validation_loss, on_step=False, on_epoch=True, prog_bar=True)
        info = {**critic_info, **policy_info}
        for key, value in info.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=False)

        return validation_loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        测试步骤

        Args:
            batch: 包含测试数据的字典
            batch_idx: 批次索引

        Returns:
            动作均方误差
        """
        observations = batch["observations"].float().to(self.device)
        # 移除不必要的squeeze操作
        predicted_actions = self(observations)
        actual_actions = batch["action_chunks"].float().to(self.device)
        action_mse = F.mse_loss(predicted_actions, actual_actions)
        self.log("test/action_mse", action_mse, on_step=False, on_epoch=True)
        return action_mse

    def on_train_batch_end(self, outputs: Union[Dict[str, Any], None], batch: Any, batch_idx: int) -> None:
        """
        训练批次结束回调，用于更新目标网络

        Args:
            outputs: 训练步骤的输出，可能为None
            batch: 当前批次数据
            batch_idx: 批次索引
        """
        if (int(self.step_idx) % self.cfg.target_update_freq) == 0:
            self.net.update_target(self.cfg.tau)

    def configure_optimizers(self) -> Tuple[Dict, ...]:
        """
        配置优化器和学习率调度器

        Returns:
            包含优化器和调度器的元组
        """
        critic_optimizer = optim.Adam(self.net.critic.parameters(), lr=self.cfg.learning_rate)
        policy_optimizer = optim.Adam(self.net.actor.parameters(), lr=self.cfg.learning_rate)
        critic_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=10, gamma=0.1)
        policy_scheduler = optim.lr_scheduler.StepLR(policy_optimizer, step_size=10, gamma=0.1)
        return (
            {"optimizer": critic_optimizer, "lr_scheduler": {"scheduler": critic_scheduler, "interval": "epoch"}},
            {"optimizer": policy_optimizer, "lr_scheduler": {"scheduler": policy_scheduler, "interval": "epoch"}},
        )