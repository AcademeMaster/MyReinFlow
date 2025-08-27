import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Tuple
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
        obs = batch["observations"].float().to(device)  # [B, obs_dim]
        obs = obs.squeeze(1)  # 从 [B, 1, obs_dim] 转换为 [B, obs_dim]
        actions = batch["action_chunks"].float().to(device)  # [B, H, A]
        next_obs = batch["next_observations"].float().to(device)  # [B, obs_dim]
        next_obs = next_obs.squeeze(1)  # 从 [B, 1, obs_dim] 转换为 [B, obs_dim]
        rewards = batch["rewards"].float().to(device)  # [B, H]
        terminated = batch["terminations"].float().to(device)  # [B, 1]

        # 获取优化器
        opt_c, opt_v, opt_p = self.optimizers()
        # 更新 step_idx
        self.step_idx += 1

        # 初始化损失值和信息字典，确保在所有条件下都有定义
        qf_loss = torch.tensor(0.0, device=device)
        vf_loss = torch.tensor(0.0, device=device)
        policy_loss = torch.tensor(0.0, device=device)
        qf_info: Dict[str, float] = {}
        vf_info: Dict[str, float] = {}
        policy_info: Dict[str, float] = {}

        # 根据更新频率分别更新各个网络组件
        if (int(self.step_idx) % self.cfg.q_update_period) == 0:
            # Critic网络更新步骤
            self.toggle_optimizer(opt_c)
            qf_loss, qf_info = self.net.loss_qf(obs, actions, next_obs, rewards, terminated)
            opt_c.zero_grad()
            self.manual_backward(qf_loss)
            if self.cfg.grad_clip_value:
                self.clip_gradients(opt_c, gradient_clip_val=self.cfg.grad_clip_value, gradient_clip_algorithm="norm")
            opt_c.step()
            self.untoggle_optimizer(opt_c)

        if (int(self.step_idx) % self.cfg.v_update_period) == 0:
            # Value function更新步骤
            self.toggle_optimizer(opt_v)
            vf_loss, vf_info = self.net.loss_vf(obs, actions)
            opt_v.zero_grad()
            self.manual_backward(vf_loss)
            if self.cfg.grad_clip_value:
                self.clip_gradients(opt_v, gradient_clip_val=self.cfg.grad_clip_value, gradient_clip_algorithm="norm")
            opt_v.step()
            self.untoggle_optimizer(opt_v)

        if (int(self.step_idx) % self.cfg.policy_update_period) == 0:
            # Policy更新步骤
            self.toggle_optimizer(opt_p)
            policy_loss, policy_info = self.net.loss_policy(obs, actions)
            opt_p.zero_grad()
            self.manual_backward(policy_loss)
            if self.cfg.grad_clip_value:
                self.clip_gradients(opt_p, gradient_clip_val=self.cfg.grad_clip_value, gradient_clip_algorithm="norm")
            opt_p.step()
            self.untoggle_optimizer(opt_p)

        # 计算总损失
        total_loss = qf_loss + vf_loss + policy_loss

        # 记录训练日志
        self.log("critic/loss", qf_loss, on_step=True, prog_bar=True)
        self.log("vf/loss", vf_loss, on_step=True, prog_bar=True)
        self.log("policy/loss", policy_loss, on_step=True, prog_bar=True)
        self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)
        
        # 记录详细信息
        info = {**qf_info, **vf_info, **policy_info}
        for k, v in info.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False)

        return {"loss": total_loss}

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
        obs = batch["observations"].float().to(device)
        obs = obs.squeeze(1) if obs.dim() > 2 else obs
        actions = batch["action_chunks"].float().to(device)
        next_obs = batch["next_observations"].float().to(device)
        next_obs = next_obs.squeeze(1) if next_obs.dim() > 2 else next_obs
        rewards = batch["rewards"].float().to(device)
        rewards = rewards.unsqueeze(-1) if rewards.dim() == 2 else rewards
        terminated = batch["terminations"].float().to(device)

        # 计算各个组件的损失
        qf_loss, qf_info = self.net.loss_qf(obs, actions, next_obs, rewards, terminated)
        vf_loss, vf_info = self.net.loss_vf(obs, actions)
        policy_loss, policy_info = self.net.loss_policy(obs, actions)

        # 计算总损失
        val_loss = qf_loss + vf_loss + policy_loss
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # 记录奖励和详细信息
        self.log("val/reward", rewards.mean(), on_step=False, on_epoch=True, prog_bar=True)

        info = {**qf_info, **vf_info, **policy_info}
        for k, v in info.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False)

        return val_loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        测试步骤

        Args:
            batch: 包含测试数据的字典
            batch_idx: 批次索引

        Returns:
            动作均方误差
        """
        obs = batch["observations"].float().to(self.device)
        obs = obs.squeeze(1) if obs.dim() > 2 else obs
        predicted_actions = self(obs)
        actual_actions = batch["action_chunks"].float().to(self.device)
        action_mse = F.mse_loss(predicted_actions, actual_actions)
        self.log("test/action_mse", action_mse, on_step=False, on_epoch=True)
        return action_mse

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """
        训练批次结束回调，用于更新目标网络

        Args:
            outputs: 训练步骤的输出
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
        opt_c = optim.Adam(self.net.critic.parameters(), lr=self.cfg.learning_rate)
        opt_v = optim.Adam(self.net.vf.parameters(), lr=self.cfg.learning_rate)
        opt_p = optim.Adam(self.net.actor.parameters(), lr=self.cfg.learning_rate)
        sched_c = optim.lr_scheduler.StepLR(opt_c, step_size=10, gamma=0.1)
        sched_v = optim.lr_scheduler.StepLR(opt_v, step_size=10, gamma=0.1)
        sched_p = optim.lr_scheduler.StepLR(opt_p, step_size=10, gamma=0.1)
        return (
            {"optimizer": opt_c, "lr_scheduler": {"scheduler": sched_c, "interval": "epoch"}},
            {"optimizer": opt_v, "lr_scheduler": {"scheduler": sched_v, "interval": "epoch"}},
            {"optimizer": opt_p, "lr_scheduler": {"scheduler": sched_p, "interval": "epoch"}},
        )