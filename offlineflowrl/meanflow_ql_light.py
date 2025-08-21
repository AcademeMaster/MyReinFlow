
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.functional import jvp
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torch import optim

# 设置Float32矩阵乘法精度以更好地利用Tensor Core
torch.set_float32_matmul_precision('high')


# ========= Config =========
@dataclass
class Config:
    hidden_dim: int = 256
    time_dim: int = 64
    pred_horizon: int = 5
    learning_rate: float = 3e-4
    grad_clip_value: float = 1.0
    cql_alpha: float = 1.0
    cql_temp: float = 1.0
    cql_num_samples: int = 10  # Added for configurability
    tau: float = 0.005
    gamma: float = 0.99
    inference_steps: int = 1
    normalize_q_loss: bool = True
    batch_size: int = 32
    num_workers: int = 4
    target_update_freq: int = 1  # 每 N 个 batch 软更新 target critic
    actor_update_freq: int = 2  # 每 N 个 batch 更新一次 actor


# ========= Small modules =========
class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TimeEmbedding(nn.Module):
    """sin/cos time embedding + MLP"""

    def __init__(self, time_dim: int, max_period: int = 10_000):
        super().__init__()
        assert time_dim % 2 == 0, "time_dim must be even"
        half = time_dim // 2
        exponents = torch.arange(half, dtype=torch.float32) / float(half)
        freqs = 1.0 / (max_period ** exponents)
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.time_dim = time_dim

    def forward(self, t: Tensor) -> Tensor:
        t = t.view(-1).float()  # [B]
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0)  # [B, half]
        enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, time_dim]
        return self.mlp(enc)


# ========= Critic (Double Q over obs + action-chunk) =========
class DoubleCriticObsAct(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, action_horizon: int):
        super().__init__()
        total_action_dim = action_dim * action_horizon

        def make_net():
            return nn.Sequential(
                nn.Linear(obs_dim + total_action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = make_net()
        self.q2 = make_net()

    @staticmethod
    def _prep(obs: Tensor, actions: Tensor) -> Tensor:
        # obs: [B, seq, obs_dim] or [B, obs_dim] - flatten sequences if present (assumes concatenation, no seq modeling)
        if obs.dim() > 2:
            obs = obs.reshape(obs.shape[0], -1)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)
        # actions: [B,H,A] or [B,H*A]
        if actions.dim() == 3:
            act = actions.reshape(actions.shape[0], -1)
        elif actions.dim() == 2:
            act = actions
        else:
            raise ValueError(f"bad actions dim={actions.dim()}")
        return torch.cat([obs, act], dim=-1)

    def forward(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        x = self._prep(obs, actions)
        return self.q1(x), self.q2(x)


# ========= Time-conditioned flow model (predicts velocity [B,H,A]) =========
# 输入obs,action*horizon,时间t,r
class MeanTimeCondFlow(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, time_dim: int, pred_horizon: int):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.pred_horizon = pred_horizon
        self.t_embed = TimeEmbedding(time_dim)
        self.r_embed = TimeEmbedding(time_dim)
        self.obs_embed = FeatureEmbedding(obs_dim, hidden_dim)
        self.noise_embed = FeatureEmbedding(action_dim, hidden_dim)
        joint_in = hidden_dim + hidden_dim + time_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(joint_in, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _norm_obs(self, obs: Tensor) -> Tensor:
        return obs.reshape(obs.shape[0], -1) if obs.dim() > 2 else (obs.unsqueeze(0) if obs.dim() == 1 else obs)

    def _norm_z(self, z: Tensor) -> Tensor:
        if z.dim() == 3: return z
        if z.dim() == 2 and z.shape[1] == self.pred_horizon * self.action_dim:
            return z.view(z.shape[0], self.pred_horizon, self.action_dim)
        if z.dim() == 1:  # Added for single sample
            return z.view(1, self.pred_horizon,
                          self.action_dim) if z.numel() == self.pred_horizon * self.action_dim else z.view(1, 1,
                                                                                                           self.action_dim)
        raise ValueError(f"bad z shape: {z.shape}")

    @staticmethod
    def _norm_time(t: Tensor, B: int) -> Tensor:
        if t.dim() == 0: return t.new_full((B,), float(t))
        if t.dim() == 1:
            if t.numel() == B: return t
            if t.numel() == 1: return t.repeat(B)
            t = t.view(-1)
            return t[:B] if t.numel() >= B else torch.cat([t, t.new_zeros(B - t.numel())], dim=0)
        return t.view(-1)[:B]

    def forward(self, obs: Tensor, z: Tensor, r: Tensor, t: Tensor) -> Tensor:
        obs = self._norm_obs(obs)
        z = self._norm_z(z)
        B, H, A = z.shape
        t = self._norm_time(t, B)
        r = self._norm_time(r, B)

        te = self.t_embed(t)  # [B, Td]
        re = self.r_embed(r)  # [B, Td]
        oe = self.obs_embed(obs)  # [B, Hd]

        ne = self.noise_embed(z.reshape(B * H, A)).view(B, H, -1)  # [B,H,Hd]
        te = te.unsqueeze(1).repeat(1, H, 1)
        re = re.unsqueeze(1).repeat(1, H, 1)
        oe = oe.unsqueeze(1).repeat(1, H, 1)
        x = torch.cat([oe, ne, re, te], dim=-1)  # [B,H,*]
        return self.net(x)  # [B,H,A]


# ========= Actor (MeanFlow) =========
class MeanFlowActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.pred_horizon = cfg.pred_horizon
        self.action_dim = action_dim
        self.model = MeanTimeCondFlow(obs_dim, action_dim, cfg.hidden_dim, cfg.time_dim, cfg.pred_horizon)

    @staticmethod
    def sample_t_r(n: int, device) -> Tuple[Tensor, Tensor]:
        t = torch.rand(n, device=device)
        r = torch.rand(n, device=device) * t
        return t, r

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, Tensor], n_steps: int = 1) -> Tensor:
        self.model.eval()
        device = next(self.parameters()).device
        obs = batch["observations"].to(device)  # [B, seq, obs_dim]
        obs_cond = obs[:, -1, :]
        x = torch.randn(obs.size(0), self.pred_horizon, self.action_dim, device=device)
        return self.sample_mean_flow(obs_cond, x, n_steps=n_steps)

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor], n_steps: int = 1) -> Tensor:
        return self.predict_action_chunk(batch, n_steps)[:, 0, :]

    def sample_mean_flow(self, obs_cond: Tensor, x: Tensor, n_steps: int = 1) -> Tensor:
        device = next(self.parameters()).device
        obs_cond, x = obs_cond.to(device), x.to(device)
        n_steps = max(1, int(n_steps))
        dt = 1.0 / n_steps
        for i in range(n_steps, 0, -1):
            r = torch.full((x.shape[0],), (i - 1) * dt, device=device)
            t = torch.full((x.shape[0],), i * dt, device=device)
            v = self.model(obs_cond, x, r, t)
            x = x - v * dt
        return torch.clamp(x, -1, 1)

    def flow_bc_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """MeanFlow 训练损失（JVP 版）"""
        device = next(self.parameters()).device
        obs = batch["observations"].to(device)  # [B, seq, obs_dim]
        act = batch["actions"].to(device)  # [B, H, A]
        noise = torch.randn_like(act)
        t, r = self.sample_t_r(act.shape[0], device=device)
        z = (1 - t.view(-1, 1, 1)) * act + t.view(-1, 1, 1) * noise
        obs_cond = obs[:, -1, :]
        v = noise - act

        obs_cond = obs_cond.requires_grad_(True)
        z = z.requires_grad_(True)
        r = r.requires_grad_(True)
        t = t.requires_grad_(True)

        v_obs = torch.zeros_like(obs_cond)
        v_z = v
        v_r = torch.zeros_like(r)
        v_t = torch.ones_like(t)

        u_pred, dudt = jvp(lambda *ins: self.model(*ins),
                           (obs_cond, z, r, t),
                           (v_obs, v_z, v_r, v_t),
                           create_graph=True)

        delta = torch.clamp(t - r, min=1e-6).view(-1, 1, 1)  # Added epsilon for stability
        u_tgt = (v - delta * dudt).detach()
        return F.mse_loss(u_pred, u_tgt)


# ========= Whole RL model (Actor + Double Q + target) =========
class ConservativeMeanFQL(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.actor = MeanFlowActor(obs_dim, action_dim, cfg)
        self.critic = DoubleCriticObsAct(obs_dim, action_dim, cfg.hidden_dim, cfg.pred_horizon)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

    @staticmethod
    def discounted_returns(rewards: Tensor, gamma: float) -> Tensor:
        # rewards: [B,H] or [B,H,1]
        if rewards.dim() == 3 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
        if rewards.dim() != 2:
            raise ValueError("rewards must be 2D [B,H] or [B,H,1]")
        B, H = rewards.shape
        factors = (gamma ** torch.arange(H, device=rewards.device, dtype=rewards.dtype)).unsqueeze(0)
        return torch.sum(rewards * factors, dim=1)  # [B]

    def loss_critic(self, obs: Tensor, actions: Tensor, next_obs: Tensor,
                    rewards: Tensor, terminated: Tensor, gamma: float) -> Tuple[Tensor, Dict]:
        B = obs.shape[0]
        with torch.no_grad():
            batch_next = {"observations": next_obs,
                          "actions": actions}  # Note: actions not used here, but kept for consistency
            next_actions = self.actor.predict_action_chunk(batch_next, n_steps=self.cfg.inference_steps)
            next_q1, next_q2 = self.target_critic(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2).view(B)

            ret = self.discounted_returns(rewards, gamma)  # [B]
            term = terminated.view(-1).float()  # Robust flatten
            bootstrap = (1.0 - term) * (gamma ** self.cfg.pred_horizon) * next_q
            target = ret + bootstrap  # [B]

        q1, q2 = self.critic(obs, actions)  # [B,1]
        td_loss = F.mse_loss(q1, target.view(B, 1)) + F.mse_loss(q2, target.view(B, 1))

        # CQL regularizer
        num_samples = self.cfg.cql_num_samples
        obs_cond = obs[:, -1, :]
        rep_obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)
        noise = torch.randn(B * num_samples, self.cfg.pred_horizon, self.actor.action_dim, device=obs.device)
        sampled_actions = self.actor.sample_mean_flow(rep_obs_cond, noise,
                                                      n_steps=self.cfg.inference_steps)  # Use config steps

        rep_obs = obs.repeat_interleave(num_samples, dim=0)
        q1s, q2s = self.critic(rep_obs, sampled_actions)
        q1s = q1s.view(B, num_samples)
        q2s = q2s.view(B, num_samples)
        temp = self.cfg.cql_temp
        cql1 = torch.logsumexp(q1s / temp, dim=1).mean() * temp - q1.mean()
        cql2 = torch.logsumexp(q2s / temp, dim=1).mean() * temp - q2.mean()
        cql = (cql1 + cql2) * self.cfg.cql_alpha

        total = td_loss + cql
        info = dict(td_loss=td_loss.item(), cql_loss=cql.item(), total_critic_loss=total.item(),
                    q1_mean=q1.mean().item(), q2_mean=q2.mean().item(), target_mean=target.mean().item())
        return total, info

    def loss_actor(self, obs: Tensor, action_batch: Tensor) -> Tuple[Tensor, Dict]:
        batch = {"observations": obs, "actions": action_batch}
        bc = self.actor.flow_bc_loss(batch)
        # Q guidance
        actor_actions = self.actor.predict_action_chunk(batch, n_steps=self.cfg.inference_steps)
        q1, q2 = self.critic(obs, actor_actions)
        q = torch.min(q1, q2)
        q_loss = -q.mean()
        if self.cfg.normalize_q_loss:
            q_loss = q_loss * (1.0 / (torch.abs(q).mean().detach() + 1e-8))
        loss = bc + q_loss
        info = dict(loss_actor=loss.item(), loss_bc_flow=bc.item(), q_loss=q_loss.item(), q_mean=q.mean().item())
        return loss, info

    def update_target(self, tau: float):
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1 - tau) + p.data * tau)


# ========= Dataset / DataModule =========
class ReplayBuffer(Dataset):
    def __init__(self, capacity=100000):
        self.buf: List[Dict[str, np.ndarray]] = []
        self.capacity = capacity
        self.idx = 0

    def add(self, item: Dict[str, np.ndarray]):
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.idx] = item
        self.idx = (self.idx + 1) % self.capacity

    def __len__(self):
        return len(self.buf)

    def __getitem__(self, i):
        return self.buf[i]


class ReplayDM(L.LightningDataModule):
    def __init__(self, train_buffer: ReplayBuffer, val_buffer: ReplayBuffer, cfg: Config):
        super().__init__()
        self.train_buffer = train_buffer
        self.val_buffer = val_buffer
        self.cfg = cfg

    def train_dataloader(self):
        return DataLoader(self.train_buffer, batch_size=self.cfg.batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_buffer, batch_size=self.cfg.batch_size, shuffle=False,
                          num_workers=self.cfg.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)


# ========= LightningModule (Auto-optim, official style) =========
class LitConservativeMeanFQL(L.LightningModule):
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.net = ConservativeMeanFQL(obs_dim, action_dim, cfg)
        # 禁用自动优化，因为我们使用多个优化器和频率控制
        self.automatic_optimization = False

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.net.actor.predict_action_chunk(batch, n_steps=self.cfg.inference_steps)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        device = self.device  # Use self.device for consistency
        obs = batch["observations"].float().to(device)
        actions = batch["actions"].float().to(device)
        next_obs = batch["next_observations"].float().to(device)
        rewards = batch["rewards"].float().to(device)
        terminated = batch["terminated"].float().to(device)

        # 获取优化器
        opt_c, opt_a = self.optimizers()

        # Critic step
        self.toggle_optimizer(opt_c)
        loss_c, info_c = self.net.loss_critic(obs, actions, next_obs, rewards, terminated, self.cfg.gamma)
        self.log("critic/loss", loss_c, on_step=True, prog_bar=True)
        for k, v in info_c.items():
            self.log(f"critic/{k}", v, on_step=True)
        opt_c.zero_grad()
        self.manual_backward(loss_c)
        if self.cfg.grad_clip_value:
            self.clip_gradients(opt_c, gradient_clip_val=self.cfg.grad_clip_value, gradient_clip_algorithm="norm")
        opt_c.step()
        self.untoggle_optimizer(opt_c)

        # Actor step - 控制 actor 更新频率：仅每 N 个 batch 更新
        if (batch_idx % self.cfg.actor_update_freq) == 0:
            self.toggle_optimizer(opt_a)
            loss_a, info_a = self.net.loss_actor(obs, actions)
            self.log("actor/loss", loss_a, on_step=True, prog_bar=True)
            for k, v in info_a.items():
                self.log(f"actor/{k}", v, on_step=True)
            opt_a.zero_grad()
            self.manual_backward(loss_a)
            if self.cfg.grad_clip_value:
                self.clip_gradients(opt_a, gradient_clip_val=self.cfg.grad_clip_value, gradient_clip_algorithm="norm")
            opt_a.step()
            self.untoggle_optimizer(opt_a)

        return {"loss": loss_c}  # Optional, for compatibility

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        device = self.device
        obs = batch["observations"].float().to(device)
        actions = batch["actions"].float().to(device)
        next_obs = batch["next_observations"].float().to(device)
        rewards = batch["rewards"].float().to(device)
        terminated = batch["terminated"].float().to(device)

        # 评估critic loss
        val_loss, info = self.net.loss_critic(obs, actions, next_obs, rewards, terminated, self.cfg.gamma)
        self.log("val/critic_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # 评估actor loss
        val_actor_loss, actor_info = self.net.loss_actor(obs, actions)
        self.log("val/actor_loss", val_actor_loss, on_step=False, on_epoch=True, prog_bar=True)

        # 记录其他指标
        for k, v in info.items():
            self.log(f"val/critic_{k}", v, on_step=False, on_epoch=True)
        for k, v in actor_info.items():
            self.log(f"val/actor_{k}", v, on_step=False, on_epoch=True)

        return val_loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int):
        predicted_actions = self(batch)
        actual_actions = batch["actions"].float()
        action_mse = F.mse_loss(predicted_actions, actual_actions)
        self.log("test/action_mse", action_mse, on_step=False, on_epoch=True)
        return action_mse

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # 目标网络软更新（按频率）
        if (batch_idx % self.cfg.target_update_freq) == 0:
            self.net.update_target(self.cfg.tau)

    def configure_optimizers(self):
        opt_c = optim.Adam(self.net.critic.parameters(), lr=self.cfg.learning_rate)
        opt_a = optim.Adam(self.net.actor.parameters(), lr=self.cfg.learning_rate)
        sched_c = optim.lr_scheduler.StepLR(opt_c, step_size=10, gamma=0.1)
        sched_a = optim.lr_scheduler.StepLR(opt_a, step_size=10, gamma=0.1)
        return (
            {"optimizer": opt_c, "lr_scheduler": {"scheduler": sched_c, "interval": "epoch"}},
            {"optimizer": opt_a, "lr_scheduler": {"scheduler": sched_a, "interval": "epoch"}},
        )


# ========= Demo main (合成数据) =========
def build_train_val_buffers(cfg: Config, obs_dim=10, action_dim=4, seq_len=1):
    """构建训练和验证缓冲区"""
    H = cfg.pred_horizon
    # 创建独立的训练和验证缓冲区
    train_buf = ReplayBuffer(1600)  # 80% 数据用于训练
    val_buf = ReplayBuffer(400)  # 20% 数据用于验证

    # 填充训练数据
    for _ in range(800):
        item = {
            "observations": np.random.randn(seq_len, obs_dim).astype(np.float32),
            "actions": np.random.randn(H, action_dim).astype(np.float32),
            "next_observations": np.random.randn(seq_len, obs_dim).astype(np.float32),
            "rewards": np.random.randn(H).astype(np.float32),
            "terminated": np.random.choice([0.0, 1.0], size=(1,)).astype(np.float32),
        }
        train_buf.add(item)

    # 填充验证数据
    for _ in range(200):
        item = {
            "observations": np.random.randn(seq_len, obs_dim).astype(np.float32),
            "actions": np.random.randn(H, action_dim).astype(np.float32),
            "next_observations": np.random.randn(seq_len, obs_dim).astype(np.float32),
            "rewards": np.random.randn(H).astype(np.float32),
            "terminated": np.random.choice([0.0, 1.0], size=(1,)).astype(np.float32),
        }
        val_buf.add(item)

    return train_buf, val_buf


def main():
    cfg = Config()
    obs_dim, action_dim = 10, 4
    # 构建独立的训练和验证缓冲区
    train_buffer, val_buffer = build_train_val_buffers(cfg, obs_dim, action_dim, seq_len=1)

    dm = ReplayDM(train_buffer, val_buffer, cfg)

    model = LitConservativeMeanFQL(obs_dim, action_dim, cfg)

    # 创建回调函数
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    checkpoint_callback = ModelCheckpoint(
        monitor='val/critic_loss',
        dirpath='checkpoints/meanflow_ql',
        filename='meanflow-ql-{epoch:02d}-{val/critic_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    early_stop = EarlyStopping(monitor="val/critic_loss", patience=5, mode="min")

    trainer = L.Trainer(
        max_epochs=20,
        limit_train_batches=100,
        limit_val_batches=20,
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, early_stop],
        precision="32-true",  # Changed from "16-mixed" to "32-true" to avoid model summary warning
    )

    # 训练模型
    trainer.fit(model=model, datamodule=dm)

    # 测试模型
    print("\nTesting the trained model:")
    trainer.test(model, dataloaders=dm.val_dataloader())


if __name__ == "__main__":
    main()
