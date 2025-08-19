import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from typing import Dict, Tuple
from torch.autograd.functional import jvp


class MeanTimeConditionedFlowModel(nn.Module):
    """带时间条件约束的流匹配模型"""

    def __init__(self, obs_dim: int, action_dim: int, config):
        """
        初始化时间条件流模型
        
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

    def forward(self, obs: torch.Tensor, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算速度场
        
        参数:
            obs: 观测张量 [B, obs_dim]
            z: 混合样本 [B, pred_horizon, action_dim]
            r: 参考时间步 [B, 1] 或 [B]
            t: 当前时间步 [B]

        返回:
            速度场 [B, pred_horizon, action_dim]
        """
        # 获取模型所在的设备
        device = next(self.parameters()).device

        # 将输入张量移动到模型设备
        obs = obs.to(device)
        z = z.to(device)
        r = r.to(device)
        t = t.to(device)

        # 确保r和t具有正确的维度
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # 嵌入时间
        t_emb = self.time_embed(t.float())  # [B, time_dim]

        # 嵌入观测
        obs_emb = self.obs_embed(obs)  # [B, hidden_dim]

        # 嵌入噪声
        # 对每个时间步独立处理噪声
        B, H, A = z.shape
        noise_emb = self.noise_embed(z.view(B * H, A))  # [B*H, hidden_dim]
        noise_emb = noise_emb.view(B, H, -1)  # [B, H, hidden_dim]

        # 扩展观测和时间嵌入以匹配时间维度
        obs_emb = obs_emb.unsqueeze(1).expand(-1, H, -1)  # [B, H, hidden_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, H, -1)  # [B, H, time_dim]

        # 合并特征
        combined = torch.cat([obs_emb, noise_emb, t_emb], dim=-1)  # [B, H, hidden_dim*2+time_dim]

        # 预测速度场
        velocity = self.joint_processor(combined)  # [B, H, action_dim]

        return velocity


class MeanFlowPolicyAgent:
    """MeanFlow策略代理"""

    def __init__(self, obs_dim: int, action_dim: int, config: "Config"):
        """
        初始化MeanFlow策略代理

        参数:
            obs_dim: 观测维度
            action_dim: 动作维度
            config: 配置对象
        """
        from config import Config  # 避免循环导入
        self.config: Config = config
        self.model = MeanTimeConditionedFlowModel(obs_dim, action_dim, config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.flow_matcher = ConditionalFlowMatcher(sigma=config.sigma)

    @staticmethod
    def sample_t_r(n_samples: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样时间点t和r
        
        参数:
            n_samples: 样本数量
            device: 设备类型
        
        返回:
            t和r的时间张量
        """
        t = torch.rand(n_samples, device=device)
        r = torch.rand(n_samples, device=device) * t
        return t.unsqueeze(1), r.unsqueeze(1)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, torch.Tensor], n_steps: int = 1) -> torch.Tensor:
        """
        预测动作块

        参数:
            batch: 包含观测数据的批次字典
            n_steps: 采样步数，1表示一步生成，>1表示多步迭代生成

        返回:
            预测的动作张量
        """
        self.model.eval()
        # 获取模型所在的设备
        device = next(self.model.parameters()).device

        observations = batch["observations"].to(device)  # [B, obs_horizon, obs_dim]

        # 使用最新的观测作为条件
        obs_cond = observations[:, -1, :]  # [B, obs_dim]

        # 生成初始噪声
        noise = torch.randn(
            observations.size(0), 
            self.config.pred_horizon, 
            self.config.action_dim
        ).to(device)
        
        # 使用MeanFlow采样
        x = self.sample_mean_flow(obs_cond, noise, n_steps=n_steps)
        
        return x

    @torch.no_grad()
    def select_action(self, batch: Dict[str, torch.Tensor], n_steps: int = 1) -> torch.Tensor:
        """
        根据环境观测选择单个动作

        参数:
            batch: 包含观测数据的批次字典
            n_steps: 采样步数

        返回:
            选择的动作
        """
        # 使用predict_action_chunk获取完整的动作块，然后只返回第一个动作
        action_chunk = self.predict_action_chunk(batch, n_steps=n_steps)
        return action_chunk[:, 0, :]  # 返回动作块中的第一个动作

    def sample_mean_flow(self, obs_cond: torch.Tensor, noise: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """
        使用MeanFlow进行采样生成动作
        
        参数:
            obs_cond: 条件观测 [B, obs_dim]
            noise: 初始噪声 [B, pred_horizon, action_dim]
            n_steps: 采样步数，1表示一步生成，>1表示多步迭代生成
            
        返回:
            生成的动作轨迹
        """
        device = next(self.model.parameters()).device
        obs_cond = obs_cond.to(device)
        x = noise.to(device)
        dt = 1.0 / n_steps
        
        for i in range(n_steps, 0, -1):
            r = torch.full((x.shape[0],), (i-1) * dt, device=device).unsqueeze(1)
            t = torch.full((x.shape[0],), i * dt, device=device).unsqueeze(1)
            velocity = self.model(obs_cond, x, r, t)
            x = x - velocity * dt
    
        return x

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播，计算MeanFlow损失

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

        # 根据Algorithm 1实现MeanFlow训练逻辑
        # 生成噪声
        noise = torch.randn_like(actions, device=device)
        
        # 采样时间步
        t, r = self.sample_t_r(actions.shape[0], device=device)
        
        # 创建混合样本
        z = (1 - t.unsqueeze(-1)) * actions + t.unsqueeze(-1) * noise
        v = noise - actions
        
        # 使用最新的观测作为条件
        obs_cond = observations[:, -1, :]  # [B, obs_dim]

        # 计算Jacobian-Vector Product
        u, dudt = jvp(
            func=self.model,
            inputs=(obs_cond, z, r, t),
            v=(torch.zeros_like(obs_cond), v, torch.zeros_like(r), torch.ones_like(t)),
            create_graph=True
        )
        
        # 计算目标速度场
        u_tgt = v - (t.unsqueeze(-1) - r.unsqueeze(-1)) * dudt
        u_tgt = u_tgt.detach()
        
        # 预测速度场
        predicted_velocity = self.model(obs_cond, z, r, t)
        
        # 计算损失
        loss = F.mse_loss(predicted_velocity, u_tgt)

        return loss


if __name__ == "__main__":
    # 测试模型功能
    from config import Config

    print("测试TimeConditionedFlowModel...")

    # 创建配置
    config = Config()
    config.action_dim = 7

    # 创建模型
    obs_dim = 23
    action_dim = 7
    model = MeanTimeConditionedFlowModel(obs_dim, action_dim, config)

    # 创建测试数据
    batch_size = 2
    obs = torch.randn(batch_size, obs_dim)
    t = torch.rand(batch_size)
    noise = torch.randn(batch_size, config.pred_horizon, action_dim)
    r = torch.zeros(batch_size, 1)

    # 测试前向传播
    output = model(obs, noise, r, t)
    print(f"输入观测维度: {obs.shape}")
    print(f"输入时间维度: {t.shape}")
    print(f"输入噪声维度: {noise.shape}")
    print(f"输出速度场维度: {output.shape}")

    print("\n测试FlowPolicyAgent...")

    # 创建代理
    agent = MeanFlowPolicyAgent(obs_dim, action_dim, config)

    # 创建测试批次
    batch = {
        "observations": torch.randn(batch_size, config.obs_horizon, obs_dim),
        "actions": torch.randn(batch_size, config.pred_horizon, action_dim)
    }

    # 测试代理前向传播
    loss = agent.forward(batch)
    print(f"损失值: {loss}")

    # 测试动作预测
    predicted_actions = agent.predict_action_chunk(batch)
    print(f"预测动作维度: {predicted_actions.shape}")
    
    # 测试一步采样方法 (等同于原来的onestep)
    predicted_actions_onestep = agent.predict_action_chunk(batch, n_steps=1)
    print(f"一步采样预测动作维度: {predicted_actions_onestep.shape}")
    
    # 测试多步采样方法 (等同于原来的iterative)
    predicted_actions_multistep = agent.predict_action_chunk(batch, n_steps=100)
    print(f"多步采样预测动作维度: {predicted_actions_multistep.shape}")

    print("\n所有测试完成!")
