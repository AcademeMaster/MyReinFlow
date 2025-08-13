import torch
import torch.nn as nn

from reinflow.models.base import SinusoidalPosEmb


class FlowMLP(nn.Module):
    """改进的FlowMLP网络，用于流匹配
    
    参数:
        horizon_steps (int): 动作序列的时间步长
        action_dim (int): 动作空间的维度
        cond_dim (int): 条件（状态）的维度
        time_dim (int): 时间嵌入的维度
        mlp_dims (list): MLP隐藏层的维度列表
        activation_type (str): 激活函数类型
        dropout_rate (float): Dropout比率
    """
    def __init__(self, horizon_steps, action_dim, cond_dim, time_dim=32, 
                 mlp_dims=[512, 512, 256], activation_type="SiLU", dropout_rate=0.1):
        super().__init__()
        self.time_dim = time_dim
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.act_dim_total = action_dim * horizon_steps
        self.cond_dim = cond_dim
        self.dropout_rate = dropout_rate

        # 改进的时间嵌入
        self.time_proj = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),  # SiLU通常比ReLU性能更好
            nn.Dropout(dropout_rate),
            nn.Linear(time_dim * 2, time_dim),
        )

        # 改进的状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2),
            nn.LayerNorm(cond_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(cond_dim * 2, cond_dim),
            nn.LayerNorm(cond_dim),
        )

        # 动作预处理
        self.action_encoder = nn.Sequential(
            nn.Linear(self.act_dim_total, self.act_dim_total),
            nn.LayerNorm(self.act_dim_total),
            nn.SiLU(),
        )

        # 特征融合层
        input_dim = self.time_dim + self.act_dim_total + cond_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        # 主MLP网络（带残差连接）
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for dim in mlp_dims:
            layer_block = nn.ModuleList([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.SiLU() if activation_type == "SiLU" else nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            
            # 添加跳跃连接的投影层（如果维度不匹配）
            if prev_dim != dim:
                skip_proj = nn.Linear(prev_dim, dim)
            else:
                skip_proj = nn.Identity()
                
            self.layers.append(nn.ModuleDict({
                'main': nn.Sequential(*layer_block),
                'skip': skip_proj
            }))
            prev_dim = dim

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, self.act_dim_total * 2),  # 双倍输出用于残差
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),  # 输出层使用较小的dropout
            nn.Linear(self.act_dim_total * 2, self.act_dim_total),
        )
        
        # 最终输出归一化
        self.out_ln = nn.LayerNorm(self.act_dim_total)
        
        # 可学习的残差缩放
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, action, time, cond):
        """前向传播
        
        参数:
            action (torch.Tensor): 动作张量，形状为 (B, Ta, Da)
            time (torch.Tensor): 时间张量，形状为 (B,) 或 (B, 1)
            cond (dict): 条件字典，包含 'state' 键
            
        返回:
            torch.Tensor: 速度场预测，形状为 (B, Ta, Da)
        """
        # action: (B, Ta, Da)
        B, Ta, Da = action.shape
        action_flat = action.view(B, -1)

        # 确保时间维度正确
        if time.dim() == 2 and time.shape[1] == 1:
            time = time.view(-1)
        time = time.to(action.device)
        
        # 编码各组件
        time_emb = self.time_proj(time)
        state = cond["state"].to(action.device)
        encoded_state = self.state_encoder(state)
        encoded_action = self.action_encoder(action_flat)

        # 特征融合
        x = torch.cat([encoded_action, time_emb, encoded_state], dim=-1)
        x = self.feature_fusion(x)
        
        # 主网络（带残差连接）
        residual = x
        for layer_dict in self.layers:
            main_out = layer_dict['main'](x)
            skip_out = layer_dict['skip'](residual)
            x = main_out + skip_out * self.residual_scale
            residual = x
        
        # 输出层
        vel = self.output_layer(x)
        vel = self.out_ln(vel)
        
        # 添加输入动作的残差连接（有助于稳定性）
        vel = vel + action_flat * 0.1
        
        return vel.view(B, Ta, Da)