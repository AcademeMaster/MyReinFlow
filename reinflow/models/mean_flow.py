import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

# 定义Sample命名元组，用于返回采样结果
Sample = namedtuple("Sample", "trajectories chains")


class MeanFlow(nn.Module):
    """MeanFlow模型 - 另一种流匹配方法
    
    参数:
        network (nn.Module): 底层网络模型
        device (torch.device): 计算设备
        horizon_steps (int): 动作序列的时间步长
        action_dim (int): 动作空间的维度
        act_min (float): 动作空间的最小值
        act_max (float): 动作空间的最大值
        obs_dim (int): 观测空间的维度
        max_denoising_steps (int): 最大去噪步数
        seed (int, optional): 随机种子
        noise_schedule (str): 噪声调度类型
        sigma (float): 噪声标准差
    """
    def __init__(self, network, device, horizon_steps, action_dim, act_min, act_max, obs_dim, 
                 max_denoising_steps=50, seed=None, noise_schedule='uniform', sigma=0.1):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.network = network.to(device)
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.data_shape = (horizon_steps, action_dim)
        self.act_range = (float(act_min), float(act_max))
        self.obs_dim = obs_dim
        self.max_denoising_steps = int(max_denoising_steps)
        self.noise_schedule = noise_schedule
        self.sigma = sigma  # 噪声标准差，用于稳定性

    def sample_time_pairs(self, batch_size):
        """采样时间对
        
        参数:
            batch_size (int): 批次大小
            
        返回:
            tuple: 包含t0和t1的元组
        """
        if self.noise_schedule == 'prioritized':
            # 更多地采样中间时间点
            u = torch.rand(batch_size, device=self.device)
            t0 = torch.sqrt(u) * 0.8  # 缩小范围以避免边界问题
            t1 = t0 + 0.1 + 0.1 * torch.rand(batch_size, device=self.device)  # 确保t1 > t0
            t1 = torch.clamp(t1, max=1.0)  # 确保t1 <= 1
        elif self.noise_schedule == 'cosine':
            # cosine schedule for better training stability
            u = torch.rand(batch_size, device=self.device) * 0.8
            t0 = torch.cos(u * np.pi / 2) ** 2
            t1 = torch.cos((u - 0.1 - 0.1 * torch.rand(batch_size, device=self.device)) * np.pi / 2) ** 2
            t1 = torch.clamp(t1, min=0.0)  # 确保t1 >= 0
        else:
            # uniform sampling
            t0 = torch.rand(batch_size, device=self.device) * 0.9  # 避免t0太接近1
            t1 = t0 + 0.1 * torch.rand(batch_size, device=self.device)  # 确保t1 > t0
            t1 = torch.clamp(t1, max=1.0)  # 确保t1 <= 1
        
        return t0, t1

    def generate_mean_flow_target(self, x1):
        """生成Mean Flow训练目标
        
        参数:
            x1 (torch.Tensor): 目标数据点
            
        返回:
            tuple: 包含(xt0, xt1, t0, t1)和v的元组
        """
        B = x1.shape[0]
        t0, t1 = self.sample_time_pairs(B)
        
        # 改进噪声生成：基于动作范围的合理噪声
        act_range = self.act_range[1] - self.act_range[0]
        noise_scale = min(act_range * 0.3, 1.0)  # 限制噪声规模
        
        # 使用更稳定的噪声初始化
        x0 = torch.randn_like(x1, device=self.device) * noise_scale
        
        # 生成两个时间点的插值
        t0_view = t0.view(-1, 1, 1)  # (B, 1, 1) for broadcasting
        t1_view = t1.view(-1, 1, 1)  # (B, 1, 1) for broadcasting
        
        xt0 = (1.0 - t0_view) * x0 + t0_view * x1
        xt1 = (1.0 - t1_view) * x0 + t1_view * x1
        
        # 计算平均速度场
        dt = (t1 - t0).view(-1, 1, 1)  # (B, 1, 1)
        v = (xt1 - xt0) / dt  # 平均速度
        
        return (xt0, xt1, t0, t1), v

    def loss(self, xt0, t0, xt1, t1, cond, v_avg):
        """计算Mean Flow损失 - 简化版本
        
        参数:
            xt0 (torch.Tensor): 时间t0的插值点
            t0 (torch.Tensor): 时间点t0
            xt1 (torch.Tensor): 时间t1的插值点
            t1 (torch.Tensor): 时间点t1
            cond (dict): 条件字典
            v_avg (torch.Tensor): 平均速度场
            
        返回:
            torch.Tensor: 损失值
        """
        # 计算中间点
        t_mid = (t0 + t1) / 2
        xt_mid = (xt0 + xt1) / 2
        
        # 直接预测中间点的速度
        v_mid = self.network(xt_mid, t_mid, cond)
        
        # 使用简化的MSE损失
        mse_loss = F.mse_loss(v_mid, v_avg, reduction='mean')
        
        # 添加正则化项以确保稳定性
        reg_loss = 0.01 * torch.mean(torch.square(v_mid))
        
        loss = mse_loss + reg_loss
        
        return loss

    @torch.no_grad()
    def sample(self, cond, inference_steps=1, record_intermediate=False, 
               clip_intermediate_actions=True):
        """Mean Flow的采样方法
        
        参数:
            cond (dict): 条件字典
            inference_steps (int): 推理步数，对于Mean Flow通常为1
            record_intermediate (bool): 是否记录中间结果
            clip_intermediate_actions (bool): 是否裁剪中间动作
            
        返回:
            Sample: 包含轨迹和链的命名元组
        """
        B = cond['state'].shape[0]
        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps, B, *self.data_shape), device=self.device)

        # 改进的初始化
        act_range = self.act_range[1] - self.act_range[0]
        noise_scale = min(act_range * 0.3, 1.0)
        x_hat = torch.randn((B,) + self.data_shape, device=self.device) * noise_scale

        # Mean Flow通常只需要一步生成
        t = torch.ones(B, device=self.device) * 0.5  # 使用中间时间点
        x_hat = self.network(x_hat, t, cond)
        
        # 动作裁剪
        if clip_intermediate_actions:
            x_hat = x_hat.clamp(self.act_range[0], self.act_range[1])

        if record_intermediate:
            x_hat_list[0] = x_hat

        return Sample(trajectories=x_hat, chains=x_hat_list if record_intermediate else None)