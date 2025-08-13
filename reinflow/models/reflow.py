import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

# 定义Sample命名元组，用于返回采样结果
Sample = namedtuple("Sample", "trajectories chains")


class ReFlow(nn.Module):
    """ReFlow模型 - 流匹配的正确实现
    
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

    def generate_trajectory(self, x1, x0, t):
        """流匹配的正确插值公式: x_t = (1-t)*x0 + t*x1
        
        参数:
            x1 (torch.Tensor): 目标数据点
            x0 (torch.Tensor): 初始噪声
            t (torch.Tensor): 时间点
            
        返回:
            torch.Tensor: 插值点
        """
        # x1,x0: (B,Ta,Da); t: (B,)
        t = t.view(-1, 1, 1)  # (B, 1, 1) for broadcasting
        xt = (1.0 - t) * x0 + t * x1
        return xt

    def sample_time(self, batch_size):
        """改进的时间采样策略
        
        参数:
            batch_size (int): 批次大小
            
        返回:
            torch.Tensor: 采样的时间点
        """
        if self.noise_schedule == 'prioritized':
            # 更多地采样中间时间点
            return torch.sqrt(torch.rand(batch_size, device=self.device))
        elif self.noise_schedule == 'cosine':
            # cosine schedule for better training stability
            u = torch.rand(batch_size, device=self.device)
            return torch.cos(u * np.pi / 2) ** 2
        else:
            # uniform sampling
            return torch.rand(batch_size, device=self.device)

    def generate_target(self, x1):
        """生成训练目标
        
        参数:
            x1 (torch.Tensor): 目标数据点
            
        返回:
            tuple: 包含(xt, t)和v的元组
        """
        B = x1.shape[0]
        t = self.sample_time(B)
        
        # 改进噪声生成：基于动作范围的合理噪声
        act_range = self.act_range[1] - self.act_range[0]
        noise_scale = min(act_range * 0.3, 1.0)  # 限制噪声规模
        
        # 使用更稳定的噪声初始化
        x0 = torch.randn_like(x1, device=self.device) * noise_scale
        
        # 流匹配插值
        xt = self.generate_trajectory(x1, x0, t)
        
        # 真正的目标速度场：dx/dt = x1 - x0
        v = x1 - x0
        
        return (xt, t), v

    def loss(self, xt, t, cond, v):
        """计算流匹配损失
        
        参数:
            xt (torch.Tensor): 插值点
            t (torch.Tensor): 时间点
            cond (dict): 条件字典
            v (torch.Tensor): 目标速度场
            
        返回:
            torch.Tensor: 损失值
        """
        v_hat = self.network(xt, t, cond)
        mse_loss = F.mse_loss(v_hat, v, reduction='none')
        
        # 改进的时间加权策略
        if self.noise_schedule != 'uniform':
            # 对边界时间点给予更多权重
            t_reshaped = t.view(-1, 1, 1)
            weights = 1.0 + 2.0 * torch.minimum(t_reshaped, 1.0 - t_reshaped)
            loss = (mse_loss * weights).mean()
        else:
            loss = mse_loss.mean()
        
        return loss

    @torch.no_grad()
    def sample(self, cond, inference_steps=20, record_intermediate=False, 
               clip_intermediate_actions=True, guidance_scale=1.0, use_euler=True):
        """改进的采样过程
        
        参数:
            cond (dict): 条件字典
            inference_steps (int): 推理步数
            record_intermediate (bool): 是否记录中间结果
            clip_intermediate_actions (bool): 是否裁剪中间动作
            guidance_scale (float): 引导尺度
            use_euler (bool): 是否使用欧拉积分
            
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

        # 更稳定的时间步长调度
        if use_euler:
            # Euler积分
            dt = 1.0 / inference_steps
            for i in range(inference_steps):
                t = torch.full((B,), i * dt, device=self.device)
                vt = self.network(x_hat, t, cond)
                
                # 可选的引导
                if guidance_scale > 1.0 and 'next_state' in cond:
                    ns = cond['next_state']
                    if ns.dim() == 3:
                        ns = ns[:, 0, :]
                    next_cond = {'state': ns}
                    vt_guided = self.network(x_hat, t, next_cond)
                    vt = vt + (vt_guided - vt) * (guidance_scale - 1.0) * 0.5

                x_hat = x_hat + vt * dt
                
                # 动作裁剪
                if clip_intermediate_actions:
                    x_hat = x_hat.clamp(self.act_range[0], self.act_range[1])

                if record_intermediate:
                    x_hat_list[i] = x_hat
        else:
            # 使用Runge-Kutta方法（更精确但更慢）
            dt = 1.0 / inference_steps
            for i in range(inference_steps):
                t = torch.full((B,), i * dt, device=self.device)
                
                # RK4步骤
                k1 = self.network(x_hat, t, cond)
                k2 = self.network(x_hat + 0.5 * dt * k1, t + 0.5 * dt, cond)
                k3 = self.network(x_hat + 0.5 * dt * k2, t + 0.5 * dt, cond)
                k4 = self.network(x_hat + dt * k3, t + dt, cond)
                
                x_hat = x_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                
                if clip_intermediate_actions:
                    x_hat = x_hat.clamp(self.act_range[0], self.act_range[1])
                    
                if record_intermediate:
                    x_hat_list[i] = x_hat

        return Sample(trajectories=x_hat, chains=x_hat_list if record_intermediate else None)