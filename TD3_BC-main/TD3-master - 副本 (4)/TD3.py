import random
import copy
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        self.max_action = max_action
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        return self.max_action * torch.tanh(self.net(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.Q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2 architecture
        self.Q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.Q1(sa), self.Q2(sa)

    def Q1Value(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.Q1(sa)


class TD3:
    ''' TD3算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, 
    sigma, actor_lr, critic_lr, tau, gamma, device, policy_noise=0.2, 
    noise_clip=0.5, policy_freq=2, replay_buffer=None, action_horizon=1):

        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.max_action = action_bound
        self.discount = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.action_horizon=action_horizon
        self.action_chunk_dim=self.action_dim*self.action_horizon

        self.actor = Actor(state_dim, self.action_chunk_dim, action_bound).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        
        self.critic = Critic(state_dim, self.action_chunk_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 通过依赖注入获取replay_buffer
        self.replay_buffer = replay_buffer

        # 创建动作buffer
        self.action_deque = deque(maxlen=action_horizon)

        # 定义折扣因子，预先计算好的（基于action_horizon）
        self.gamma_powers = torch.pow(self.discount, torch.arange(self.action_horizon, dtype=torch.float).to(self.device))

    @torch.no_grad()
    def take_action(self, state, add_noise=True):
        # 如果还有动作，就每次单步取出动作用于执行
        if len(self.action_deque) > 0:
            action = self.action_deque.popleft()
        else:
            # 执行actor推理action chunking，并保存
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            action_chunk = self.actor(state)
            if add_noise:
                noise = torch.normal(0, self.sigma, size=action_chunk.shape).to(self.device)
                action_chunk = action_chunk + noise
                action_chunk = torch.clamp(action_chunk, -self.max_action, self.max_action)
            
            # 将action chunk分解为单个动作并存入deque
            action_chunk_np = action_chunk.cpu().data.numpy().flatten()
            for i in range(self.action_horizon):
                start_idx = i * self.action_dim
                end_idx = (i + 1) * self.action_dim
                single_action = action_chunk_np[start_idx:end_idx]
                self.action_deque.append(torch.tensor(single_action, dtype=torch.float).to(self.device))
            
            # 取出第一个动作
            action = self.action_deque.popleft()
            
        return action.cpu().data.numpy().flatten()


    # 计算前H步的环境累计折扣奖励 G=sum(gamma^t * reward_t)=r1+gamma*r2+...+gamma^H*rH
    def compute_discounted_reward(self, rewards_tensor):
        """计算多步折扣奖励（并行优化版本）
        
        Args:
            rewards_list: List[List] - 每个样本的奖励序列
            actual_steps: torch.Tensor - 每个样本的实际步数
            
        Returns:
            torch.Tensor - 折扣奖励张量 [batch_size, 1]
        """

        # 并行计算折扣奖励：逐元素相乘后按行求和
        discounted_matrix = rewards_tensor * self.gamma_powers
        discounted_rewards = discounted_matrix.sum(dim=1, keepdim=True)
        return discounted_rewards

    def update(self, batch_size):
        if self.replay_buffer is None or self.replay_buffer.size() < batch_size:
            return
            
        self.total_it += 1
        
        # 多步采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # noise应该与actor_target输出形状一致，而不是与actions一致
            target_action = self.actor_target(next_states)  # [batch_size, action_chunk_dim]
            noise = (torch.randn_like(target_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (target_action + noise).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_states, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            # 计算多步折扣奖励
            discounted_rewards = self.compute_discounted_reward(rewards)
            
            # 计算目标Q值：G + γ^n * Q(s', a')
            target_Q = discounted_rewards + (1 - dones) * (self.discount ** self.action_horizon) * target_Q


        current_Q1, current_Q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1Value(states, self.actor(states)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)