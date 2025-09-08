import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, action_chunk_size=1):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		# 输出action_chunk_size个时间步的动作
		self.l3 = nn.Linear(256, action_dim * action_chunk_size)
		
		self.max_action = max_action
		self.action_dim = action_dim
		self.action_chunk_size = action_chunk_size
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		action_chunk = self.max_action * torch.tanh(self.l3(a))
		# 重塑为 (batch_size, action_chunk_size, action_dim)
		return action_chunk.view(-1, self.action_chunk_size, self.action_dim)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, action_chunk_size=1):
		super(Critic, self).__init__()

		self.action_chunk_size = action_chunk_size
		# 输入为state + flattened action chunk
		input_dim = state_dim + action_dim * action_chunk_size
		
		# Q1 architecture
		self.l1 = nn.Linear(input_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(input_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		# 如果action是3D tensor (batch_size, action_chunk_size, action_dim)，展平为2D
		if len(action.shape) == 3:
			action = action.view(action.size(0), -1)  # flatten action chunk
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		# 如果action是3D tensor (batch_size, action_chunk_size, action_dim)，展平为2D
		if len(action.shape) == 3:
			action = action.view(action.size(0), -1)  # flatten action chunk
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		action_chunk_size=1
	):

		self.action_chunk_size = action_chunk_size
		self.actor = Actor(state_dim, action_dim, max_action, action_chunk_size).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim, action_chunk_size).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0
		
		# Action chunk执行状态管理
		self.current_action_chunk = None
		self.action_step_index = 0
		self.chunk_exhausted = True


	def select_action(self, state):
		"""
		选择动作，支持action chunk的依次执行
		当action_chunk_size > 1时，会依次返回序列中的每个动作
		只有当前序列执行完毕后才会重新推理生成新的动作序列
		"""
		# 如果当前动作序列已执行完毕或者是第一次调用，则重新推理
		if self.chunk_exhausted or self.current_action_chunk is None:
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			self.current_action_chunk = self.actor(state).cpu().data.numpy()  # shape: (1, action_chunk_size, action_dim)
			self.action_step_index = 0
			self.chunk_exhausted = False
		
		# 获取当前步骤的动作
		current_action = self.current_action_chunk[0, self.action_step_index, :]  # shape: (action_dim,)
		
		# 更新步骤索引
		self.action_step_index += 1
		
		# 检查是否已执行完所有动作
		if self.action_step_index >= self.action_chunk_size:
			self.chunk_exhausted = True
		
		return current_action

	def reset_action_chunk(self):
		"""
		重置动作序列状态，通常在episode结束时调用
		确保下一个episode开始时重新推理生成新的动作序列
		"""
		self.current_action_chunk = None
		self.action_step_index = 0
		self.chunk_exhausted = True

	def compute_multistep_target(self, next_state, multi_step_reward, not_done):
		"""
		计算多步TD目标
		args:
			next_state: 下一个状态 (batch_size, state_dim)
			multi_step_reward: 多步reward (batch_size, action_chunk_size)
			not_done: 是否未结束 (batch_size, 1)
		returns:
			multistep_target: 多步TD目标 (batch_size, 1)
		"""
		with torch.no_grad():
			# 计算目标动作（下一个状态的action chunk）
			noise = (torch.randn(next_state.size(0), self.action_chunk_size, self.action_dim) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(device)
			next_action_chunk = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
			
			# 计算目标Q值
			target_Q1, target_Q2 = self.critic_target(next_state, next_action_chunk)
			target_Q = torch.min(target_Q1, target_Q2)
			
			# 计算多步TD目标：sum(gamma^i * r_i) + gamma^n * Q(s_{t+n}, a_{t+n})
			multistep_target = 0
			for i in range(self.action_chunk_size):
				multistep_target += (self.discount ** i) * multi_step_reward[:, i:i+1]
			
			# 添加n步后的Q值
			multistep_target += not_done * (self.discount ** self.action_chunk_size) * target_Q
			
			return multistep_target


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer - 现在返回action chunks和多步rewards
		state, action_chunk, next_state, multi_step_reward, not_done = replay_buffer.sample(batch_size)

		# 计算多步TD目标
		target_Q = self.compute_multistep_target(next_state, multi_step_reward, not_done)

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action_chunk)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# 初始化返回值
		actor_loss_value = None

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss - 使用当前状态生成的action chunk
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			actor_loss_value = actor_loss.item()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		# 返回损失值用于记录
		return {
			'critic_loss': critic_loss.item(),
			'actor_loss': actor_loss_value,
			'q1_value': current_Q1.mean().item(),
			'q2_value': current_Q2.mean().item(),
			'target_q': target_Q.mean().item()
		}


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		