import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), action_chunk_size=1):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.action_chunk_size = action_chunk_size
		self.action_dim = action_dim

		self.state = np.zeros((max_size, state_dim))
		# 存储action chunks: (max_size, action_chunk_size, action_dim)
		self.action_chunk = np.zeros((max_size, action_chunk_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		# 存储多步rewards: (max_size, action_chunk_size)
		self.multi_step_reward = np.zeros((max_size, action_chunk_size))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action_chunk, next_state, multi_step_reward, done):
		"""
		添加经验到buffer
		args:
			state: 当前状态
			action_chunk: action序列，shape: (action_chunk_size, action_dim)
			next_state: action_chunk_size步后的状态
			multi_step_reward: 多步reward序列，shape: (action_chunk_size,)
			done: 是否结束
		"""
		self.state[self.ptr] = state
		self.action_chunk[self.ptr] = action_chunk
		self.next_state[self.ptr] = next_state
		self.multi_step_reward[self.ptr] = multi_step_reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		# 确保采样的索引不会导致多步采样越界
		max_start_idx = max(0, self.size - self.action_chunk_size)
		if max_start_idx == 0:
			# 如果buffer中数据不足，使用现有数据
			ind = np.random.randint(0, self.size, size=batch_size)
		else:
			ind = np.random.randint(0, max_start_idx + 1, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action_chunk[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.multi_step_reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)