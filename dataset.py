import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MinariFlowDataset(Dataset):
    """处理Minari数据集的自定义Dataset类"""
    
    def __init__(self, dataset, config):
        """
        参数:
            dataset: Minari数据集
            config: Config对象
        """
        self.config = config
        self.episodes = []
        self.stats = self._compute_stats(dataset)
        
        # 处理每个episode
        for episode in tqdm(dataset, desc="处理数据集"):
            ep_len = len(episode.actions)
            
            # 跳过太短的序列
            if ep_len < config.pred_horizon + config.obs_horizon:
                continue
                
            # 准备数据
            obs = self._normalize(episode.observations, self.stats["observations"])
            actions = self._normalize(episode.actions, self.stats["actions"])
            
            # 提取子序列
            num_segments = (ep_len - config.obs_horizon - config.pred_horizon + 1)
            
            for i in range(num_segments):
                start_obs = i
                end_obs = start_obs + config.obs_horizon
                start_act = end_obs
                end_act = start_act + config.pred_horizon
                
                # 确保索引不越界
                if end_act <= len(actions):
                    self.episodes.append({
                        "observations": obs[start_obs:end_obs],
                        "actions": actions[start_act:end_act]
                    })
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        item = self.episodes[idx]
        return {
            "observations": torch.as_tensor(item["observations"], dtype=torch.float32),
            "actions": torch.as_tensor(item["actions"], dtype=torch.float32)
        }
    
    def _compute_stats(self, dataset):
        """计算整个数据集的统计量"""
        all_obs = []
        all_actions = []
        
        for episode in dataset:
            all_obs.append(episode.observations)
            all_actions.append(episode.actions)
        
        all_obs = np.concatenate(all_obs, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        
        return {
            "observations": {
                "min": all_obs.min(axis=0),
                "max": all_obs.max(axis=0),
                "mean": all_obs.mean(axis=0),
                "std": all_obs.std(axis=0)
            },
            "actions": {
                "min": all_actions.min(axis=0),
                "max": all_actions.max(axis=0),
                "mean": all_actions.mean(axis=0),
                "std": all_actions.std(axis=0)
            }
        }
    
    def _normalize(self, data, stats):
        """数据归一化"""
        if self.config.normalize_data:
            return (data - stats["mean"]) / (stats["std"] + 1e-8)
        return data
    
    def denormalize(self, data, stats):
        """数据反归一化"""
        if self.config.normalize_data:
            return data * stats["std"] + stats["mean"]
        return data


def collate_fn_fixed(batch):
    """固定长度序列的批处理函数"""
    return {
        "observations": torch.stack([item["observations"] for item in batch]),
        "actions": torch.stack([item["actions"] for item in batch])
    }


if __name__ == "__main__":
    # 测试数据集功能
    print("测试MinariFlowDataset功能...")
    
    # 创建一个模拟的Minari数据集用于测试
    class MockEpisode:
        def __init__(self, length, obs_dim, action_dim):
            self.observations = np.random.randn(length, obs_dim).astype(np.float32)
            self.actions = np.random.randn(length, action_dim).astype(np.float32)
    
    class MockMinariDataset:
        def __init__(self, num_episodes=3, episode_length=50, obs_dim=10, action_dim=3):
            self.episodes = [
                MockEpisode(episode_length, obs_dim, action_dim) 
                for _ in range(num_episodes)
            ]
        
        def __iter__(self):
            return iter(self.episodes)
    
    # 创建配置
    from config import Config
    config = Config()
    config.obs_horizon = 2
    config.pred_horizon = 5
    config.normalize_data = True
    
    # 创建模拟数据集
    mock_dataset = MockMinariDataset(num_episodes=3, episode_length=20, obs_dim=10, action_dim=3)
    
    # 创建MinariFlowDataset
    flow_dataset = MinariFlowDataset(mock_dataset, config)
    
    print(f"数据集大小: {len(flow_dataset)}")
    
    # 测试获取数据
    if len(flow_dataset) > 0:
        sample = flow_dataset[0]
        print(f"观测数据维度: {sample['observations'].shape}")
        print(f"动作数据维度: {sample['actions'].shape}")
        
        # 测试批次处理
        from torch.utils.data import DataLoader
        dataloader = DataLoader(flow_dataset, batch_size=2, collate_fn=collate_fn_fixed)
        batch = next(iter(dataloader))
        print(f"批次观测数据维度: {batch['observations'].shape}")
        print(f"批次动作数据维度: {batch['actions'].shape}")
    
    print("\n数据集测试完成!")