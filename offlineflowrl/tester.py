import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import minari
import gymnasium as gym
from tqdm import tqdm
from gymnasium import spaces
# 创建模型并加载权重（使用检查点中的维度信息）

from meanflow_ql import  ConservativeMeanFQLModel, Config as MeanFlowQLConfig

class FlowModelTester:
    """流匹配模型测试器"""
    
    def __init__(self, config: "Config", checkpoint_path=None, render_mode=None):
        from config import Config  # 避免循环导入
        self.config: Config = config
        
        # 确定检查点路径
        if checkpoint_path is None:
            # 如果没有指定检查点，尝试加载最新
            checkpoints = [f for f in os.listdir(config.checkpoint_dir) if f.startswith("flow_ema_")]
            if not checkpoints:
                raise FileNotFoundError("未找到检查点文件")
            checkpoints.sort(reverse=True)
            checkpoint_path = os.path.join(config.checkpoint_dir, checkpoints[0])
        
        print(f"加载模型权重: {checkpoint_path}")
        # 修复加载问题，添加weights_only=False参数
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 从检查点中获取模型维度信息
        obs_dim_from_ckpt = checkpoint["model_state"]["obs_embed.net.0.weight"].shape[1]
        action_dim_from_ckpt = checkpoint["model_state"]["noise_embed.net.0.weight"].shape[1]
        
        print(f"从检查点加载的维度信息: obs_dim={obs_dim_from_ckpt}, action_dim={action_dim_from_ckpt}")
        
        # 更新配置中的action_dim
        self.config.action_dim = action_dim_from_ckpt
        self.obs_dim_from_ckpt = obs_dim_from_ckpt  # 保存检查点中的观测维度
        self.action_dim_from_ckpt = action_dim_from_ckpt  # 保存检查点中的动作维度
        
        # 加载数据集以恢复环境
        minari_dataset = minari.load_dataset(config.dataset_name)
        
        # 尝试使用指定的渲染模式恢复环境
        self.eval_env = None
        if render_mode is not None and render_mode != "none":
            try:
                self.eval_env = minari_dataset.recover_environment(eval_env=True, render_mode=render_mode)
            except TypeError:
                # 兼容旧版本不支持 render_mode 参数
                try:
                    self.eval_env = minari_dataset.recover_environment(eval_env=True)
                except Exception:
                    self.eval_env = minari_dataset.recover_environment()
                print("警告: 当前Minari版本不支持render_mode参数")
            except Exception as e:
                print(f"使用render_mode恢复环境时出错: {e}")

        # 如果还没有成功创建评估环境，则使用默认方式
        if self.eval_env is None:
            try:
                self.eval_env = minari_dataset.recover_environment(eval_env=True)
            except Exception:
                # 最后的备选方案
                base_env = minari_dataset.recover_environment()
                env_spec = getattr(base_env, 'spec', None)
                env_id = getattr(env_spec, 'id', None) if env_spec else None
                if env_id:
                    render_mode_valid = render_mode if render_mode != "none" else None
                    self.eval_env = gym.make(env_id, render_mode=render_mode_valid)
                else:
                    self.eval_env = base_env
        
        # 获取环境信息并处理 None 的情况
        obs_space = self.eval_env.observation_space
        act_space = self.eval_env.action_space

        obs_shape = obs_space.shape if isinstance(obs_space, spaces.Box) else (1,)
        action_shape = act_space.shape if isinstance(act_space, spaces.Box) else (1,)

        obs_dim_env = int(np.prod(obs_shape)) if obs_shape is not None else 1
        action_dim_env = int(np.prod(action_shape)) if action_shape is not None else 1
        
        print(f"环境中的维度信息: obs_dim={obs_dim_env}, action_dim={action_dim_env}")
        
        # 检查检查点中的配置格式
        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            # 从检查点恢复配置
            saved_config = checkpoint["config"]
            for key, value in saved_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # 创建离线强化学习模型配置
        meanflow_ql_config = MeanFlowQLConfig(
            hidden_dim=self.config.hidden_dim,
            time_dim=self.config.time_dim,
            pred_horizon=self.config.pred_horizon,
            learning_rate=self.config.learning_rate,
            grad_clip_value=self.config.grad_clip_value,
            cql_alpha=self.config.cql_alpha,
            cql_temp=self.config.cql_temp,
            tau=self.config.tau,
            gamma=self.config.gamma,
            inference_steps=self.config.inference_steps,
            normalize_q_loss=self.config.normalize_q_loss,
            device=self.config.device
        )

        self.agent = ConservativeMeanFQLModel(obs_dim_from_ckpt, action_dim_from_ckpt, meanflow_ql_config)
        self.agent.actor.model.to(device)
        self.agent.actor.model.load_state_dict(checkpoint["model_state"])
        self.stats = checkpoint.get("stats", None)
        self.agent.actor.model.eval()
        
        # 保存环境的动作维度
        self.action_dim_env = action_dim_env
        
        # 如果环境维度与检查点维度不一致，禁用归一化
        if obs_dim_env != obs_dim_from_ckpt or action_dim_env != action_dim_from_ckpt:
            print("警告: 环境维度与检查点维度不一致，禁用数据归一化")
            self.config.normalize_data = False
            # 创建一个索引来选择与模型匹配的观测维度
            if obs_dim_env > obs_dim_from_ckpt:
                self.obs_indices = torch.arange(obs_dim_from_ckpt)
                print(f"将从环境观测中选择前{obs_dim_from_ckpt}个维度")
            elif obs_dim_env < obs_dim_from_ckpt:
                # 如果环境观测维度小于模型期望维度，需要进行填充
                self.obs_padding = obs_dim_from_ckpt - obs_dim_env
                print(f"将为观测添加{self.obs_padding}个零填充维度")
                self.obs_indices = None
            else:
                self.obs_indices = None
                print("环境观测维度等于模型期望维度")
                
            # 对于动作维度，如果环境需要更多维度，则填充0
            if action_dim_env > action_dim_from_ckpt:
                self.action_padding = action_dim_env - action_dim_from_ckpt
                print(f"将为动作添加{self.action_padding}个零填充维度")
            elif action_dim_env < action_dim_from_ckpt:
                self.action_truncate = action_dim_from_ckpt - action_dim_env
                print(f"将从动作中截取前{action_dim_env}个维度")
            else:
                self.action_padding = 0
                self.action_truncate = 0
        else:
            self.obs_indices = None
            self.obs_padding = 0
            self.action_padding = 0
            self.action_truncate = 0
    
    def test(self):
        """测试模型在环境中的表现"""
        print("开始测试...")
        total_rewards = []
        
        # 定义设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for episode in range(self.config.test_episodes):
            start_time = time.time()
            obs, _ = self.eval_env.reset()
            obs_history = [obs] * self.config.obs_horizon
            episode_reward = 0.0  # 明确指定为浮点数
            step_count = 0
            done = False
            pbar = tqdm(total=self.config.max_steps, desc=f"Episode {episode}")
            
            while not done and step_count < self.config.max_steps:
                # 准备观测数据
                obs_arr = np.array(obs_history)
                if self.stats and self.config.normalize_data:
                    obs_arr = self._normalize(obs_arr, self.stats["observations"])
                
                # 转换为张量并调整维度以匹配模型期望的维度 [B, obs_horizon, obs_dim]
                obs_tensor = torch.as_tensor(obs_arr, device=device, dtype=torch.float32).unsqueeze(0)  # [1, obs_horizon, obs_dim]
                
                # 如果需要，调整观测数据维度以匹配模型期望的维度
                if self.obs_indices is not None:
                    obs_tensor = torch.index_select(obs_tensor, 2, self.obs_indices.to(device))
                elif hasattr(self, 'obs_padding') and self.obs_padding > 0:
                    # 如果观测维度不足，进行零填充
                    obs_tensor = torch.nn.functional.pad(obs_tensor, (0, self.obs_padding), mode='constant', value=0)
                
                batch = {
                    "observations": obs_tensor
                }
                action_seq = self.agent.actor.predict_action_chunk(batch, n_steps=self.config.inference_steps)
                
                # 反归一化
                if self.stats and self.config.normalize_data:
                    action_seq = self._denormalize(action_seq, self.stats["actions"])
                
                # 执行动作序列
                for i in range(min(self.config.pred_horizon, self.config.action_horizon)):
                    if done:
                        break
                    
                    action = action_seq[0, i]  # 从 [B, pred_horizon, action_dim] 中提取 [action_dim]
                    
                    # 如果需要，调整动作维度以匹配环境期望的维度
                    if hasattr(self, 'action_padding') and self.action_padding > 0:
                        # 填充0以增加动作维度
                        padded_action = np.pad(action, (0, self.action_padding), mode='constant')
                        action = padded_action
                    elif hasattr(self, 'action_truncate') and self.action_truncate > 0:
                        # 截取动作以减少维度
                        action = action[:self.action_dim_env]
                    
                    # 确保动作是CPU上的NumPy数组
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    
                    next_obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += float(reward)  # 确保类型转换
                    step_count += 1
                    
                    # 更新观测历史
                    obs_history.pop(0)
                    obs_history.append(next_obs)
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
            
            pbar.close()
            duration = time.time() - start_time
            fps = step_count / duration
            total_rewards.append(episode_reward)
            print(f"Episode {episode}: reward = {episode_reward:.2f}, steps = {step_count}, "
                  f"duration = {duration:.2f}s, FPS = {fps:.1f}")
        
        # 显示总结统计
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"\n测试完成，平均奖励: {avg_reward:.2f} (±{np.std(total_rewards):.2f})")
    
    def _normalize(self, data, stats):
        """数据归一化"""
        return (data - stats["mean"]) / (stats["std"] + 1e-8) if self.config.normalize_data else data
        
    def _denormalize(self, data, stats):
        """数据反归一化"""
        if self.config.normalize_data:
            # 确保在相同设备上进行操作
            device = data.device
            std = torch.as_tensor(stats["std"], device=device)
            mean = torch.as_tensor(stats["mean"], device=device)
            return data * std + mean
        return data


class OfflineFlowRLTester:
    """离线强化学习模型测试器"""
    
    def __init__(self, config: "Config", checkpoint_path=None, render_mode=None):
        from config import Config  # 避免循环导入
        self.config: Config = config
        
        # 确定检查点路径
        if checkpoint_path is None:
            # 如果没有指定检查点，尝试加载最新
            checkpoints = [f for f in os.listdir(config.checkpoint_dir) if f.startswith("offline_flow_rl_")]
            if not checkpoints:
                raise FileNotFoundError("未找到离线强化学习检查点文件")
            checkpoints.sort(reverse=True)
            checkpoint_path = os.path.join(config.checkpoint_dir, checkpoints[0])
        
        print(f"加载离线强化学习模型权重: {checkpoint_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 从检查点中获取模型维度信息
        obs_dim_from_ckpt = checkpoint["model_state"]["actor.model.obs_embed.net.0.weight"].shape[1]
        action_dim_from_ckpt = checkpoint["model_state"]["actor.model.noise_embed.net.0.weight"].shape[1]
        
        print(f"从检查点加载的维度信息: obs_dim={obs_dim_from_ckpt}, action_dim={action_dim_from_ckpt}")
        
        # 更新配置中的action_dim
        self.config.action_dim = action_dim_from_ckpt
        self.obs_dim_from_ckpt = obs_dim_from_ckpt  # 保存检查点中的观测维度
        self.action_dim_from_ckpt = action_dim_from_ckpt  # 保存检查点中的动作维度
        
        # 加载数据集以恢复环境
        minari_dataset = minari.load_dataset(config.dataset_name)
        
        # 尝试使用指定的渲染模式恢复环境
        self.eval_env = None
        if render_mode is not None and render_mode != "none":
            try:
                self.eval_env = minari_dataset.recover_environment(eval_env=True, render_mode=render_mode)
            except TypeError:
                # 兼容旧版本不支持 render_mode 参数
                try:
                    self.eval_env = minari_dataset.recover_environment(eval_env=True)
                except Exception:
                    self.eval_env = minari_dataset.recover_environment()
                print("警告: 当前Minari版本不支持render_mode参数")
            except Exception as e:
                print(f"使用render_mode恢复环境时出错: {e}")

        # 如果还没有成功创建评估环境，则使用默认方式
        if self.eval_env is None:
            try:
                self.eval_env = minari_dataset.recover_environment(eval_env=True)
            except Exception:
                # 最后的备选方案
                base_env = minari_dataset.recover_environment()
                env_spec = getattr(base_env, 'spec', None)
                env_id = getattr(env_spec, 'id', None) if env_spec else None
                if env_id:
                    render_mode_valid = render_mode if render_mode != "none" else None
                    self.eval_env = gym.make(env_id, render_mode=render_mode_valid)
                else:
                    self.eval_env = base_env
        
        # 获取环境信息并处理 None 的情况
        obs_space = self.eval_env.observation_space
        act_space = self.eval_env.action_space

        obs_shape = obs_space.shape if isinstance(obs_space, spaces.Box) else (1,)
        action_shape = act_space.shape if isinstance(act_space, spaces.Box) else (1,)

        obs_dim_env = int(np.prod(obs_shape)) if obs_shape is not None else 1
        action_dim_env = int(np.prod(action_shape)) if action_shape is not None else 1
        
        print(f"环境中的维度信息: obs_dim={obs_dim_env}, action_dim={action_dim_env}")
        
        # 检查检查点中的配置格式
        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            # 从检查点恢复配置
            saved_config = checkpoint["config"]
            for key, value in saved_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # 创建离线强化学习模型
        meanflow_ql_config = MeanFlowQLConfig(
            hidden_dim=self.config.hidden_dim,
            time_dim=self.config.time_dim,
            pred_horizon=self.config.pred_horizon,
            learning_rate=self.config.learning_rate,
            grad_clip_value=self.config.grad_clip_value,
            cql_alpha=self.config.cql_alpha,
            cql_temp=self.config.cql_temp,
            tau=self.config.tau,
            gamma=self.config.gamma,
            inference_steps=self.config.inference_steps,
            normalize_q_loss=self.config.normalize_q_loss,
            device=self.config.device
        )
        
        self.model = ConservativeMeanFQLModel(obs_dim_from_ckpt, action_dim_from_ckpt, meanflow_ql_config)
        self.model.to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.stats = checkpoint.get("stats", None)
        
        # 保存环境的动作维度
        self.action_dim_env = action_dim_env
        
        # 如果环境维度与检查点维度不一致，处理维度适配
        if obs_dim_env != obs_dim_from_ckpt or action_dim_env != action_dim_from_ckpt:
            print("警告: 环境维度与检查点维度不一致")
            # 创建一个索引来选择与模型匹配的观测维度
            if obs_dim_env > obs_dim_from_ckpt:
                self.obs_indices = torch.arange(obs_dim_from_ckpt)
                print(f"将从环境观测中选择前{obs_dim_from_ckpt}个维度")
            elif obs_dim_env < obs_dim_from_ckpt:
                # 如果环境观测维度小于模型期望维度，需要进行填充
                self.obs_padding = obs_dim_from_ckpt - obs_dim_env
                print(f"将为观测添加{self.obs_padding}个零填充维度")
                self.obs_indices = None
            else:
                self.obs_indices = None
                print("环境观测维度等于模型期望维度")
                
            # 对于动作维度，如果环境需要更多维度，则填充0
            if action_dim_env > action_dim_from_ckpt:
                self.action_padding = action_dim_env - action_dim_from_ckpt
                print(f"将为动作添加{self.action_padding}个零填充维度")
            elif action_dim_env < action_dim_from_ckpt:
                self.action_truncate = action_dim_from_ckpt - action_dim_env
                print(f"将从动作中截取前{action_dim_env}个维度")
            else:
                self.action_padding = 0
                self.action_truncate = 0
        else:
            self.obs_indices = None
            self.obs_padding = 0
            self.action_padding = 0
            self.action_truncate = 0
    
    def test(self):
        """测试离线强化学习模型在环境中的表现"""
        print("开始测试离线强化学习模型...")
        total_rewards = []
        
        # 定义设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for episode in range(self.config.test_episodes):
            start_time = time.time()
            obs, _ = self.eval_env.reset()
            obs_history = [obs] * self.config.obs_horizon
            episode_reward = 0.0  # 明确指定为浮点数
            step_count = 0
            done = False
            pbar = tqdm(total=self.config.max_steps, desc=f"Episode {episode}")
            
            while not done and step_count < self.config.max_steps:
                # 准备观测数据
                obs_arr = np.array(obs_history)
                if self.stats and self.config.normalize_data:
                    obs_arr = self._normalize(obs_arr, self.stats["observations"])
                
                # 转换为张量并调整维度以匹配模型期望的维度 [B, obs_horizon, obs_dim]
                obs_tensor = torch.as_tensor(obs_arr, device=device, dtype=torch.float32).unsqueeze(0)  # [1, obs_horizon, obs_dim]
                
                # 如果需要，调整观测数据维度以匹配模型期望的维度
                if self.obs_indices is not None:
                    obs_tensor = torch.index_select(obs_tensor, 2, self.obs_indices.to(device))
                elif hasattr(self, 'obs_padding') and self.obs_padding > 0:
                    # 如果观测维度不足，进行零填充
                    obs_tensor = torch.nn.functional.pad(obs_tensor, (0, self.obs_padding), mode='constant', value=0)
                
                batch = {
                    "observations": obs_tensor
                }
                
                # 使用离线强化学习模型生成动作
                action_seq = self.model(batch)
                
                # 反归一化
                if self.stats and self.config.normalize_data:
                    action_seq = self._denormalize(action_seq, self.stats["actions"])
                
                # 执行动作序列
                for i in range(min(self.config.pred_horizon, self.config.action_horizon)):
                    if done:
                        break
                    
                    action = action_seq[0, i]  # 从 [B, pred_horizon, action_dim] 中提取 [action_dim]
                    
                    # 如果需要，调整动作维度以匹配环境期望的维度
                    if hasattr(self, 'action_padding') and self.action_padding > 0:
                        # 填充0以增加动作维度
                        padded_action = np.pad(action, (0, self.action_padding), mode='constant')
                        action = padded_action
                    elif hasattr(self, 'action_truncate') and self.action_truncate > 0:
                        # 截取动作以减少维度
                        action = action[:self.action_dim_env]
                    
                    # 确保动作是CPU上的NumPy数组
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    
                    next_obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += float(reward)  # 确保类型转换
                    step_count += 1
                    
                    # 更新观测历史
                    obs_history.pop(0)
                    obs_history.append(next_obs)
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
            
            pbar.close()
            duration = time.time() - start_time
            fps = step_count / duration
            total_rewards.append(episode_reward)
            print(f"Episode {episode}: reward = {episode_reward:.2f}, steps = {step_count}, "
                  f"duration = {duration:.2f}s, FPS = {fps:.1f}")
        
        # 显示总结统计
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"\n离线强化学习模型测试完成，平均奖励: {avg_reward:.2f} (±{np.std(total_rewards):.2f})")
    
    def _normalize(self, data, stats):
        """数据归一化"""
        return (data - stats["mean"]) / (stats["std"] + 1e-8) if self.config.normalize_data else data
        
    def _denormalize(self, data, stats):
        """数据反归一化"""
        if self.config.normalize_data:
            # 确保在相同设备上进行操作
            device = data.device
            std = torch.as_tensor(stats["std"], device=device)
            mean = torch.as_tensor(stats["mean"], device=device)
            return data * std + mean
        return data