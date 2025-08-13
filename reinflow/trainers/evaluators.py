import torch
import numpy as np
import gymnasium as gym
import time
import wandb


def evaluate_reflow(model, env, device, inference_steps=20, num_episodes=10, 
                  render=False, record_video=False, wandb_log=True):
    """评估ReFlow模型
    
    参数:
        model (nn.Module): ReFlow模型
        env (gym.Env): 评估环境
        device (torch.device): 计算设备
        inference_steps (int): 推理步数
        num_episodes (int): 评估轮次
        render (bool): 是否渲染
        record_video (bool): 是否录制视频
        wandb_log (bool): 是否使用WandB记录
        
    返回:
        float: 平均奖励
    """
    model.eval()
    episode_rewards = []
    
    # 视频录制设置
    if record_video:
        video_frames = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            # 归一化观测（如果模型使用归一化）
            if hasattr(model, 'dataset') and hasattr(model.dataset, 'normalize_obs'):
                norm_obs = model.dataset.normalize_obs(obs)
            else:
                norm_obs = obs
            
            # 准备条件
            obs_tensor = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0).to(device)
            cond = {'state': obs_tensor}
            
            # 采样动作
            with torch.no_grad():
                sample = model.sample(cond, inference_steps=inference_steps, 
                                      record_intermediate=False, 
                                      clip_intermediate_actions=True)
                action_seq = sample.trajectories
            
            # 获取第一个动作
            action = action_seq[0, 0].cpu().numpy()
            
            # 反归一化动作（如果模型使用归一化）
            if hasattr(model, 'dataset') and hasattr(model.dataset, 'denormalize_act'):
                action = model.dataset.denormalize_act(action)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 更新状态和奖励
            obs = next_obs
            episode_reward += reward
            step_count += 1
            
            # 渲染
            if render:
                env.render()
                
            # 录制视频
            if record_video:
                frame = env.render()
                video_frames.append(frame)
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}, Steps = {step_count}")
    
    # 计算平均奖励
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"Evaluation completed: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 记录到WandB
    if wandb_log:
        wandb.log({
            'eval/reflow_mean_reward': mean_reward,
            'eval/reflow_std_reward': std_reward,
            'eval/reflow_inference_steps': inference_steps,
        })
        
        # 上传视频
        if record_video and len(video_frames) > 0:
            wandb.log({"eval/reflow_video": wandb.Video(np.array(video_frames), fps=30)})
    
    return mean_reward


def evaluate_mean_flow(model, env, device, inference_steps=1, num_episodes=10, 
                     render=False, record_video=False, wandb_log=True):
    """评估MeanFlow模型
    
    参数:
        model (nn.Module): MeanFlow模型
        env (gym.Env): 评估环境
        device (torch.device): 计算设备
        inference_steps (int): 推理步数
        num_episodes (int): 评估轮次
        render (bool): 是否渲染
        record_video (bool): 是否录制视频
        wandb_log (bool): 是否使用WandB记录
        
    返回:
        float: 平均奖励
    """
    model.eval()
    episode_rewards = []
    
    # 视频录制设置
    if record_video:
        video_frames = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            # 归一化观测（如果模型使用归一化）
            if hasattr(model, 'dataset') and hasattr(model.dataset, 'normalize_obs'):
                norm_obs = model.dataset.normalize_obs(obs)
            else:
                norm_obs = obs
            
            # 准备条件
            obs_tensor = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0).to(device)
            cond = {'state': obs_tensor}
            
            # 采样动作
            with torch.no_grad():
                sample = model.sample(cond, inference_steps=inference_steps, 
                                      record_intermediate=False, 
                                      clip_intermediate_actions=True)
                action_seq = sample.trajectories
            
            # 获取第一个动作
            action = action_seq[0, 0].cpu().numpy()
            
            # 反归一化动作（如果模型使用归一化）
            if hasattr(model, 'dataset') and hasattr(model.dataset, 'denormalize_act'):
                action = model.dataset.denormalize_act(action)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 更新状态和奖励
            obs = next_obs
            episode_reward += reward
            step_count += 1
            
            # 渲染
            if render:
                env.render()
                
            # 录制视频
            if record_video:
                frame = env.render()
                video_frames.append(frame)
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}, Steps = {step_count}")
    
    # 计算平均奖励
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"Evaluation completed: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 记录到WandB
    if wandb_log:
        wandb.log({
            'eval/meanflow_mean_reward': mean_reward,
            'eval/meanflow_std_reward': std_reward,
            'eval/meanflow_inference_steps': inference_steps,
        })
        
        # 上传视频
        if record_video and len(video_frames) > 0:
            wandb.log({"eval/meanflow_video": wandb.Video(np.array(video_frames), fps=30)})
    
    return mean_reward


def compare_methods(reflow_model, meanflow_model, env, device, inference_steps_list=[1, 5, 10, 20, 50], 
                    num_episodes=10, wandb_log=True):
    """比较ReFlow和MeanFlow方法
    
    参数:
        reflow_model (nn.Module): ReFlow模型
        meanflow_model (nn.Module): MeanFlow模型
        env (gym.Env): 评估环境
        device (torch.device): 计算设备
        inference_steps_list (list): 推理步数列表
        num_episodes (int): 评估轮次
        wandb_log (bool): 是否使用WandB记录
        
    返回:
        dict: 比较结果
    """
    print("Comparing ReFlow and MeanFlow methods...")
    
    # 结果存储
    results = {
        'inference_steps': inference_steps_list,
        'reflow_rewards': [],
        'meanflow_rewards': [],
        'reflow_times': [],
        'meanflow_times': [],
    }
    
    # 对每个推理步数进行评估
    for steps in inference_steps_list:
        print(f"\nEvaluating with {steps} inference steps:")
        
        # 评估ReFlow
        print("\nEvaluating ReFlow:")
        start_time = time.time()
        reflow_reward = evaluate_reflow(reflow_model, env, device, 
                                       inference_steps=steps, 
                                       num_episodes=num_episodes, 
                                       wandb_log=False)
        reflow_time = time.time() - start_time
        
        # 评估MeanFlow
        print("\nEvaluating MeanFlow:")
        start_time = time.time()
        meanflow_reward = evaluate_mean_flow(meanflow_model, env, device, 
                                           inference_steps=min(steps, 5),  # MeanFlow通常只需要少量步骤
                                           num_episodes=num_episodes, 
                                           wandb_log=False)
        meanflow_time = time.time() - start_time
        
        # 存储结果
        results['reflow_rewards'].append(reflow_reward)
        results['meanflow_rewards'].append(meanflow_reward)
        results['reflow_times'].append(reflow_time)
        results['meanflow_times'].append(meanflow_time)
        
        print(f"\nResults for {steps} inference steps:")
        print(f"ReFlow: Reward = {reflow_reward:.2f}, Time = {reflow_time:.2f}s")
        print(f"MeanFlow: Reward = {meanflow_reward:.2f}, Time = {meanflow_time:.2f}s")
        
        # 记录到WandB
        if wandb_log:
            wandb.log({
                'compare/inference_steps': steps,
                'compare/reflow_reward': reflow_reward,
                'compare/meanflow_reward': meanflow_reward,
                'compare/reflow_time': reflow_time,
                'compare/meanflow_time': meanflow_time,
                'compare/reward_diff': reflow_reward - meanflow_reward,
                'compare/time_diff': reflow_time - meanflow_time,
                'compare/efficiency': (reflow_reward / reflow_time) - (meanflow_reward / meanflow_time),
            })
    
    # 创建性能对比图
    if wandb_log:
        # 性能对比
        performance_data = [[step, rf, mf] for step, rf, mf in 
                           zip(results['inference_steps'], 
                               results['reflow_rewards'], 
                               results['meanflow_rewards'])]
        performance_table = wandb.Table(data=performance_data, 
                                      columns=["Inference Steps", "ReFlow Reward", "MeanFlow Reward"])
        wandb.log({"compare/performance": wandb.plot.line(
            performance_table, "Inference Steps", ["ReFlow Reward", "MeanFlow Reward"],
            title="Performance Comparison")})
        
        # 性能差异
        diff_data = [[step, rf - mf] for step, rf, mf in 
                    zip(results['inference_steps'], 
                        results['reflow_rewards'], 
                        results['meanflow_rewards'])]
        diff_table = wandb.Table(data=diff_data, 
                               columns=["Inference Steps", "Reward Difference (ReFlow - MeanFlow)"])
        wandb.log({"compare/reward_diff": wandb.plot.line(
            diff_table, "Inference Steps", "Reward Difference (ReFlow - MeanFlow)",
            title="Performance Difference")})
        
        # 效率对比
        efficiency_data = [[step, rt, mt] for step, rt, mt in 
                          zip(results['inference_steps'], 
                              results['reflow_times'], 
                              results['meanflow_times'])]
        efficiency_table = wandb.Table(data=efficiency_data, 
                                     columns=["Inference Steps", "ReFlow Time", "MeanFlow Time"])
        wandb.log({"compare/efficiency": wandb.plot.line(
            efficiency_table, "Inference Steps", ["ReFlow Time", "MeanFlow Time"],
            title="Efficiency Comparison")})
        
        # 效率-性能权衡
        tradeoff_data = [[rf/rt, mf/mt, f"ReFlow ({step} steps)", f"MeanFlow ({step} steps)"] 
                        for step, rf, mf, rt, mt in 
                        zip(results['inference_steps'], 
                            results['reflow_rewards'], 
                            results['meanflow_rewards'],
                            results['reflow_times'], 
                            results['meanflow_times'])]
        tradeoff_table = wandb.Table(data=tradeoff_data, 
                                   columns=["Reward/Time", "Reward/Time", "Method", "Method"])
        wandb.log({"compare/tradeoff": wandb.plot.scatter(
            tradeoff_table, x="Reward/Time", y="Reward/Time", title="Efficiency-Performance Tradeoff")})
    
    return results


def create_generation_visualization(reflow_model, meanflow_model, dataset, device, 
                                   num_samples=5, wandb_log=True):
    """创建生成可视化
    
    参数:
        reflow_model (nn.Module): ReFlow模型
        meanflow_model (nn.Module): MeanFlow模型
        dataset (Dataset): 数据集
        device (torch.device): 计算设备
        num_samples (int): 样本数量
        wandb_log (bool): 是否使用WandB记录
        
    返回:
        dict: 可视化结果
    """
    print("Creating generation visualization...")
    
    # 获取样本
    samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
    
    # 结果存储
    results = {
        'samples': [],
        'reflow_multi_step': [],
        'reflow_single_step': [],
        'meanflow': [],
    }
    
    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}")
        
        # 准备条件
        cond = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in sample.items() if k != 'actions'}
        
        # 真实动作
        true_actions = sample['actions'].unsqueeze(0)
        
        # ReFlow多步生成
        with torch.no_grad():
            reflow_multi = reflow_model.sample(cond, inference_steps=20, 
                                             record_intermediate=True, 
                                             clip_intermediate_actions=True)
        
        # ReFlow单步生成
        with torch.no_grad():
            reflow_single = reflow_model.sample(cond, inference_steps=1, 
                                              record_intermediate=False, 
                                              clip_intermediate_actions=True)
        
        # MeanFlow生成
        with torch.no_grad():
            meanflow_sample = meanflow_model.sample(cond, inference_steps=1, 
                                                 record_intermediate=False, 
                                                 clip_intermediate_actions=True)
        
        # 存储结果
        results['samples'].append(true_actions.cpu().numpy())
        results['reflow_multi_step'].append(reflow_multi.trajectories.cpu().numpy())
        results['reflow_single_step'].append(reflow_single.trajectories.cpu().numpy())
        results['meanflow'].append(meanflow_sample.trajectories.cpu().numpy())
        
        # 记录到WandB
        if wandb_log:
            # 创建动作序列的可视化
            for action_idx in range(min(3, true_actions.shape[1])):  # 最多显示3个动作维度
                # 提取动作序列
                true_seq = true_actions[0, :, action_idx].cpu().numpy()
                reflow_multi_seq = reflow_multi.trajectories[0, :, action_idx].cpu().numpy()
                reflow_single_seq = reflow_single.trajectories[0, :, action_idx].cpu().numpy()
                meanflow_seq = meanflow_sample.trajectories[0, :, action_idx].cpu().numpy()
                
                # 创建数据表
                steps = list(range(len(true_seq)))
                action_data = [[step, true, rm, rs, mf] for step, true, rm, rs, mf in 
                              zip(steps, true_seq, reflow_multi_seq, reflow_single_seq, meanflow_seq)]
                action_table = wandb.Table(data=action_data, 
                                         columns=["Step", "True", "ReFlow (20 steps)", 
                                                 "ReFlow (1 step)", "MeanFlow"])
                
                # 记录图表
                wandb.log({f"generation/sample_{i+1}_action_{action_idx+1}": wandb.plot.line(
                    action_table, "Step", ["True", "ReFlow (20 steps)", 
                                         "ReFlow (1 step)", "MeanFlow"],
                    title=f"Sample {i+1}, Action {action_idx+1} Generation")})
    
    return results