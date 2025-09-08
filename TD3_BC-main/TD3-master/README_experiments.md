# TD3 实验系统使用指南

本项目提供了一个改进的TD3实验系统，支持命令行参数配置、配置文件管理和自动化实验管理。

## 功能特性

- 🚀 **命令行参数支持**: 通过命令行灵活配置所有实验参数
- 📁 **配置文件管理**: 支持JSON配置文件的保存和加载
- 📊 **自动实验管理**: 自动创建实验目录、保存结果和模型
- 🔄 **多实验批处理**: 支持批量运行多个实验配置
- 💾 **结果持久化**: 自动保存训练结果、配置和模型

## 快速开始

### 1. 基本使用

```bash
# 使用默认参数运行
python main.py

# 指定环境和训练步数
python main.py --env_name Ant-v5 --max_timesteps 500000

# 设置多步动作序列
python main.py --action_horizon 4 --exp_name multistep_experiment
```

### 2. 使用配置文件

```bash
# 从配置文件运行
python main.py --config example_config.json

# 保存当前配置
python main.py --save_config --exp_name my_experiment
```

### 3. 批量实验

```bash
# 运行预定义的多个实验
python run_experiments.py
```

## 命令行参数详解

### 基本参数
- `--env_name`: 环境名称 (默认: Ant-v5)
- `--seed`: 随机种子 (默认: 0)
- `--max_timesteps`: 最大训练步数 (默认: 200000)
- `--exp_name`: 实验名称 (默认: td3_experiment)

### 网络参数
- `--actor_lr`: Actor学习率 (默认: 3e-4)
- `--critic_lr`: Critic学习率 (默认: 3e-4)
- `--hidden_dim`: 隐藏层维度 (默认: 256)
- `--gamma`: 折扣因子 (默认: 0.99)
- `--tau`: 软更新参数 (默认: 0.005)

### 训练参数
- `--batch_size`: 批次大小 (默认: 256)
- `--buffer_size`: 经验回放缓冲区大小 (默认: 1000000)
- `--start_timesteps`: 随机探索步数 (默认: 1000)
- `--action_horizon`: 动作序列长度 (默认: 4)

### TD3特有参数
- `--policy_noise`: 策略噪声 (默认: 0.5)
- `--noise_clip`: 噪声裁剪 (默认: 1.0)
- `--policy_freq`: 策略更新频率 (默认: 2)
- `--expl_noise`: 探索噪声 (默认: 0.1)

### 评估和保存参数
- `--eval_freq`: 评估频率 (默认: 5000)
- `--save_freq`: 模型保存频率 (默认: 50000)
- `--log_freq`: 日志记录频率 (默认: 1000)
- `--render_eval`: 评估时是否渲染
- `--save_model`: 是否保存模型

## 实验目录结构

每次运行实验都会创建一个带时间戳的目录：

```
experiments/
└── experiment_name_20231201_143022/
    ├── config.json          # 实验配置
    ├── results.json         # 训练结果
    ├── models/
    │   └── final_model.pth  # 最终模型
    └── logs/                # 日志文件
```

## 配置文件示例

```json
{
  "env_name": "Ant-v5",
  "seed": 42,
  "max_timesteps": 500000,
  "exp_name": "ant_multistep_experiment",
  "actor_lr": 0.0003,
  "critic_lr": 0.0003,
  "action_horizon": 4,
  "batch_size": 256,
  "eval_freq": 5000,
  "save_model": true,
  "render_eval": false
}
```

## 实验示例

### 1. 比较不同动作序列长度

```bash
# 单步动作
python main.py --action_horizon 1 --exp_name single_step

# 4步动作序列
python main.py --action_horizon 4 --exp_name multi_step_4

# 8步动作序列
python main.py --action_horizon 8 --exp_name multi_step_8
```

### 2. 超参数调优

```bash
# 不同学习率
python main.py --actor_lr 1e-4 --critic_lr 1e-4 --exp_name low_lr
python main.py --actor_lr 1e-3 --critic_lr 1e-3 --exp_name high_lr

# 不同批次大小
python main.py --batch_size 128 --exp_name small_batch
python main.py --batch_size 512 --exp_name large_batch
```

### 3. 不同环境测试

```bash
# 不同的MuJoCo环境
python main.py --env_name HalfCheetah-v5 --exp_name halfcheetah
python main.py --env_name Walker2d-v5 --exp_name walker2d
python main.py --env_name Hopper-v5 --exp_name hopper
```

## 结果分析

训练完成后，可以通过以下方式分析结果：

1. **查看results.json**: 包含训练奖励、评估分数等数据
2. **加载保存的模型**: 用于进一步测试或部署
3. **比较不同实验**: 通过实验目录对比不同配置的效果

## 注意事项

1. 确保有足够的磁盘空间存储实验结果
2. 长时间训练建议使用`nohup`或`screen`
3. GPU内存不足时可以减小`batch_size`
4. 首次运行建议使用较小的`max_timesteps`进行测试

## 故障排除

- **内存不足**: 减小`buffer_size`或`batch_size`
- **训练不稳定**: 调整学习率或增加`start_timesteps`
- **收敛慢**: 增加`max_timesteps`或调整网络结构