# Offline Reinforcement Learning with MeanFlow

基于MeanFlow的离线强化学习实现，使用PyTorch Lightning框架。

## 项目结构

```
offlineflowrl_light/
├── config.py              # 配置参数定义
├── dataset.py             # 数据集处理模块
├── main.py                # 主程序入口
├── meanflow_ql.py         # MeanFlow Q-Learning模型实现
├── eval_online.py         # 独立在线评估脚本
└── README.md              # 项目说明文档
```

## 功能特性

- 使用MeanFlow算法进行动作序列生成
- 结合CQL (Conservative Q-Learning) 进行离线强化学习
- 支持历史观测序列输入
- 使用PyTorch Lightning进行训练管理

## 安装依赖

确保已安装以下依赖：

```bash
pip install torch torchvision lightning minari numpy tqdm
```

## 训练模型

### 基本训练命令

```bash
python main.py train --dataset mujoco/pusher/expert-v0 --epochs 100 --batch-size 4096
```

### 带混合精度的训练

```bash
python main.py train --dataset mujoco/pusher/expert-v0 --epochs 100 --batch-size 4096 --mixed-precision 16-mixed
```

### 自定义参数训练

```bash
python main.py train --dataset mujoco/pusher/expert-v0 --epochs 50 --batch-size 2048 --learning-rate 1e-4
```

## 测试模型

### 离线测试（动作预测MSE）

```bash
python main.py test --dataset mujoco/pusher/expert-v0
```

### 在线评估（环境奖励）

在线评估可以通过两种方式运行：

1. 使用主脚本进行在线评估（可能会遇到一些问题）：
```bash
python main.py test --dataset mujoco/pusher/expert-v0 --checkpoint path/to/checkpoint.ckpt
```

2. 使用独立的评估脚本（推荐）：
```bash
python eval_online.py  --dataset mujoco/pusher/expert-v0 --checkpoint path/to/checkpoint.ckpt
```

### 跳过在线评估

如果只想运行离线测试，可以使用以下命令跳过在线评估：

```bash
python main.py test --dataset mujoco/pusher/expert-v0 --checkpoint path/to/checkpoint.ckpt --skip-online-eval
```

## 配置参数

主要配置参数可以在 `config.py` 中找到：

- `num_epochs`: 训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `obs_horizon`: 观测序列长度
- `pred_horizon`: 动作预测序列长度
- `inference_steps`: 推理步数
- `cql_alpha`: CQL正则化系数
- `gamma`: 折扣因子

## 模型架构

```bash
tensorboard --logdir .
```


### Actor (MeanFlow)

使用基于流匹配的扩散模型生成动作序列。

### Critic (Double Q-network)

双Q网络估计状态-动作值函数，结合CQL正则化防止过估计。

## 算法细节

### MeanFlow

实现基于流匹配的动作生成，通过雅可比向量积(JVP)计算速度场。

### CQL

实现保守Q学习，通过正则化项防止对未见过的状态-动作对过度估计Q值。

## 性能优化建议

1. 使用混合精度训练 (`--mixed-precision 16-mixed`) 提高训练速度
2. 根据GPU显存调整批次大小，8GB显存推荐使用4096
3. 启用多数据加载工作进程 (`num_workers` > 0) 提高数据加载效率（注意Windows兼容性）

## 注意事项

1. 训练过程中会自动保存最佳模型检查点
2. 支持早停机制，验证损失连续5轮不改善时停止训练
3. 推理时模型内部维护观测历史，无需外部管理
4. 在线评估推荐使用独立的评估脚本以避免潜在问题