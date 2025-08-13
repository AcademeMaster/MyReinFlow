# ReInFlow: 流匹配强化学习实现

这个项目实现了基于流匹配（Flow Matching）的强化学习方法，包括ReFlow和MeanFlow两种算法。

## 项目结构

```
.
├── reinflow/              # 主要代码包
│   ├── models/           # 模型定义
│   │   ├── base.py       # 基础模块
│   │   ├── flow_mlp.py   # FlowMLP网络
│   │   ├── reflow.py     # ReFlow模型
│   │   └── mean_flow.py  # MeanFlow模型
│   ├── data/             # 数据处理
│   │   └── minari_dataset.py  # Minari数据集处理
│   ├── trainers/         # 训练和评估
│   │   ├── trainers.py   # 训练器
│   │   └── evaluators.py # 评估器
│   └── utils/            # 工具函数
│       └── helpers.py    # 辅助函数
├── scripts/              # 脚本
│   ├── train.py          # 训练脚本
│   └── evaluate.py       # 评估脚本
├── main.py               # 主入口
└── README.md             # 项目说明
```

## 安装依赖

```bash
pip install torch numpy gymnasium minari wandb matplotlib
```

## 使用方法

### 训练模型

```bash
python main.py --mode train --dataset hopper-medium-expert-v2 --method both --epochs 100 --batch_size 64 --horizon_steps 10 --inference_steps 20 --noise_schedule prioritized --normalize
```

主要参数说明：
- `--dataset`: Minari数据集名称
- `--method`: 训练方法，可选 'reflow', 'meanflow', 'both'
- `--epochs`: 训练轮次
- `--batch_size`: 批处理大小
- `--horizon_steps`: 动作序列的时间步长
- `--inference_steps`: 推理步数
- `--noise_schedule`: 噪声调度类型，可选 'uniform', 'prioritized', 'cosine'
- `--normalize`: 是否归一化数据

### 评估模型

```bash
python main.py --mode eval --dataset hopper-medium-expert-v2 --method both --inference_steps 20 --num_episodes 10 --compare --visualize
```

主要参数说明：
- `--dataset`: Minari数据集名称
- `--method`: 评估方法，可选 'reflow', 'meanflow', 'both'
- `--inference_steps`: 推理步数
- `--num_episodes`: 评估轮次
- `--compare`: 是否比较两种方法
- `--visualize`: 是否创建生成可视化
- `--render`: 是否渲染评估
- `--record_video`: 是否录制评估视频

## 算法说明

### ReFlow

ReFlow是一种基于流匹配的强化学习方法，它通过学习从噪声到目标动作的连续轨迹来生成动作序列。ReFlow使用多步推理过程，通过求解常微分方程来生成高质量的动作序列。

### MeanFlow

MeanFlow是ReFlow的简化版本，它通过学习平均速度场来直接生成动作序列，通常只需要一步推理。MeanFlow计算效率更高，但在某些情况下生成质量可能不如ReFlow。

## 主要特点

- 模块化设计，易于扩展和修改
- 支持多种噪声调度策略
- 支持多种推理方法（Euler和Runge-Kutta积分）
- 集成WandB记录训练和评估指标
- 提供详细的性能对比和可视化工具

## 参考

- Flow Matching for Generative Modeling
- Diffusion Models for Reinforcement Learning