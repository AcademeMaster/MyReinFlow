# MyReinFlow

基于流匹配(Flow Matching)的强化学习框架，用于机器人操作任务的动作生成。

## 项目结构

```
MyReinFlow/
├── action_head/                 # 动作头模块
│   ├── __init__.py
│   ├── action_encoder.py        # 动作编码器
│   ├── cross_attention_dit.py   # 交叉注意力DiT模型
│   └── flow_matching_action_head.py  # 流匹配动作头
├── checkpoint_t/                # 模型检查点目录
├── config.py                    # 配置管理
├── dataset.py                   # 数据集处理
├── flow_matching.py             # 流匹配模型定义
├── meanflow.py                  # MeanFlow模型定义
├── model.py                     # 核心模型定义
├── trainer.py                   # 训练器
├── tester.py                    # 测试器
├── main.py                      # 程序入口
└── pyproject.toml               # 项目配置
```

## 功能特性

- 基于条件流匹配(Conditional Flow Matching)的动作生成模型
- 支持Diffusion Transformer (DiT) 架构
- 支持MeanFlow算法进行一步生成
- 模块化设计，易于扩展和维护
- 支持多种机器人环境（基于Minari数据集）
- 可配置的训练和测试参数
- 支持大批次训练以提高训练效率

## 技术架构

本项目采用模块化设计，主要包括以下几个核心组件：

1. **配置管理模块** ([config.py](file:///D:/PycharmProjects/MyReinFlow/config.py)) - 管理所有训练和测试相关的配置参数
2. **数据集处理模块** ([dataset.py](file:///D:/PycharmProjects/MyReinFlow/dataset.py)) - 处理Minari数据集，生成训练所需的序列数据
3. **流匹配模型模块** ([model.py](file:///D:/PycharmProjects/MyReinFlow/model.py)) - 定义标准的流匹配模型和代理
4. **MeanFlow模型模块** ([meanflow.py](file:///D:/PycharmProjects/MyReinFlow/meanflow.py)) - 定义MeanFlow模型和代理，支持一步生成
5. **训练器模块** ([trainer.py](file:///D:/PycharmProjects/MyReinFlow/trainer.py)) - 实现模型训练逻辑
6. **测试器模块** ([tester.py](file:///D:/PycharmProjects/MyReinFlow/tester.py)) - 实现模型测试和评估逻辑
7. **动作头模块** ([action_head/](file:///D:/PycharmProjects/MyReinFlow/action_head/)) - 包含高级动作生成组件

## 安装依赖

```bash
pip install -e .
```

或者使用 poetry:

```bash
poetry install
```

## 使用方法

### 训练模型

```bash
# 基本训练命令
python main.py train --dataset mujoco/pusher/expert-v0 --epochs 100

# 使用大批次大小加速训练（适用于有足够显存的情况）
python main.py train --dataset mujoco/pusher/expert-v0 --epochs 100 --batch-size 4096

# 使用混合精度训练（需要支持的GPU）
python main.py train --dataset mujoco/pusher/expert-v0 --epochs 100 --mixed-precision fp16
```

### 测试模型

```bash
# 基本测试命令
python main.py test --dataset mujoco/pusher/expert-v0 --checkpoint ./checkpoint_t/flow_ema_0100.pth

# 不显示渲染界面的测试
python main.py test --dataset mujoco/pusher/expert-v0 --checkpoint ./checkpoint_t/flow_ema_0100.pth --render none

# 指定测试轮数
python main.py test --dataset mujoco/pusher/expert-v0 --checkpoint ./checkpoint_t/flow_ema_0100.pth --test-episodes 20
```

## 参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `mujoco/humanoid/expert-v0` | Minari数据集名称 |
| `--epochs` | `100` | 训练轮数 |
| `--batch-size` | `128` | 批次大小 |
| `--learning-rate` | `1e-4` | 学习率 |
| `--mixed-precision` | `fp16` | 混合精度训练模式 |
| `--gradient-accumulation-steps` | `1` | 梯度累积步数 |

### 测试参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 无 | 测试时使用的模型路径 |
| `--n-steps` | `1` | 采样步数，1表示一步生成，>1表示多步迭代生成 |

## MeanFlow 算法说明

MeanFlow 是一种改进的流匹配算法，它允许通过一步生成来获得最终的动作序列，而不是传统的迭代生成方法。这大大提高了推理速度，同时保持了生成质量。

使用 MeanFlow 时，可以通过设置 `--n-steps 1` 参数来启用一步生成模式。

## 性能优化建议

在具有8GB显存的GPU上，可以将批次大小设置为4096以加快训练速度。对于推理阶段，使用一步生成（n_steps=1）可以获得最佳性能。

## 已知问题

1. 在某些环境下可能需要调整批次大小以适应显存限制
2. 多步生成模式可能会导致推理时间较长

## 未来改进方向

1. 支持更多类型的流匹配算法
2. 增加更多动作头架构选项
3. 提供模型压缩和加速功能
4. 增加可视化工具以更好地理解模型行为
