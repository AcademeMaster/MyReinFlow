# MyReinFlow

一个包含 ReFlow/MeanFlow 流模型与行为克隆 (Behavioral Cloning, BC) 的研究与实验仓库，使用 PyTorch 实现，支持 WandB 实验追踪。本项目提供统一的CLI界面，便于管理不同的训练任务。

## 🚀 快速开始

### 统一CLI使用方式

```bash
# 训练行为克隆模型
python myreinflow_cli.py bc --dataset mujoco/pusher/expert-v0 --epochs 50

# 训练FQL模型  
python myreinflow_cli.py fql --dataset mujoco/pusher/expert-v0 --epochs 20

# 运行2D MeanFlow示例
python myreinflow_cli.py mean-flow-2d --epochs 10000

# 显示帮助
python myreinflow_cli.py --help
python myreinflow_cli.py bc --help
```

### 直接运行脚本

```bash
# 直接运行训练脚本
python scripts/train_behavioral_cloning.py --epochs 50
python scripts/train_fql.py --epochs 20
python examples/mean_flow_2d_example.py
```

## 📂 项目结构

```
MyReinFlow/
├── 📜 myreinflow_cli.py          # 统一CLI入口
├── 📜 pyproject.toml             # 项目配置与依赖
├── 📜 README.md                  # 项目文档
│
├── 📂 reflow/                    # 核心流模型实现
│   ├── __init__.py
│   ├── reflow.py                 # ReFlow实现
│   ├── meanflow.py               # MeanFlow实现  
│   ├── mlp_flow.py               # MLP流网络
│   ├── 📂 common/                # 通用组件
│   │   ├── critic.py             # Critic网络
│   │   ├── gaussian.py           # 高斯分布工具
│   │   ├── mlp.py                # MLP网络
│   │   └── ...                   # 其他通用模块
│   └── 📂 ft_baselines/          # 微调基线方法
│       ├── fql.py                # FQL实现
│       └── utils.py              # 工具函数
│
├── 📂 scripts/                   # 训练脚本
│   ├── train_behavioral_cloning.py  # 行为克隆训练
│   └── train_fql.py              # FQL训练
│
├── 📂 examples/                  # 示例与演示
│   ├── mean_flow_2d_example.py   # 2D MeanFlow可视化示例
│   ├── flow.py                   # 流模型基础示例
│   └── main.py                   # 其他示例
│
├── 📂 configs/                   # 配置文件
│   └── default.yaml              # 默认配置
│
├── 📂 experiments/               # 实验输出
│   └── (训练产生的模型文件)
│
└── 📂 wandb/                     # WandB日志 (自动生成)
```

## 目录

- [环境与依赖](#环境与依赖)
- [行为克隆 (Behavioral Cloning)](#行为克隆-behavioral-cloning)
- [FQL训练](#fql训练)
- [MeanFlow vs ReFlow](#meanflow-vs-reflow)
- [配置管理](#配置管理)

## 环境与依赖

建议使用 Conda 环境：

```bash
conda activate your_env_name
```

如果需要 WandB：

```bash
conda install wandb -y
# 首次使用需登录
wandb login
```

其他 Python 依赖请参考 `pyproject.toml`。如需本地安装本项目（可选）：

```bash
pip install -e .
```

## 行为克隆 (Behavioral Cloning)

一个使用 PyTorch 实现的行为克隆训练器，支持连续与离散动作，支持 WandB 记录。已修复原始版本中的 tensor 维度不匹配问题，并改进了数据处理与评估逻辑。

### 主要特性

- ✅ 中文注释与结构化代码：使用类/函数模块化，易读易扩展
- ✅ 集成 WandB（可选）：训练与评估指标可视化
- ✅ 灵活配置：命令行参数与配置类
- ✅ 本地数据集：默认使用 `mujoco/pusher/expert-v0`（Minari）
- ✅ 连续/离散动作空间均支持
- ✅ 评估流程完善：训练过程定期评估

### 使用方法

```bash
# 使用CLI（推荐）
python myreinflow_cli.py bc --dataset mujoco/pusher/expert-v0 --epochs 50

# 直接运行脚本
python scripts/train_behavioral_cloning.py --epochs 50

# 禁用 WandB
python myreinflow_cli.py bc --no_wandb --epochs 10
```

### 命令行参数

- `--dataset`: 数据集名称 (默认: mujoco/pusher/expert-v0)
- `--epochs`: 训练轮数 (默认: 50)
- `--batch_size`: 批大小 (默认: 256)
- `--lr`: 学习率 (默认: 3e-4)
- `--eval_freq`: 评估频率 (默认: 5)
- `--seed`: 随机种子 (默认: 42)
- `--no_wandb`: 禁用 WandB 记录

## FQL训练

Flow Q-Learning (FQL) 结合了流模型与强化学习的方法。

### 使用方法

```bash
# 使用CLI（推荐）
python myreinflow_cli.py fql --dataset mujoco/pusher/expert-v0 --epochs 20

# 直接运行脚本  
python scripts/train_fql.py --epochs 20

# 仅优化行为克隆流模型
python myreinflow_cli.py fql --only_optimize_bc_flow
```

### 主要参数

- `--dataset`: 数据集名称
- `--epochs`: 训练轮数 (默认: 20)
- `--cond_steps`: 条件观测步数 (默认: 1)
- `--horizon_steps`: 预测动作步数 (默认: 4)
- `--lr_flow/--lr_actor/--lr_critic`: 各组件学习率
- `--only_optimize_bc_flow`: 仅优化行为克隆流模型

## MeanFlow vs ReFlow

本项目实现了两种流模型：

1. **ReFlow**：基于瞬时速度场的传统流模型
2. **MeanFlow**：基于平均速度场的改进模型，支持一步生成

### 核心区别

**ReFlow（瞬时速度场）**

- 学习瞬时速度场 `v(x_t, t)`
- 训练目标：`v(x_t, t) = x_1 - x_0`
- 多步采样：`x_{t+dt} = x_t + v(x_t, t) * dt`

**MeanFlow（平均速度场）**

- 学习平均速度场 `u(x_t, t, r)`，其中 `t >= r`
- 训练目标：`u(x_t, t, r) = v(x_t, t) - (t-r) * ∂u/∂t`
- 一步生成：`x_0 = x_1 - u(x_1, 1, 0)`

### 2D可视化示例

```bash
# 运行2D MeanFlow示例
python myreinflow_cli.py mean-flow-2d --epochs 10000

# 或直接运行
python examples/mean_flow_2d_example.py
```

## 配置管理

项目使用YAML配置文件统一管理默认参数，位于 `configs/default.yaml`。你可以：

1. 修改默认配置
2. 创建自定义配置文件
3. 通过命令行参数覆盖配置

## WandB集成

项目集成了WandB用于实验追踪，记录内容包括：

- `train/loss`: 训练损失
- `eval/mean_return`: 平均回报  
- `eval/std_return`: 回报标准差
- `eval/mean_length`: 平均轨迹长度

## 故障排除

### 内存不足

```bash
python myreinflow_cli.py bc --batch_size 64
```

### WandB相关问题

```bash
python myreinflow_cli.py bc --no_wandb
```

### 数据集未就绪

确认 Minari 已安装并且本地存在对应数据集：

```python
import minari
print(minari.list_local_datasets())
```

## 输出文件

- 训练模型：保存在 `experiments/` 目录
- WandB日志：自动保存在 `wandb/` 目录  
- 可视化图片：保存在 `figures/` 目录（2D示例）

## 参考

- ReFlow：Rectified Flow 相关理论与方法
- MeanFlow：平均速度场思想，与最优传输理论相关的改进方向

如需帮助或反馈，欢迎提交 Issue/PR。

### 训练过程（伪代码片段）

ReFlow 训练：

```python
def generate_target(self, x1):
	t = self.sample_time(batch_size=x1.shape[0])
	x0 = torch.randn(x1.shape, device=self.device)
	xt = self.generate_trajectory(x1, x0, t)  # xt = t*x1 + (1-t)*x0
	v = x1 - x0  # 瞬时速度
	return (xt, t), v

def loss(self, xt, t, obs, v):
	v_hat = self.network(xt, t, obs)
	return F.mse_loss(input=v_hat, target=v)
```

MeanFlow 训练：

```python
def generate_target_mean(self, x1):
	t, r = self.sample_time_pair(batch_size)  # t >= r
	x0 = torch.randn(x1.shape, device=self.device)
	zt = (1 - t) * x1 + t * x0
	v = x0 - x1  # 瞬时速度
	return (zt, t, r), v

def loss_mean(self, zt, t, r, obs, v):
	# 使用 JVP 计算速度场的时间导数
	u, dudt = jvp(
		func=self.network,
		inputs=(zt, r, t),
		v=(v, torch.zeros_like(r), torch.ones_like(t))
	)
	# 平均速度目标
	u_tgt = v - (t - r) * dudt
	predicted_velocity = self.network(zt, t, obs)
	return F.mse_loss(predicted_velocity, u_tgt)
```

### 采样过程

ReFlow（多步）：

```python
def sample(self, cond, inference_steps):
	x_hat = torch.randn(data_shape, device=self.device)
	dt = 1.0 / inference_steps
	for i in range(inference_steps):
		t = torch.linspace(0, 1, inference_steps)[i]
		vt = self.network(x_hat, t, cond)
		x_hat += vt * dt  # 前向积分
	return x_hat
```

MeanFlow（一步或少步）：

```python
def sample_one_step(self, cond):
	x_hat = torch.randn(data_shape, device=self.device)
	t = torch.ones(batch_size, device=self.device)
	velocity = self.network(x_hat, t, cond)
	x_hat = x_hat - velocity  # 一步变换
	return x_hat

def sample_mean_flow(self, cond, inference_steps):
	x_hat = torch.randn(data_shape, device=self.device)
	dt = 1.0 / inference_steps
	for i in range(inference_steps, 0, -1):
		r = (i-1) * dt
		t = i * dt
		velocity = self.network(x_hat, t, cond)
		x_hat = x_hat - velocity * dt  # 反向积分
	return x_hat
```

### 实际应用优势

ReFlow 优势
- 理论简单，训练稳定
- 适合追求高精度的场景
- 可通过增加采样步数提升质量

MeanFlow 优势
- 一步生成，推理速度快
- 少步采样即可达到较好效果
- 计算高效，适合实时/资源受限应用

### 使用建议

选择 ReFlow 当：
- 对生成质量要求极高，且不敏感于推理速度
- 需要渐进式生成过程

选择 MeanFlow 当：
- 需要实时生成（如在线强化学习）
- 计算资源有限，或需要低延迟

### 代码使用示例

ReFlow：

```python
reflow = ReFlow(network, device, ...)
(xt, t), v = reflow.generate_target(x1)
loss = reflow.loss(xt, t, obs, v)
sample = reflow.sample(cond, inference_steps=50)
```

MeanFlow：

```python
meanflow = MeanFlow(network, device, ...)
(zt, t, r), v = meanflow.generate_target_mean(x1)
loss = meanflow.loss_mean(zt, t, r, obs, v)
result = meanflow.sample_one_step(cond)
sample = meanflow.sample_mean_flow(cond, inference_steps=4)
```

运行完整示例：

```powershell
python mean_flow_example.py
```

> 参见仓库中的 `reflow/` 目录（包含 `meanflow.py`, `reflow.py`, `mlp_flow.py` 等）以获取实现细节。

---

## 项目结构速览

关键文件/目录：

- `behavioral_cloning.py`：行为克隆主脚本。
- `mean_flow_example.py`：MeanFlow 示例脚本。
- `reflow/`：核心流模型实现（MeanFlow、ReFlow、网络结构与公共模块）。
- `pyproject.toml`：项目依赖与打包配置。
- `wandb/`：若启用 WandB，运行日志与工件将出现在此处。

---

## 参考

- ReFlow：Rectified Flow 相关理论与方法。
- MeanFlow：平均速度场思想，与最优传输理论相关的改进方向。

如需帮助或反馈，欢迎提交 Issue/PR。
