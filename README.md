# radarODE-MTL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`radarODE-MTL` 是一个基于 [PyTorch](https://pytorch.org/) 和多任务学习框架 [LibMTL](https://github.com/median-research-group/LibMTL) 构建的开源仓库，用于从毫米波雷达信号无接触重建 ECG（心电图）。

## 目录

- [论文](#论文)
- [项目特点](#项目特点)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [数据集准备](#数据集准备)
- [模型结构](#模型结构)
- [配置说明](#配置说明)
- [训练与测试](#训练与测试)
- [项目结构](#项目结构)
- [引用](#引用)

## 论文

本仓库包含以下论文的官方实现：

1. **radarODE**: [An ODE-embedded deep learning model for contactless ECG reconstruction from millimeter-wave radar](https://arxiv.org/abs/2408.01672) - IEEE Transactions on Mobile Computing, 2025

2. **radarODE-MTL**: [A Multi-Task Learning Framework with Eccentric Gradient Alignment for Robust Radar-Based ECG Reconstruction](https://arxiv.org/abs/2410.08656) - IEEE Transactions on Instrumentation and Measurement, 2025

## 项目特点

- **多任务学习**: 同时学习 ECG 形状重建、PPI（峰峰间隔）估计和 Anchor 定位
- **ODE 嵌入**: 使用常微分方程解码器提供鲁棒的 ECG 参考
- **多种加权策略**: 支持 EGA、DWA、GradNorm、MGDA 等 19 种 MTL 优化策略
- **配置驱动**: 通过 YAML 配置文件控制所有训练参数，无需命令行参数

## 环境配置

### 系统要求

- Python >= 3.8
- CUDA >= 11.8（推荐使用 GPU）

### 安装步骤

推荐使用 [uv](https://docs.astral.sh/uv/) 进行环境管理，速度更快：

```bash
# 1. 克隆仓库
git clone https://github.com/ZYY0844/radarODE-MTL.git
cd radarODE-MTL

# 2. 使用 uv 创建虚拟环境
uv venv

# 3. 激活虚拟环境
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 4. 安装基础依赖（不包含 torch）
uv pip install -r requirements-base.txt

# 5. 单独安装 PyTorch（根据您的 CUDA 版本选择）
# CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
# CPU 版本
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 使用 pip（备选方案）

如果您更习惯使用 pip：

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装基础依赖
pip install -r requirements-base.txt

# 安装 PyTorch（CUDA 12.1）
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 主要依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | >= 2.2 | 深度学习框架（需单独安装） |
| numpy | >= 1.24 | 数值计算 |
| scipy | >= 1.10 | 科学计算 |
| tqdm | >= 4.66 | 进度条显示 |
| torchinfo | >= 1.8 | 模型信息查看 |
| neurokit2 | >= 0.2.7 | ECG 信号处理 |
| ssqueezepy | >= 0.6.5 | 时频分析 |
| mat73 | >= 0.65 | MATLAB 数据读取 |
| pyyaml | >= 6.0 | 配置文件解析 |

## 快速开始

### 1. 准备数据集

从 [MMECG Dataset](https://github.com/jinbochen0823/RCG2ECG) 下载数据集，放置于 `./Dataset` 目录。

### 2. 修改配置文件

编辑 `config.yaml` 文件，设置数据路径和训练参数：

```yaml
general:
  mode: train          # 运行模式: train | test | cross_vali
  gpu_id: '0'          # GPU 编号

data:
  dataset_path: ./Dataset  # 数据集路径
  train_bs: 22             # batch size

training:
  epochs: 200              # 训练轮数

weighting:
  method: EGA              # 加权策略
```

### 3. 运行训练

```bash
cd Projects/radarODE_plus
python main.py
```

## 数据集准备

### 数据集结构

```
Dataset/
├── obj1_NB_1_/
│   ├── sst_seg_0.npy      # 频谱图 (SST)
│   ├── ecg_seg_0.npy      # 真实 ECG
│   ├── anchor_seg_0.npy   # R 峰位置
│   ├── sst_seg_1.npy
│   ├── ecg_seg_1.npy
│   └── anchor_seg_1.npy
│   └── ...
├── obj2_NB_2_/
│   └── ...
└── obj30_PE_91_/
    └── ...
```

### 数据格式

| 文件 | 形状 | 说明 |
|------|------|------|
| `sst_seg_*.npy` | (50, 71, 120) | 频谱图，50 通道 × 71 频率 × 120 时间 |
| `ecg_seg_*.npy` | (N,) | 原始 ECG 信号 |
| `anchor_seg_*.npy` | (M,) | R 峰位置索引 |

### 使用原始数据

如果您有原始雷达和 ECG 数据，可以使用预处理脚本：

```bash
python tools/prepare_radarode_dataset.py --input_raw_path /path/to/raw/data --output_path ./Dataset
```

详细说明请参考 [REPRODUCE.md](REPRODUCE.md)。

## 模型结构

### 整体架构

```
输入: SST 频谱图 (B, 50, 71, 120)
           │
           ▼
    ┌──────────────┐
    │   Backbone   │  DCN-ResNet + Squeeze Module
    │  (共享编码器)  │
    └──────┬───────┘
           │
           ▼
    共享特征 (B, 1024, 31)
           │
    ┌──────┼──────┬──────────┐
    │      │      │          │
    ▼      ▼      ▼          ▼
┌─────┐ ┌─────┐ ┌─────┐
│ECG  │ │ PPI │ │Anchor│
│解码器│ │解码器│ │解码器 │
└──┬──┘ └──┬──┘ └──┬──┘
   │       │       │
   ▼       ▼       ▼
ECG形状  PPI预测  Anchor预测
(B,1,200) (B,260)  (B,200)
```

### 三大任务

| 任务 | 输出 | 说明 |
|------|------|------|
| **ECG_shape** | (B, 1, 200) | 降采样后的 ECG 形状重建 |
| **PPI** | (B, 260) | 峰峰间隔（心跳周期）估计 |
| **Anchor** | (B, 200) | R 峰位置预测 |

### 核心组件

- **Backbone**: 可变形卷积 ResNet (DCN-ResNet)，提取空间特征
- **Squeeze Module**: 通道压缩，减少计算量
- **LSTM-CNN Encoder**: 时序特征编码
- **CNN-LSTM Decoder**: ECG 信号解码（可选 ODE 求解器）

## 配置说明

配置文件 `config.yaml` 采用 YAML 格式，分为以下几个部分：

### 通用配置

```yaml
general:
  mode: train          # 运行模式: train | test | cross_vali
  seed: 777            # 随机种子
  gpu_id: '0'          # GPU 编号，CPU 设为 'cpu'
  save_path: ./Model_saved  # 模型保存路径
  load_path: null      # 加载检查点路径
  save_name: EGA       # 检查点名称
```

### 数据配置

```yaml
data:
  dataset_path: ./Dataset
  train_bs: 22         # 训练 batch size
  test_bs: 22          # 测试 batch size
  num_workers: 8       # DataLoader 工作进程
  aug_snr: 100         # 数据增强 SNR (100=无增强)
  test_ids: '75,76,77,78,79,80,81,82,83,84,85'  # 测试集 ID
```

### 加权策略

```yaml
weighting:
  method: EGA          # 策略选择
  # 可选: EW, UW, GradNorm, GLS, RLW, MGDA, IMTL,
  #       PCGrad, GradVac, CAGrad, GradDrop, DWA,
  #       Nash_MTL, MoCo, Aligned_MTL, DB_MTL,
  #       Given_weight, EGA, STCH
```

### 优化器与调度器

```yaml
optimizer:
  optim: sgd           # adam | sgd | adagrad | rmsprop
  lr: 5.0e-3           # 学习率
  weight_decay: 5.0e-4
  momentum: 0.937      # 仅 SGD

scheduler:
  method: cos          # step | cos | exp | null
  eta_min: 5.0e-5      # cos 调度器最小学习率
  T_max: 100           # cos 调度器周期
```

## 训练与测试

### 训练模式

```bash
# 确保 config.yaml 中 mode: train
cd Projects/radarODE_plus
python main.py
```

训练过程中会保存：
- `best_{name}.pt`: 最佳模型权重
- `cur_{name}.pt`: 当前模型权重
- `best_{name}_train.npy`: 训练损失记录
- `best_{name}_test.npy`: 测试损失记录

### 测试模式

```yaml
# config.yaml
general:
  mode: test
  load_path: ./Model_saved/best_EGA.pt
```

### 交叉验证

```yaml
# config.yaml
general:
  mode: cross_vali
```

### 使用预训练模型

```yaml
general:
  mode: train
  load_path: ./Model_saved/best_EGA.pt  # 加载预训练权重继续训练
```

## 项目结构

```
radarODE-MTL/
├── config.yaml              # 全局配置文件
├── requirements.txt         # 依赖列表
├── README.md               # 说明文档
│
├── LibMTL/                 # MTL 框架核心
│   ├── config.py           # 配置加载
│   ├── trainer.py          # 训练器
│   ├── loss.py             # 损失函数
│   ├── metrics.py          # 评估指标
│   ├── utils.py            # 工具函数
│   ├── weighting/          # 加权策略
│   │   ├── EGA.py         # 本论文提出的策略
│   │   ├── DWA.py
│   │   └── ...
│   └── architecture/       # MTL 架构
│       ├── HPS.py
│       └── ...
│
├── Projects/
│   └── radarODE_plus/      # 主项目
│       ├── main.py         # 入口文件
│       ├── spectrum_dataset.py  # 数据集
│       └── nets/           # 网络模块
│           ├── model.py    # 模型定义
│           ├── encoder.py  # 编码器
│           ├── decoder.py  # 解码器
│           ├── PPI_decoder.py
│           ├── ODE_solver.py
│           └── backbone/
│               └── dcnresnet_backbone.py
│
├── tools/                  # 工具脚本
│   └── prepare_radarode_dataset.py
│
└── image/                  # 图片资源
```

## 常见问题

### Q: 如何更换加权策略？

修改 `config.yaml` 中的 `weighting.method` 字段：

```yaml
weighting:
  method: DWA  # 改为 DWA
  T: 2.0       # DWA 的特定参数
```

### Q: 如何只关注 ECG 重建而不考虑噪声鲁棒性？

使用 `Given_weight` 策略：

```yaml
weighting:
  method: Given_weight
```

### Q: 内存不足怎么办？

减小 batch size 或使用梯度累积：

```yaml
data:
  train_bs: 8  # 减小 batch size
```

### Q: 如何使用 CPU 训练？

```yaml
general:
  gpu_id: 'cpu'
```

## 相关工作

我们最近的工作 [CFT-RFcardi](https://github.com/ZYY0844/CFT-RFcardi) 提供了更易用的框架和预处理数据集，建议查看以快速实现和验证。

## 引用

如果您使用本仓库的代码，请引用我们的论文：

```bibtex
@article{zhang2024radarODE,
  title={{radarODE: An ODE-embedded deep learning model for contactless ECG reconstruction from millimeter-wave radar}},
  author={Yuanyuan Zhang and Runwei Guan and Lingxiao Li and Rui Yang and Yutao Yue and Eng Gee Lim},
  journal={IEEE Transactions on Mobile Computing},
  year={2025},
  publisher={IEEE},
  month={Apr.}
}

@article{zhang2024radarODE-MTL,
  title={{radarODE-MTL}: A Multi-task learning framework with eccentric gradient alignment for robust radar-based {ECG} reconstruction},
  author={Yuanyuan Zhang and Rui Yang and Yutao Yue and Eng Gee Lim},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2025},
  publisher={IEEE},
  month={Apr.}
}
```

## 联系方式

如有任何问题，请：
- 在 [Issues](https://github.com/ZYY0844/radarODE-MTL/issues) 中提问
- 发送邮件至 yz52357@uga.edu

## 许可证

本项目采用 [MIT License](./LICENSE) 开源协议。