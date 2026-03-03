# radarODE-MTL 复现指南

本指南提供了一个实用的命令行工作流，用于复现本仓库中的数据集准备和模型训练。

它支持：

1. `rcg` 模式（推荐，与论文一致）：预处理雷达通道（`RCG`）+ 原始 ECG
2. `adc` 模式（实验性）：原始雷达 ADC 立方体 + 原始 ECG

生成的训练文件符合仓库预期：

- `sst_seg_*.npy`，形状为 `(50, 71, 120)`
- `ecg_seg_*.npy`，具有可变的周期长度
- `anchor_seg_*.npy`，形状为 `(200,)`

## 1) 环境设置

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) 准备原始输入文件

每个试验文件应包含 ECG 和雷达数据。

### 选项 A：RCG 模式（推荐）
这与 MMECG 风格的数据匹配，其中雷达已经转换为 50 通道心脏信号。

预期的键：

- `ECG`：1D 波形
- `RCG`：2D 数组 `[时间, 通道]`
- 可选：`id`、`physistatus`

### 选项 B：ADC 模式（实验性）
预期的键：

- `ECG`：1D 波形
- `radar_adc`：4D 数组 `[帧, 啁啾, 接收天线, ADC 样本]`（复数或实部+虚部打包）

注意：ADC 模式使用参考提取流程（距离 FFT + 相位提取 + 心脏频带排序）。它旨在用于可复现的处理，但确切行为可能与原始 MMECG 内部预处理不同。

## 3) 生成 radarODE-MTL 片段

### 3.1 RCG 模式

```bash
python tools/prepare_radarode_dataset.py \
  --input ./raw_trials \
  --output ./Dataset \
  --radar-source rcg \
  --glob "*.mat" \
  --signal-fs 200 \
  --sst-time-fs 30 \
  --sst-window-sec 4 \
  --sst-freq-min 1 \
  --sst-freq-max 25 \
  --sst-freq-bins 71 \
  --expected-radar-channels 50 \
  --save-manifest
```

### 3.2 ADC 模式

```bash
python tools/prepare_radarode_dataset.py \
  --input ./adc_trials \
  --output ./Dataset \
  --radar-source adc \
  --adc-key radar_adc \
  --signal-fs 200 \
  --adc-slowtime-fs 200 \
  --adc-num-tx 3 \
  --expected-radar-channels 50 \
  --save-manifest
```

输出文件夹示例：

```text
Dataset/
  obj1_NB_trial001/
    sst_seg_0.npy
    ecg_seg_0.npy
    anchor_seg_0.npy
    ...
```

## 4) 训练 radarODE-MTL

```bash
python Projects/radarODE_plus/main.py \
  --mode train \
  --dataset_path ./Dataset \
  --save_path ./Model_saved \
  --save_name EGA_default \
  --gpu_id 0 \
  --batch_size 22 \
  --n_epochs 200 \
  --weighting EGA \
  --arch HPS \
  --optim sgd \
  --lr 0.005 \
  --weight_decay 0.0005 \
  --momentum 0.937 \
  --scheduler cos \
  --eta_min 0.00005 \
  --T_max 100 \
  --aug_snr 100
```

## 5) 使用保存的模型进行测试

```bash
python Projects/radarODE_plus/main.py \
  --mode test \
  --dataset_path ./Dataset \
  --save_path ./Model_saved \
  --load_path ./Model_saved/best_EGA_default.pt \
  --save_name EGA_default \
  --gpu_id 0 \
  --batch_size 22 \
  --aug_snr 100
```

## 6) 本流程中使用的默认设计选择

这些默认值遵循论文/仓库的行为：

- ECG 峰值检测：NeuroKit2 (`ECG_R_Peaks`)
- 单周期 ECG 目标：可变长度周期，之后在数据加载器中重采样为 200
- SST 设置：Morlet 同步挤压 CWT，频率范围 `[1, 25] Hz`
- SST 片段长度：4 秒
- SST 时间下采样：30 Hz（4 秒对应 `120` 个时间箱）
- 雷达通道：50

## 7) 故障排除

- `Cannot find ECG/RCG key`：传递显式的 `--ecg-key`、`--rcg-key` 或 `--adc-key`。
- MATLAB v7.3 加载问题：确保已安装 `mat73`。
- 生成的片段很少：检查 ECG 质量和 `--signal-fs` 值。
- GPU OOM：减少 `--batch_size` 或 `--num_workers`。
- 小数据集的 `TEST: nan ...`：减少 `--batch_size`（数据加载器使用 `drop_last=True`）。
