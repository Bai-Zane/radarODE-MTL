# radarODE-MTL 复现指南

本仓库实现了一个端到端的可复现流程，基于论文/仓库假设（50 通道、1-25 Hz 的 SST、4 秒窗口、30 Hz SST 时间轴、NeuroKit2 R 峰提取）。

**新增内容**
- 完整复现指南：[REPRODUCE.md](F:/project/radarODE-MTL/REPRODUCE.md)
- 依赖文件：[requirements.txt](F:/project/radarODE-MTL/requirements.txt)
- 原始数据预处理 CLI（RCG 模式 + 原始 ADC 模式）：[tools/prepare_radarode_dataset.py](F:/project/radarODE-MTL/tools/prepare_radarode_dataset.py)
- README 中指向新流程的条目：[README.md](F:/project/radarODE-MTL/README.md)

**可用性的关键修复**
- 训练入口现在由 CLI 驱动（移除了硬编码的数据集/保存路径和硬编码的训练配置）：[main.py](F:/project/radarODE-MTL/Projects/radarODE_plus/main.py)
- 在训练中将 `aug_snr` 正确传递到数据集构造中。
- 使数据集 ID 分割对自定义数据集更稳健（`--test_ids`、自动 ID 发现）。
- 修复了锚点解码器输出不匹配问题（现在与长度为 200 的 `anchor_seg` 一致）。
- 训练器中的 CPU 回退（不强制仅 CUDA）：[trainer.py](F:/project/radarODE-MTL/LibMTL/trainer.py)
- 修复了未提供 epoch 缓冲区时的测试模式崩溃。
- 放宽了数据加载器中的文件夹 ID 匹配以支持生成的数据集命名：
  - [spectrum_dataset.py](F:/project/radarODE-MTL/Projects/radarODE_plus/spectrum_dataset.py)
  - [dataloader.py](F:/project/radarODE-MTL/Projects/radarODE_plus/dataloader.py)
- 移除了编码器中未使用的硬依赖导入：[encoder.py](F:/project/radarODE-MTL/Projects/radarODE_plus/nets/encoder.py)

**如何运行（快速路径）**
1. 安装依赖：
```bash
pip install -r requirements.txt
```
2. 从原始试验构建数据集：
```bash
python tools/prepare_radarode_dataset.py --input <raw_dir> --output ./Dataset --radar-source rcg
```
或（实验性 ADC 路径）：
```bash
python tools/prepare_radarode_dataset.py --input <raw_dir> --output ./Dataset --radar-source adc --adc-key radar_adc
```
3. 训练：
```bash
python Projects/radarODE_plus/main.py --mode train --dataset_path ./Dataset --save_path ./Model_saved
```

**验证完成**
- 所有修改的 Python 文件通过 `py_compile` 检查。
- 在合成试验数据上对预处理 CLI 进行端到端测试；生成了有效的：
  - `sst_seg_*.npy` `(50,71,120)`
  - `ecg_seg_*.npy` 可变长度
  - `anchor_seg_*.npy` `(200,)`
- 在生成的合成数据集上模型测试模式冒烟运行成功。
