# radarODE-MTL

``radarODE-MTL`` 是一个基于 [PyTorch](https://pytorch.org/) 和多任务学习 (MTL) 框架 [LibMTL](https://github.com/median-research-group/LibMTL) 构建的开源仓库，包含以下内容：

论文代码：
1. [radarODE: An ODE-embedded deep learning model for contactless ECG reconstruction from millimeter-wave radar](https://arxiv.org/abs/2408.01672)
2. [radarODE-MTL: A Multi-Task Learning Framework with Eccentric Gradient Alignment for Robust Radar-Based ECG Reconstruction](https://arxiv.org/abs/2410.08656)

演示文稿：
1. radarODE
2. radarODE-MTL
3. 流行的 MTL 优化策略介绍

我们最近的工作 [CFT-RFcardi](https://github.com/ZYY0844/CFT-RFcardi) 采用了一个易于使用的框架和预处理数据集。我们建议查看我们的新工作以快速实现和验证。

:partying_face: 如有任何问题，请在 Issues 中提出或发送邮件至 [:email:](yz52357@uga.edu)。

## 直观解释和介绍
基于雷达的生命体征监测已经投入了几十年的研究，该领域始终追求以更好的噪声鲁棒性（特别是对于身体运动或移动中的受试者）捕获更精细的心脏信号。

本仓库包含我们的两项研究，使用多任务学习（MTL）范式和常微分方程（ODE）来提高 ECG 恢复的鲁棒性，其中 **radarODE** 主要改进鲁棒的单周期 ECG 恢复，而 **radarODE-MTL** 最终专注于长期 ECG 恢复。

radarODE 的核心思想是将长期心脏活动解构为单个心脏周期，并利用 ODE 解码器提供的鲁棒性来生成忠实的 ECG 参考，从而帮助长期 ECG 恢复中的域转换。

**不同雷达质量下的鲁棒性说明**

<img src='image/radarODE_MTL_result.jpg' width=700>

## 引用

如果您发现我们的工作对您的研究有帮助，请引用我们的论文：
```
@article{zhang2024radarODE,
  title={{radarODE: An ODE-embedded deep learning model for contactless ECG reconstruction from millimeter-wave radar}},
  author={Yuanyuan Zhang and Runwei Guan and Lingxiao Li and Rui Yang and Yutao Yue and Eng Gee Lim},
  journal={IEEE Transactions on Mobile Computing},
  year={2025},
  publisher={IEEE},
  month={Apr.}
}
@article{zhang2024radarODE-MTL,
  title={{radarODE-MTL}: A Multi-task learning tramework with eccentric gradient alignment for robust radar-based {ECG} reconstruction},
  author={Yuanyuan Zhang and Rui Yang and Yutao Yue and Eng Gee Lim},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2025},
  publisher={IEEE},
  month={Apr.}
}
```

## 数据集下载和准备
请访问 [MMECG Dataset](https://github.com/jinbochen0823/RCG2ECG) 下载数据集。

文件结构如下：
```
Dataset
└───obj1_NB_1_
│   │   sst_seg_0.npy
│   │   anchor_seg_0.npy
│   │   ecg_seg_0.npy
│   │   ...
│   ...
└───obj30_PE_91_
│   │   ...
│   │   sst_seg_215.npy
│   │   anchor_seg_215.npy
│   │   ecg_seg_215.npy
```

radarODE 和 radarODE-MTL 的输入大小为 50x71x120 的频谱图（例如 sst_seg_0.npy），其中 71 表示频率维度，120 表示 4 秒片段的时间维度。真实 ECG、锚点和周期长度可以按照论文中的方式构建。您可以使用示例代码 [MMECG_to_SST](MMECG_to_SST.m) 生成 SST 频谱图，或者使用任何时频表示工具。

## 可复现流程（原始雷达 + 原始 ECG）
在以下文件中提供了一个端到端、可运行的预处理/训练路径：

- [REPRODUCE.md](REPRODUCE.md)

它包括：

1. 环境设置（`requirements.txt`）
2. 原始数据预处理脚本（`tools/prepare_radarode_dataset.py`）
3. 生成 `sst_seg_*.npy`、`ecg_seg_*.npy`、`anchor_seg_*.npy`
4. 用于 `Projects/radarODE_plus/main.py` 的训练/测试命令

## 运行模型
您可以在以下文件中找到参数和设置：

```shell
radarODE-MTL/Projects/radarODE_plus/main.py
```
模型摘要位于：

```shell
radarODE-MTL/Projects/radarODE_plus/nets/model.py
```

### 提示
如果您只想实现 ECG 恢复而不太关心噪声鲁棒性，请将 [此处](https://github.com/ZYY0844/radarODE-MTL/blob/54f4bb658dad03778d2e47032d6482819fe76791/Projects/radarODE_plus/main.py#L170) 改为 ``Given_weight``，并直接使用 4 秒 ECG 作为真实标签。

有关可用的 MTL 架构、优化策略和数据集的更多详细信息，请参阅 [LibMTL](https://github.com/median-research-group/LibMTL)。

## 快速介绍

radarODE-MTL 的完整演示文稿位于：

```shell
radarODE/Presentations/radarODE_MTL_Presentation.pdf
```
<!--
### Overall Framework for radarODE
<img src='image/radarODE.jpg' width=700>

### Overall Framework for radarODE-MTL
<img src='image/radarODE_MTL.jpg' width=700>
 -->


## 许可证

``radarODE-MTL`` 根据 [MIT](./LICENSE) 许可证发布。
