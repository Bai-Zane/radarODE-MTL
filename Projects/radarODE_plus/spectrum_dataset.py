import os
import re

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


def normalize_to_01(data):
    """将数据归一化到 [0, 1] 区间（NumPy 版本）。"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def normalize_to_01_torch(data):
    """将数据归一化到 [0, 1] 区间（PyTorch 版本，按批次处理）。"""
    for i in range(data.size(0)):
        data[i] = (data[i] - torch.min(data[i])) / (torch.max(data[i]) - torch.min(data[i]))
    return data


def _collect_sample_files(directory):
    """收集目录中所有样本文件组（sst, ecg, anchor）。"""
    file_groups = []
    for root, _, files in os.walk(directory):
        num_samples = len(files) // 3
        for i in range(num_samples):
            file_groups.append([
                os.path.join(root, f"sst_seg_{i}.npy"),
                os.path.join(root, f"ecg_seg_{i}.npy"),
                os.path.join(root, f"anchor_seg_{i}.npy")
            ])
    return file_groups


def _add_gaussian_noise(sst, snr_db):
    """向 SST 数据添加指定 SNR 的高斯噪声。"""
    for i in range(sst.shape[0]):
        signal_power = np.mean(sst[i] ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        sst[i] = sst[i] + np.sqrt(noise_power) * np.random.randn(*sst[i].shape)
    return sst


def _add_abrupt_noise(sst, length=1):
    """向 SST 数据添加指定长度的突发噪声。"""
    snr_db = -9 if length <= 10 else 0
    if length > 10:
        length -= 10

    noise_length = min(int(length * 30), sst.shape[-1] - 1)
    start = np.random.randint(0, sst.shape[-1] - noise_length)

    for i in range(sst.shape[0]):
        signal_power = np.mean(sst[i][:, start:start + noise_length] ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(*sst[i][:, start:start + noise_length].shape)
        sst[i][:, start:start + noise_length] += noise
    return sst


def _downsample(ecg, target_len=200):
    """对 ECG 信号进行降采样。"""
    return np.interp(np.linspace(0, len(ecg), target_len), np.arange(len(ecg)), ecg)


def _find_subject_path(index, base_path):
    """根据索引查找受试者目录路径。"""
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            if re.search(f'_{index}_', dir_name) or re.match(fr'obj{index}_', dir_name, re.IGNORECASE):
                return os.path.join(root, dir_name)
    return None


class SpectrumECGDataset(Dataset):
    """频谱图-ECG 数据集。"""

    def __init__(self, sst_ecg_root, aug_snr=100, align_length=200):
        super().__init__()
        self.sst_ecg_root = sst_ecg_root
        self.align_length = align_length
        self.aug_snr = aug_snr
        self.sample_files = _collect_sample_files(self.sst_ecg_root)
        # 用于突发噪声的随机索引（20% 样本）
        self.noise_indices = set(np.random.choice(
            len(self.sample_files),
            size=int(0.2 * len(self.sample_files)),
            replace=False
        ))

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        index = index % len(self.sample_files)
        sst_path, ecg_path, anchor_path = self.sample_files[index]

        sst_data = np.load(sst_path)
        ecg_data = np.load(ecg_path)
        anchor_data = np.load(anchor_path)

        # PPI 信息：将 ECG 填充到目标长度
        ppi_info = np.pad(ecg_data, (0, 260 - ecg_data.shape[-1]),
                          constant_values=-10)

        # 降采样 ECG 用于形状任务
        ecg_data = np.expand_dims(_downsample(ecg_data), 0)
        ppi_info = np.expand_dims(ppi_info, 0)
        anchor_data = np.expand_dims(anchor_data, 0)

        # 数据增强
        if self.aug_snr < 100:
            sst_data = _add_gaussian_noise(sst_data, self.aug_snr)
        elif self.aug_snr > 100 and index in self.noise_indices:
            sst_data = _add_abrupt_noise(sst_data, self.aug_snr % 100)

        sst_data = torch.from_numpy(sst_data).float()
        ecg_data = torch.from_numpy(ecg_data).float()
        ppi_info = torch.from_numpy(ppi_info).float()
        anchor_data = torch.from_numpy(anchor_data).float()

        return sst_data, {'ECG_shape': ecg_data, 'PPI': ppi_info, 'Anchor': anchor_data}


def dataset_concat(ID_selected, data_root, aug_snr=100):
    """连接多个受试者的数据集。"""
    datasets = []
    for ID in ID_selected:
        subject_path = _find_subject_path(ID, data_root)
        if subject_path:
            datasets.append(SpectrumECGDataset(sst_ecg_root=subject_path, aug_snr=aug_snr))
    return ConcatDataset(datasets)

if __name__ == '__main__':
    root = '/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_step/'
    ID = np.arange(3, 7)
    dataset = dataset_concat(ID, root, aug_snr = 101)
    print(ID, len(dataset))
    count = 0
    for item in dataset:
        print(item[0].size(), item[1]['ECG_shape'].size(), item[1]['PPI'].size(), item[1]['Anchor'].size())
        # break

