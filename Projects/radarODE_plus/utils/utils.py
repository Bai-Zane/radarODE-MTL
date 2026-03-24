import torch
import torch.nn as nn
import numpy as np

from LibMTL.loss import AbsLoss
from LibMTL.metrics import AbsMetric
from Projects.radarODE_plus.spectrum_dataset import normalize_to_01_torch

criterion_mse = nn.MSELoss()


def _cross_entropy_loss_shape(ecg_pred, ecg_gt):
    """计算 ECG 形状的交叉熵损失。"""
    ecg_pred = ecg_pred.squeeze(1)
    prob = ecg_gt.squeeze(1).softmax(dim=1)
    return nn.CrossEntropyLoss()(ecg_pred, prob)


def _cross_entropy_loss_ppi(ecg_pred, ecg_gt):
    """计算 PPI 的交叉熵损失。"""
    ecg_pred = ecg_pred.squeeze(1)
    ecg_gt = ecg_gt.squeeze(1)

    # 计算有效 PPI 位置
    valid_counts = ecg_gt.size(1) - (ecg_gt == -10).sum(dim=1)
    batch_indices = torch.arange(ecg_gt.size(0))

    ecg_gt = torch.zeros_like(ecg_gt)
    ecg_gt[batch_indices, valid_counts - 1] = 1000

    prob = ecg_gt.softmax(dim=1)
    return nn.CrossEntropyLoss()(ecg_pred, prob)


def _compute_ppi_error(ecg_pred, ecg_gt):
    """计算 PPI 预测误差。"""
    ecg_pred = ecg_pred.squeeze(1)
    ecg_gt = ecg_gt.squeeze(1)

    valid_counts = ecg_gt.size(1) - (ecg_gt == -10).sum(dim=1) + 1
    ppi_pred = ecg_pred.argmax(dim=1)

    return torch.mean(torch.abs(ppi_pred - valid_counts) / 200)


class shapeMetric(AbsMetric):
    """ECG 形状评估指标。"""

    def __init__(self):
        super().__init__()
        self.mse_record = []
        self.ce_record = []
        self.norm_mse_record = []

    def update_fun(self, pred, gt):
        gt = normalize_to_01_torch(gt.clone().detach()).to(pred.device)
        pred_norm = normalize_to_01_torch(pred.clone().detach())

        self.mse_record.append(criterion_mse(pred, gt).item())
        self.norm_mse_record.append(criterion_mse(pred_norm, gt).item())
        self.ce_record.append(_cross_entropy_loss_shape(pred, gt).item())
        self.bs.append(pred.size(0))

    def score_fun(self):
        records = np.array(self.mse_record)
        batch_size = np.array(self.bs)
        mse = (records * batch_size).sum() / batch_size.sum()

        ce = (np.array(self.ce_record) * batch_size).sum() / batch_size.sum()
        norm_mse = (np.array(self.norm_mse_record) * batch_size).sum() / batch_size.sum()

        return [norm_mse, mse, ce]

    def reinit(self):
        self.mse_record = []
        self.ce_record = []
        self.norm_mse_record = []
        self.bs = []


class shapeLoss(AbsLoss):
    """ECG 形状损失函数。"""

    def compute_loss(self, pred, gt):
        gt = normalize_to_01_torch(gt.clone().detach()).to(pred.device)
        return criterion_mse(pred, gt)


class ppiMetric(AbsMetric):
    """PPI 评估指标。"""

    def __init__(self):
        super().__init__()
        self.ce_record = []
        self.ppi_record = []

    def update_fun(self, pred, gt):
        self.ce_record.append(_cross_entropy_loss_ppi(pred, gt).item())
        self.ppi_record.append(_compute_ppi_error(pred, gt).item())
        self.bs.append(pred.size(0))

    def score_fun(self):
        records = np.array(self.ce_record)
        batch_size = np.array(self.bs)
        ce = (records * batch_size).sum() / batch_size.sum()

        ppi = (np.array(self.ppi_record) * batch_size).sum() / batch_size.sum()
        return [ppi, ce]

    def reinit(self):
        self.ce_record = []
        self.ppi_record = []
        self.bs = []


class ppiLoss(AbsLoss):
    """PPI 损失函数。"""

    def compute_loss(self, pred, gt):
        return _cross_entropy_loss_ppi(pred, gt)


class anchorMetric(AbsMetric):
    """Anchor 评估指标。"""

    def update_fun(self, pred, gt):
        pred_norm = normalize_to_01_torch(pred.clone().detach())
        gt = gt.clone().detach()
        mse = criterion_mse(pred_norm, gt)
        self.record.append(mse.item())
        self.bs.append(pred.size(0))

    def score_fun(self):
        records = np.array(self.record)
        batch_size = np.array(self.bs)
        return [(records * batch_size).sum() / batch_size.sum()]


class anchorLoss(AbsLoss):
    """Anchor 损失函数。"""

    def compute_loss(self, pred, gt):
        gt = normalize_to_01_torch(gt.clone().detach()).to(pred.device)
        return criterion_mse(pred, gt)