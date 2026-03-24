import numpy as np
import torch
import torch.nn as nn


class AbsLoss:
    """损失函数的抽象基类。"""

    def __init__(self):
        self.record = []
        self.bs = []

    def compute_loss(self, pred, gt):
        """计算损失。

        Args:
            pred: 预测张量。
            gt: 真实标签张量。

        Returns:
            损失值。
        """
        raise NotImplementedError

    def _update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size(0))
        return loss

    def _average_loss(self):
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record * bs).sum() / bs.sum()

    def _reinit(self):
        self.record = []
        self.bs = []


class CELoss(AbsLoss):
    """交叉熵损失函数。"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt)


class KLDivLoss(AbsLoss):
    """KL 散度损失函数。"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.KLDivLoss()

    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt)


class L1Loss(AbsLoss):
    """平均绝对误差（MAE）损失函数。"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt)


class MSELoss(AbsLoss):
    """均方误差（MSE）损失函数。"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt)


class RMSELoss(AbsLoss):
    """均方根误差（RMSE）损失函数。"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def compute_loss(self, pred, gt):
        return torch.sqrt(self.loss_fn(pred, gt))