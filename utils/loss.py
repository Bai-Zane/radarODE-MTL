import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AbsLoss(object):
    r"""损失函数的抽象类。
    """
    def __init__(self):
        self.record = []
        self.bs = []

    def compute_loss(self, pred, gt):
        r"""计算损失。

        参数:
            pred (torch.Tensor): 预测张量。
            gt (torch.Tensor): 真实标签张量。

        返回:
            torch.Tensor: 损失值。
        """
        pass

    def _update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size()[0])
        return loss

    def _average_loss(self):
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record*bs).sum()/bs.sum()

    def _reinit(self):
        self.record = []
        self.bs = []

class CELoss(AbsLoss):
    r"""交叉熵损失函数。
    """
    def __init__(self):
        super(CELoss, self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss

class KLDivLoss(AbsLoss):
    r"""KL 散度损失函数。
    """
    def __init__(self):
        super(KLDivLoss, self).__init__()

        self.loss_fn = nn.KLDivLoss()

    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss

class L1Loss(AbsLoss):
    r"""平均绝对误差（MAE）损失函数。
    """
    def __init__(self):
        super(L1Loss, self).__init__()

        self.loss_fn = nn.L1Loss()

    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss

class MSELoss(AbsLoss):
    r"""均方误差（MSE）损失函数。
    """
    def __init__(self):
        super(MSELoss, self).__init__()

        self.loss_fn = nn.MSELoss()

    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss
