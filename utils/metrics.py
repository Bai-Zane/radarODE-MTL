import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AbsMetric(object):
    r"""任务性能指标的抽象类。

    属性:
        record (list): 每次迭代的指标分数列表。
        bs (list): 每次迭代的数据数量列表。
    """
    def __init__(self):
        self.record = []
        self.bs = []

    @property
    def update_fun(self, pred, gt):
        r"""计算每次迭代的指标分数并更新 :attr:`record`。

        参数:
            pred (torch.Tensor): 预测张量。
            gt (torch.Tensor): 真实标签张量。
        """
        pass

    @property
    def score_fun(self):
        r"""计算最终分数（当一个 epoch 结束时）。

        返回:
            list: 指标分数列表。
        """
        pass

    def reinit(self):
        r"""重置 :attr:`record` 和 :attr:`bs`（当一个 epoch 结束时）。
        """
        self.record = []
        self.bs = []

# 准确率
class AccMetric(AbsMetric):
    r"""计算准确率。
    """
    def __init__(self):
        super(AccMetric, self).__init__()

    def update_fun(self, pred, gt):
        r"""
        """
        pred = F.softmax(pred, dim=-1).max(-1)[1]
        self.record.append(gt.eq(pred).sum().item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r"""
        """
        return [(sum(self.record)/sum(self.bs))]


# L1 误差
class L1Metric(AbsMetric):
    r"""计算平均绝对误差（MAE）。
    """
    def __init__(self):
        super(L1Metric, self).__init__()

    def update_fun(self, pred, gt):
        r"""
        """
        abs_err = torch.abs(pred - gt)
        self.record.append(abs_err.item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r"""
        """
        records = np.array(self.record)
        batch_size = np.array(self.bs)
        return [(records*batch_size).sum()/(sum(batch_size))]

