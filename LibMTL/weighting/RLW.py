import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class RLW(AbsWeighting):
    r"""Random Loss Weighting (RLW，随机损失加权).

    该方法在 `Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (TMLR 2022) <https://openreview.net/forum?id=jjtFD8A1Wx>`_ 中提出，
    并由我们实现。

    """
    def __init__(self):
        super(RLW, self).__init__()
        
    def backward(self, losses, **kwargs):
        batch_weight = F.softmax(torch.randn(self.task_num), dim=-1).to(self.device)
        loss = torch.mul(losses, batch_weight).sum()
        loss.backward()
        return batch_weight.detach().cpu().numpy()
