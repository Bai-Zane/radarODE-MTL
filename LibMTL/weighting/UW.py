import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class UW(AbsWeighting):
    r"""Uncertainty Weights (UW，不确定性权重).

    该方法在 `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR 2018) <https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf>`_ 中提出，
    并由我们实现。

    """
    def __init__(self):
        super(UW, self).__init__()
        
    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([-0.5]*self.task_num, device=self.device))
        
    def backward(self, losses, **kwargs):
        loss = (losses/(2*self.loss_scale.exp())+self.loss_scale/2).sum()
        loss.backward()
        return (1/(2*torch.exp(self.loss_scale))).detach().cpu().numpy()
