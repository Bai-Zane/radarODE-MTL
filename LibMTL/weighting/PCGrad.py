import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class PCGrad(AbsWeighting):
    r"""Project Conflicting Gradients (PCGrad，投影冲突梯度).

    该方法在 `Gradient Surgery for Multi-Task Learning (NeurIPS 2020) <https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html>`_ 中提出，
    并由我们实现。

    .. warning::
            PCGrad 不支持表示梯度，即 ``rep_grad`` 必须为 ``False``。

    """
    def __init__(self):
        super(PCGrad, self).__init__()
        
    def backward(self, losses, **kwargs):
        batch_weight = np.ones(len(losses))
        if self.rep_grad:
            raise ValueError('No support method PCGrad with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward') # [task_num, grad_dim]
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            random.shuffle(task_index)
            for tn_j in task_index:
                g_ij = torch.dot(pc_grads[tn_i], grads[tn_j])
                if g_ij < 0:
                    pc_grads[tn_i] -= g_ij * grads[tn_j] / (grads[tn_j].norm().pow(2)+1e-8)
                    batch_weight[tn_j] -= (g_ij/(grads[tn_j].norm().pow(2)+1e-8)).item()
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        return batch_weight
