import torch, random, copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class DB_MTL(AbsWeighting):
    r"""DB-MTL (Dual-Balancing Multi-Task Learning).

    该方法通过双平衡策略实现多任务学习中的梯度平衡。

    Args:
        DB_beta (float): 梯度缓冲区的更新率。
        DB_beta_sigma (float): 更新率的衰减系数。

    .. warning::
            DB_MTL 不支持表示梯度，即 ``rep_grad`` 必须为 ``False``。

    """

    def __init__(self):
        super(DB_MTL, self).__init__()

    def init_param(self):
        self.step = 0
        self._compute_grad_dim()
        self.grad_buffer = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        
    def backward(self, losses, **kwargs):
        self.step += 1
        beta = kwargs['DB_beta']
        beta_sigma = kwargs['DB_beta_sigma']

        batch_weight = np.ones(len(losses))
        if self.rep_grad:
            raise ValueError('No support method DB_MTL with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            batch_grads = self._compute_grad(torch.log(losses+1e-8), mode='backward') # [task_num, grad_dim]

        self.grad_buffer = batch_grads + (beta/self.step**beta_sigma) * (self.grad_buffer - batch_grads)

        u_grad = self.grad_buffer.norm(dim=-1)

        alpha = u_grad.max() / (u_grad + 1e-8)
        new_grads = sum([alpha[i] * self.grad_buffer[i] for i in range(self.task_num)])

        self._reset_grad(new_grads)
        return batch_weight