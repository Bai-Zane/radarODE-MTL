import torch, sys, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AbsWeighting(nn.Module):
    r"""权重策略的抽象类。
    """
    def __init__(self):
        super(AbsWeighting, self).__init__()

    def init_param(self):
        r"""定义并初始化特定权重方法所需的一些可训练参数。
        """
        pass

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1

    def _get_grads(self, losses, mode='backward'):
        r"""此函数用于返回表示或共享参数的梯度。

        如果 ``rep_grad`` 为 ``True``，它返回一个包含两个元素的列表。第一个元素是 \
        表示的梯度，大小为 [task_num, batch_size, rep_size]。 \
        第二个元素是调整大小后的梯度，大小为 [task_num, -1]，这意味着 \
        每个任务的梯度被调整为一个向量。

        如果 ``rep_grad`` 为 ``False``，它返回共享参数的梯度，大小 \
        为 [task_num, -1]，这意味着每个任务的梯度被调整为一个向量。
        """
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
                except:
                    raise ValueError('The representation dimensions of different tasks must be consistent')
            return [per_grads, grads]
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode)
            return grads

    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        r"""此函数用于重置梯度并执行反向传播。

        Args:
            batch_weight (torch.Tensor): 大小为 [task_num] 的张量。
            per_grad (torch.Tensor): 如果 ``rep_grad`` 为 True，则需要此项。表示的梯度。
            grads (torch.Tensor): 如果 ``rep_grad`` 为 False，则需要此项。共享参数的梯度。
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                # transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                transformed_grad = sum([batch_weight[i] * per_grads[i] for i in range(self.task_num)])
                self.rep.backward(transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = True if (tn+1)!=self.task_num else False
                    self.rep[task].backward(batch_weight[tn]*per_grads[tn], retain_graph=rg)
        else:
            # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
            # test = ([batch_weight[i] * grads[i] for i in range(self.task_num)])
            new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
            self._reset_grad(new_grads)

    @property
    def backward(self, losses, **kwargs):
        r"""
        Args:
            losses (list): 每个任务损失的列表。
            kwargs (dict): 权重方法的超参数字典。
        """
        pass
