import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AbsArchitecture(nn.Module):
    r"""MTL架构的抽象类。

    Args:
        task_name (list): 所有任务的字符串列表。
        encoder_class (class): 神经网络类。
        decoders (dict): 名称-解码器对的字典，类型为 (:class:`str`, :class:`torch.nn.Module`)。
        rep_grad (bool): 如果为 ``True``，可以计算每个任务表示的梯度。
        multi_input (bool): 如果每个任务有自己的输入数据则为 ``True``，否则为 ``False``。
        device (torch.device): 模型和数据将被分配到的设备。
        kwargs (dict): 架构超参数的字典。

    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(AbsArchitecture, self).__init__()

        self.task_name = task_name
        self.task_num = len(task_name)
        self.encoder_class = encoder_class
        self.decoders = decoders
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.device = device
        self.kwargs = kwargs

        if self.rep_grad:
            self.rep_tasks = {}
            self.rep = {}

    def forward(self, inputs, task_name=None):
        r"""

        Args:
            inputs (torch.Tensor): 输入数据。
            task_name (str, default=None): 如果 ``multi_input`` 为 ``True``，对应 ``inputs`` 的任务名称。

        Returns:
            dict: 名称-预测对的字典，类型为 (:class:`str`, :class:`torch.nn.Module`)。
        """
        out = {}
        s_rep = self.encoder(inputs)
        same_rep = True if not isinstance(s_rep, list) and not self.multi_input else False
        for tn, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue
            ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
            ss_rep = self._prepare_rep(ss_rep, task, same_rep)
            out[task] = self.decoders[task](ss_rep)
        return out

    def get_share_params(self):
        r"""返回模型的共享参数。
        """
        return self.encoder.parameters()

    def zero_grad_share_params(self):
        r"""将共享参数的梯度设置为零。
        """
        self.encoder.zero_grad(set_to_none=False)

    def _prepare_rep(self, rep, task, same_rep=None):
        if self.rep_grad:
            if not same_rep:
                self.rep[task] = rep
            else:
                self.rep = rep
            self.rep_tasks[task] = rep.detach().clone()
            self.rep_tasks[task].requires_grad = True
            return self.rep_tasks[task]
        else:
            return rep
