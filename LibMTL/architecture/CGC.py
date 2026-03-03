import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.MMoE import MMoE


class CGC(MMoE):
    r"""定制化门控 (Customized Gate Control, CGC)。

    该方法在 `Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations (ACM RecSys 2020 Best Paper) <https://dl.acm.org/doi/10.1145/3383313.3412236>`_ 中提出，
    并由我们实现。

    Args:
        img_size (list): 输入数据的尺寸。例如，[3, 244, 244] 表示输入图像的尺寸为 3x224x224。
        num_experts (list): 所有任务共享的专家数量和每个任务特定的专家数量。每个专家是一个编码器网络。

    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(CGC, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        
        self.num_experts = {task: self.kwargs['num_experts'][tn+1] for tn, task in enumerate(self.task_name)}
        self.num_experts['share'] = self.kwargs['num_experts'][0]
        self.experts_specific = nn.ModuleDict({task: nn.ModuleList([encoder_class() for _ in range(self.num_experts[task])]) for task in self.task_name})
        self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(self.input_size, 
                                                                          self.num_experts['share']+self.num_experts[task]),
                                                                nn.Softmax(dim=-1)) for task in self.task_name})
        
    def forward(self, inputs, task_name=None):
        experts_shared_rep = torch.stack([e(inputs) for e in self.experts_shared])
        out = {}
        for task in self.task_name:
            if task_name is not None and task != task_name:
                continue
            experts_specific_rep = torch.stack([e(inputs) for e in self.experts_specific[task]])
            selector = self.gate_specific[task](torch.flatten(inputs, start_dim=1)) 
            gate_rep = torch.einsum('ij..., ji -> j...', 
                                    torch.cat([experts_shared_rep, experts_specific_rep], dim=0), 
                                    selector)
            gate_rep = self._prepare_rep(gate_rep, task, same_rep=False)
            out[task] = self.decoders[task](gate_rep)
        return out      

