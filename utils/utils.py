import random, torch, os
import numpy as np
import torch.nn as nn

def get_root_dir():
    r"""返回项目的根路径。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def set_random_seed(seed):
    r"""设置随机种子以确保可重复性。

    参数:
        seed (int, default=0): 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_device(gpu_id):
    r"""设置模型和数据将被分配到的设备。

    参数:
        gpu_id (str, default='0'): GPU 的 ID。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

def count_parameters(model):
    r'''计算模型的参数数量。

    参数:
        model (torch.nn.Module): 神经网络模块。
    '''
    trainable_params = 0
    non_trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            non_trainable_params += p.numel()
    print('='*40)
    print('总参数:', trainable_params + non_trainable_params)
    print('可训练参数:', trainable_params)
    print('不可训练参数:', non_trainable_params)

def count_improvement(base_result, new_result, weight):
    r"""计算两个结果之间的改进量，计算公式为

    .. math::
        \Delta_{\mathrm{p}}=100\%\times \frac{1}{T}\sum_{t=1}^T
        \frac{1}{M_t}\sum_{m=1}^{M_t}\frac{(-1)^{w_{t,m}}(B_{t,m}-N_{t,m})}{B_{t,m}}.

    参数:
        base_result (dict): 所有任务的所有指标分数的字典。
        new_result (dict): 与 ``base_result`` 结构相同。
        weight (dict): 与 ``base_result`` 结构相同，但每个元素是二进制整数，表示高分更好还是低分更好。

    返回:
        float: ``new_result`` 与 ``base_result`` 之间的改进量。

    示例::

        base_result = {'A': [96, 98], 'B': [0.2]}
        new_result = {'A': [93, 99], 'B': [0.5]}
        weight = {'A': [1, 0], 'B': [1]}

        print(count_improvement(base_result, new_result, weight))
    """
    improvement = 0
    count = 0
    for task in list(base_result.keys()):
        improvement += (((-1)**np.array(weight[task]))*\
                        (np.array(base_result[task])-np.array(new_result[task]))/\
                         np.array(base_result[task])).mean()
        count += 1
    return improvement/count
