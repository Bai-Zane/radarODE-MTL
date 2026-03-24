import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn

# 用于 VSCode 路径解析
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from LibMTL.config import load_config, prepare_args
from LibMTL.utils import set_random_seed, set_device
from LibMTL import Trainer
from Projects.radarODE_plus.nets.model import backbone, shapeDecoder
from Projects.radarODE_plus.nets.PPI_decoder import PPI_decoder
from Projects.radarODE_plus.spectrum_dataset import dataset_concat
from Projects.radarODE_plus.utils.utils import shapeMetric, shapeLoss, ppiMetric, ppiLoss, anchorMetric, anchorLoss


def _discover_available_ids(dataset_path):
    """发现数据集中所有可用的对象 ID。"""
    if not os.path.isdir(dataset_path):
        return []
    ids = []
    for name in os.listdir(dataset_path):
        m = re.match(r'obj(\d+)_', name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def _parse_test_ids(test_ids_str):
    """解析测试 ID 字符串为整数列表。"""
    values = []
    for item in test_ids_str.split(','):
        item = item.strip()
        if item:
            try:
                values.append(int(item))
            except ValueError:
                continue
    return values


def _create_dataloader(dataset, batch_size, num_workers):
    """创建数据加载器。"""
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


class RadarODEPlus(Trainer):
    """radarODE-MTL 多任务学习模型。"""

    def __init__(self, task_dict, weighting, architecture, encoder_class,
                 decoders, rep_grad, multi_input, optim_param, scheduler_param,
                 save_path, load_path, model_name, **kwargs):
        super().__init__(
            task_dict=task_dict,
            weighting=weighting,
            architecture=architecture,
            encoder_class=encoder_class,
            decoders=decoders,
            rep_grad=rep_grad,
            multi_input=multi_input,
            optim_param=optim_param,
            scheduler_param=scheduler_param,
            save_path=save_path,
            load_path=load_path,
            modelName=model_name,
            **kwargs
        )


def main(params):
    """主函数，处理训练、测试和交叉验证模式。"""
    kwargs, optim_param, scheduler_param = prepare_args(params)

    # 发现可用 ID 并划分训练/测试集
    available_ids = _discover_available_ids(params.dataset_path)
    ID_all = np.array(available_ids) if available_ids else np.arange(1, params.id_train_max + 1)

    requested_test = _parse_test_ids(params.test_ids)
    ID_test = np.array([i for i in requested_test if i in ID_all], dtype=int)
    if ID_test.size == 0:
        ID_test = np.array([int(ID_all[-1])], dtype=int)

    ID_train = np.array([i for i in ID_all if i not in set(ID_test)], dtype=int)
    if ID_train.size == 0:
        print(f'警告: 没有剩余独占的训练ID；使用所有ID进行训练。ID_all={ID_all}, ID_test={ID_test}')
        ID_train = ID_all.copy()

    print('ID_test', ID_test)

    # 定义任务
    task_dict = {
        'ECG_shape': {
            'metrics': ['norm_MSE', 'MSE', 'CE'],
            'metrics_fn': shapeMetric(),
            'loss_fn': shapeLoss(),
            'weight': [0, 0, 0]
        },
        'PPI': {
            'metrics': ['PPI_sec', 'CE'],
            'metrics_fn': ppiMetric(),
            'loss_fn': ppiLoss(),
            'weight': [0, 0]
        },
        'Anchor': {
            'metrics': ['MSE'],
            'metrics_fn': anchorMetric(),
            'loss_fn': anchorLoss(),
            'weight': [0]
        }
    }

    # 定义骨干网络和解码器
    encoder_class = lambda: backbone(in_channels=50)
    decoders = nn.ModuleDict({
        'ECG_shape': shapeDecoder(),
        'PPI': PPI_decoder(output_dim=260),
        'Anchor': PPI_decoder(output_dim=200)
    })

    if params.mode == 'train':
        _run_train(params, ID_train, ID_test, task_dict, encoder_class, decoders,
                   kwargs, optim_param, scheduler_param)
    elif params.mode == 'test':
        testloader = _create_dataloader(
            dataset_concat(ID_test, params.dataset_path, params.aug_snr),
            params.test_bs, params.num_workers
        )
        model = _create_model(params, task_dict, encoder_class, decoders,
                              kwargs, optim_param, scheduler_param)
        model.test(testloader)
    elif params.mode == 'cross_vali':
        _run_cross_validation(params, ID_all, task_dict, encoder_class, decoders,
                              kwargs, optim_param, scheduler_param)
    else:
        raise ValueError(f'不支持的模式: {params.mode}')


def _create_model(params, task_dict, encoder_class, decoders, kwargs, optim_param, scheduler_param):
    """创建并返回模型实例。"""
    return RadarODEPlus(
        task_dict=task_dict,
        weighting=params.weighting,
        architecture=params.arch,
        encoder_class=encoder_class,
        decoders=decoders,
        rep_grad=params.rep_grad,
        multi_input=params.multi_input,
        optim_param=optim_param,
        scheduler_param=scheduler_param,
        save_path=params.save_path,
        load_path=params.load_path,
        model_name=params.save_name,
        **kwargs
    )


def _run_train(params, ID_train, ID_test, task_dict, encoder_class, decoders,
                kwargs, optim_param, scheduler_param):
    """执行训练模式。"""
    trainloader = _create_dataloader(
        dataset_concat(ID_train, params.dataset_path, params.aug_snr),
        params.train_bs, params.num_workers
    )
    testloader = _create_dataloader(
        dataset_concat(ID_test, params.dataset_path, params.aug_snr),
        params.test_bs, params.num_workers
    )
    model = _create_model(params, task_dict, encoder_class, decoders,
                          kwargs, optim_param, scheduler_param)
    model.train(trainloader, testloader, params.epochs)


def _run_cross_validation(params, ID_all, task_dict, encoder_class, decoders,
                           kwargs, optim_param, scheduler_param):
    """执行交叉验证模式。"""
    folds = np.array_split(ID_all, min(10, len(ID_all)))
    for i, ID_test in enumerate(folds):
        if len(ID_test) == 0:
            continue
        ID_train = np.array([idx for idx in ID_all if idx not in set(ID_test)], dtype=int)
        if ID_train.size == 0:
            continue

        params.save_name = f'{params.weighting}_cross_vali_{i}'
        trainloader = _create_dataloader(
            dataset_concat(ID_train, params.dataset_path, params.aug_snr),
            params.train_bs, params.num_workers
        )
        testloader = _create_dataloader(
            dataset_concat(ID_test, params.dataset_path, params.aug_snr),
            params.test_bs, params.num_workers
        )
        model = _create_model(params, task_dict, encoder_class, decoders,
                              kwargs, optim_param, scheduler_param)
        model.train(trainloader, testloader, params.epochs)


if __name__ == "__main__":
    params = load_config()
    os.makedirs(params.save_path, exist_ok=True)

    set_device(params.gpu_id)
    set_random_seed(params.seed)

    if not params.save_name:
        params.save_name = f'{params.weighting}'

    main(params)
