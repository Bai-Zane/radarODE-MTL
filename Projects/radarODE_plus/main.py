import torch, os, sys, re
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# 用于vscode
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.utils import set_random_seed, set_device
from LibMTL.model import resnet_dilated
from LibMTL import Trainer
from utils.utils import shapeMetric, shapeLoss, ppiMetric, ppiLoss, anchorMetric, anchorLoss

from spectrum_dataset import dataset_concat
from nets.PPI_decoder import PPI_decoder
from nets.model import backbone, shapeDecoder
from LibMTL.config import prepare_args
import argparse



def parse_args(parser):
    parser.set_defaults(
        weighting='EGA',
        arch='HPS',
        optim='sgd',
        lr=5e-3,
        weight_decay=5e-4,
        momentum=0.937,
        scheduler='cos',
    )
    parser.add_argument('--train_bs', default=32, type=int,
                        help='训练时的batch size')
    parser.add_argument('--test_bs', default=32, type=int,
                        help='测试时的batch size')
    parser.add_argument('--epochs', default=200,
                        type=int, help='训练轮数')
    parser.add_argument('--dataset_path', default='./Dataset',
                        type=str, help='数据集根路径')
    parser.add_argument('--save_name', default='EGA',
                        type=str, help='检查点名称后缀')
    parser.add_argument('--num_workers', default=8, type=int, help='数据加载器工作进程数')
    # 如果为True，仅选择100个样本进行训练和测试
    parser.add_argument('--select_sample', default=False,
                        type=bool, help='选择样本')
    parser.add_argument('--aug_snr', default=100, type=int, help='100表示无数据增强，否则为SNR值')
    parser.add_argument('--n_epochs', default=200, type=int, help='训练轮数')
    parser.add_argument('--batch_size', default=22, type=int, help='训练/测试batch size')
    parser.add_argument('--eta_min', default=5e-5, type=float, help='余弦调度器的eta_min')
    parser.add_argument('--T_max', default=100, type=int, help='余弦调度器的T_max')
    parser.add_argument('--id_train_max', default=88, type=int, help='分割中包含的最大试验ID')
    parser.add_argument('--test_ids', default='75,76,77,78,79,80,81,82,83,84,85',
                        type=str, help='逗号分隔的测试ID；不存在的ID将被忽略')
    return parser.parse_args()

def _discover_available_ids(dataset_path):
    if not os.path.isdir(dataset_path):
        return []
    ids = []
    for name in os.listdir(dataset_path):
        m = re.match(r'obj(\d+)_', name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(list(set(ids)))


def _parse_test_ids(test_ids_str):
    values = []
    for item in test_ids_str.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            continue
    return values


def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)
    available_ids = _discover_available_ids(params.dataset_path)
    if available_ids:
        ID_all = np.array(available_ids)
    else:
        ID_all = np.arange(1, params.id_train_max + 1)

    requested_test = _parse_test_ids(params.test_ids)
    ID_test = np.array([i for i in requested_test if i in ID_all], dtype=int)
    if ID_test.size == 0:
        # 小型/自定义数据集的回退处理
        ID_test = np.array([int(ID_all[-1])], dtype=int)
    ID_train = np.array([i for i in ID_all if i not in set(ID_test)], dtype=int)
    if ID_train.size == 0:
        print(f'警告: 没有剩余独占的训练ID；使用所有ID进行训练。ID_all={ID_all}, ID_test={ID_test}')
        ID_train = ID_all.copy()

    # ID_test = np.array([1])
    # ID_train = np.array([2])
    print('ID_test', ID_test)

    radarODE_train_set = dataset_concat(
        ID_selected=ID_train, data_root=params.dataset_path, aug_snr=params.aug_snr)
    radarODE_test_set = dataset_concat(
        ID_selected=ID_test, data_root=params.dataset_path, aug_snr=params.aug_snr)

    trainloader = torch.utils.data.DataLoader(
        dataset=radarODE_train_set, batch_size=params.train_bs, shuffle=True, num_workers=params.num_workers, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(
        dataset=radarODE_test_set, batch_size=params.test_bs, shuffle=True, num_workers=params.num_workers, pin_memory=True, drop_last=True)

    # 定义任务
    task_dict = {'ECG_shape': {'metrics': ['norm_MSE', 'MSE', 'CE'],
                               'metrics_fn': shapeMetric(),
                               'loss_fn': shapeLoss(),
                               'weight': [0, 0, 0]},
                 'PPI': {'metrics': ['PPI_sec', 'CE'],
                         'metrics_fn': ppiMetric(),
                         'loss_fn': ppiLoss(),
                         'weight': [0, 0]},
                 'Anchor': {'metrics': ['MSE'],
                            'metrics_fn': anchorMetric(),
                            'loss_fn': anchorLoss(),
                            'weight': [0]}}

    # # 定义骨干网络和编/解码器
    def encoder_class():
        return backbone(in_channels=50)
    num_out_channels = {'PPI': 260, 'Anchor': 200}
    decoders = nn.ModuleDict({'ECG_shape': shapeDecoder(),
                              'PPI': PPI_decoder(output_dim=num_out_channels['PPI']),
                              'Anchor': PPI_decoder(output_dim=num_out_channels['Anchor'])})

    class radarODE_plus(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class,
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, modelName, **kwargs):
            super(radarODE_plus, self).__init__(task_dict=task_dict,
                                             weighting=weighting,
                                             architecture=architecture,
                                             encoder_class=encoder_class,
                                             decoders=decoders,
                                             rep_grad=rep_grad,
                                             multi_input=multi_input,
                                             optim_param=optim_param,
                                             scheduler_param=scheduler_param,
                                             modelName=modelName,
                                             **kwargs)


    radarODE_plus_model = radarODE_plus(task_dict=task_dict,
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
                          modelName=params.save_name,
                          **kwargs)
    if params.mode == 'train':
        radarODE_plus_model.train(trainloader, testloader, params.epochs)
    elif params.mode == 'test':
        radarODE_plus_model.test(testloader)
    elif params.mode == 'cross_vali':
        folds = np.array_split(ID_all, min(10, len(ID_all)))
        for i, ID_test in enumerate(folds):
            if len(ID_test) == 0:
                continue
            ID_train = np.array([idx for idx in ID_all if idx not in set(ID_test)], dtype=int)
            if ID_train.size == 0:
                continue
            params.save_name = f'{params.weighting}_cross_vali_{i}'
            radarODE_train_set = dataset_concat(
                ID_selected=ID_train, data_root=params.dataset_path, aug_snr=params.aug_snr)
            radarODE_test_set = dataset_concat(
                ID_selected=ID_test, data_root=params.dataset_path, aug_snr=params.aug_snr)

            trainloader = torch.utils.data.DataLoader(
                dataset=radarODE_train_set, batch_size=params.train_bs, shuffle=True, num_workers=params.num_workers, pin_memory=True, drop_last=True)
            testloader = torch.utils.data.DataLoader(
                dataset=radarODE_test_set, batch_size=params.test_bs, shuffle=True, num_workers=params.num_workers, pin_memory=True, drop_last=True)
            radarODE_plus_model = radarODE_plus(task_dict=task_dict,
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
                          modelName=params.save_name,
                          **kwargs)
            radarODE_plus_model.train(trainloader, testloader, params.epochs)
    else:
        raise ValueError


if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    os.makedirs(params.save_path, exist_ok=True)

    # 设置设备
    set_device(params.gpu_id)
    # 设置随机种子
    set_random_seed(params.seed)
    params.train_bs, params.test_bs = params.batch_size, params.batch_size
    params.epochs = params.n_epochs
    # 100表示无噪声，否则为SNR值，6,3,0,-1,-2,-3表示SNR，101表示1秒严重突发噪声，111表示1秒轻微突发噪声
    if not params.save_name:
        params.save_name = f'{params.weighting}'

    main(params)
