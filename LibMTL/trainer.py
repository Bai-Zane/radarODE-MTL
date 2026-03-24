import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method


class Trainer(nn.Module):
    """多任务学习训练器。

    这是一个统一且可扩展的多任务学习训练框架。

    Args:
        task_dict (dict): 任务配置字典，包含 metrics、metrics_fn、loss_fn、weight。
        weighting (str): 加权策略名称。
        architecture (str): 架构名称。
        encoder_class (class): 编码器类。
        decoders (nn.ModuleDict): 解码器模块字典。
        rep_grad (bool): 是否对共享表示计算梯度。
        multi_input (bool): 每个任务是否使用独立输入。
        optim_param (dict): 优化器配置。
        scheduler_param (dict): 学习率调度器配置。
        save_path (str): 模型保存路径。
        load_path (str): 模型加载路径。
        modelName (str): 模型名称。
        **kwargs: 其他参数。
    """

    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders,
                 rep_grad, multi_input, optim_param, scheduler_param,
                 save_path=None, load_path=None, modelName='best', **kwargs):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.scheduler_param = scheduler_param
        self.save_path = save_path
        self.load_path = load_path
        self.modelName = modelName

        self._prepare_model(weighting, architecture, encoder_class, decoders)
        self._prepare_optimizer(optim_param, scheduler_param)

        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)

    def _prepare_model(self, weighting, architecture, encoder_class, decoders):
        weighting_cls = weighting_method.__dict__[weighting]
        architecture_cls = architecture_method.__dict__[architecture]

        class MTLModel(architecture_cls, weighting_cls):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super().__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()

        self.model = MTLModel(
            task_name=self.task_name,
            encoder_class=encoder_class,
            decoders=decoders,
            rep_grad=self.rep_grad,
            multi_input=self.multi_input,
            device=self.device,
            kwargs=self.kwargs['arch_args']
        ).to(self.device)

        if self.load_path is not None:
            load_path = self.load_path
            if os.path.isdir(load_path):
                load_path = os.path.join(load_path, 'best.pt')
            self.model.load_state_dict(torch.load(load_path), strict=False)
            print(f'Load Model from - {load_path}')

        count_parameters(self.model)

    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adagrad': torch.optim.Adagrad,
            'rmsprop': torch.optim.RMSprop,
        }
        scheduler_dict = {
            'exp': torch.optim.lr_scheduler.ExponentialLR,
            'step': torch.optim.lr_scheduler.StepLR,
            'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
            'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
        }

        optim_args = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_args)

        if scheduler_param is not None:
            scheduler_args = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_args)
        else:
            self.scheduler = None

    def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except StopIteration:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])

        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label

    def process_preds(self, preds, task_name=None):
        """对每个任务的预测进行处理。

        默认不进行任何处理。如有需要，可以重写此函数。
        """
        return preds

    def _compute_loss(self, preds, gts, task_name=None):
        if not self.multi_input:
            train_losses = torch.zeros(self.task_num).to(self.device)
            for tn, task in enumerate(self.task_name):
                train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])
        else:
            train_losses = self.meter.losses[task_name]._update_loss(preds, gts)
        return train_losses

    def _prepare_dataloaders(self, dataloaders):
        if not self.multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in self.task_name:
                loader[task] = [dataloaders[task], iter(dataloaders[task])]
                batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def train(self, train_dataloaders, test_dataloaders, epochs, val_dataloaders=None, return_weight=False):
        """多任务学习的训练过程。

        Args:
            train_dataloaders: 训练数据加载器。
            test_dataloaders: 测试数据加载器。
            epochs (int): 总训练轮数。
            val_dataloaders: 验证数据加载器（可选）。
            return_weight (bool): 是否返回损失权重。
        """
        print(os.path.join(self.save_path, f'best_{self.modelName}.pt'))
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch

        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.test_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs

        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')

            for batch_index in tqdm(range(train_batch)):
                if not self.multi_input:
                    train_inputs, train_gts = self._process_data(train_loader)
                    train_preds = self.model(train_inputs)
                    train_preds = self.process_preds(train_preds)
                    train_losses = self._compute_loss(train_preds, train_gts)
                    self.meter.update(train_preds, train_gts)
                else:
                    train_losses = torch.zeros(self.task_num).to(self.device)
                    for tn, task in enumerate(self.task_name):
                        train_input, train_gt = self._process_data(train_loader[task])
                        train_pred = self.model(train_input, task)[task]
                        train_pred = self.process_preds(train_pred, task)
                        train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                        self.meter.update(train_pred, train_gt, task)

                self.optimizer.zero_grad(set_to_none=False)
                w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()

            print(w, '\t')
            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            np.save(os.path.join(self.save_path, f'best_{self.modelName}_train.npy'), self.model.train_loss_buffer)
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()

            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)

            self.test(test_dataloaders, self.model.test_loss_buffer, epoch=epoch, mode='test')

            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()

            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, f'best_{self.modelName}.pt'))
                print(f'\033[31mBest Model\033[0m {epoch} to {os.path.join(self.save_path, f"best_{self.modelName}.pt")}')
                np.save(os.path.join(self.save_path, f'best_{self.modelName}_best_result.npy'), self.meter.best_result)
            else:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, f'cur_{self.modelName}.pt'))

        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight

    def test(self, test_dataloaders, test_loss_buffer=None, epoch=None, mode='test', return_improvement=False):
        """多任务学习的测试过程。

        Args:
            test_dataloaders: 测试数据加载器。
            test_loss_buffer: 测试损失缓冲区。
            epoch (int): 当前轮数。
            mode (str): 测试模式。
            return_improvement (bool): 是否返回改进量。
        """
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)

        self.model.eval()
        self.meter.record_time('begin')

        with torch.no_grad():
            if not self.multi_input:
                for batch_index in tqdm(range(test_batch)):
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_preds = self.process_preds(test_preds)
                    self._compute_loss(test_preds, test_gts)
                    self.meter.update(test_preds, test_gts)
            else:
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):
                        test_input, test_gt = self._process_data(test_loader[task])
                        test_pred = self.model(test_input, task)[task]
                        test_pred = self.process_preds(test_pred)
                        self._compute_loss(test_pred, test_gt, task)
                        self.meter.update(test_pred, test_gt, task)

        self.meter.record_time('end')
        self.meter.get_score()

        if test_loss_buffer is not None and epoch is not None:
            test_loss_buffer[:, epoch] = self.meter.loss_item
            np.save(os.path.join(self.save_path, f'best_{self.modelName}_test.npy'), test_loss_buffer)

        self.meter.display(epoch=epoch, mode=mode)
        improvement = self.meter.improvement
        self.meter.reinit()

        if return_improvement:
            return improvement

    def test_visual(self, test_dataloaders, epoch=None, mode='test'):
        """用于多任务学习的可视化和评估。"""
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        inputs, preds, gts, losses = [], [], [], []

        self.model.eval()
        self.meter.record_time('begin')

        with torch.no_grad():
            for batch_index in tqdm(range(test_batch)):
                test_inputs, test_gts = self._process_data(test_loader)
                test_preds = self.model(test_inputs)
                test_preds = self.process_preds(test_preds)
                test_losses = self._compute_loss(test_preds, test_gts)
                inputs.append(test_inputs)
                preds.append(test_preds)
                gts.append(test_gts)
                losses.append(test_losses)
                self.meter.update(test_preds, test_gts)

        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode)
        improvement = self.meter.improvement
        self.meter.reinit()

        return inputs, preds, gts, losses, improvement
