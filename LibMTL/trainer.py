import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from tqdm import tqdm

class Trainer(nn.Module):
    r'''多任务学习训练器。

    这是一个统一且可扩展的多任务学习训练框架。

    Args:
        task_dict (dict): 一个由名称-信息对组成的字典，类型为(:class:`str`, :class:`dict`)。\
                            每个任务的子字典有四个条目，关键字分别为 **metrics**、\
                            **metrics_fn**、**loss_fn**、**weight**，每个都对应一个 :class:`list`。
                            **metrics** 列表有 ``m`` 个字符串，表示该任务的 ``m`` 个评估指标名称。\
                            **metrics_fn** 列表有两个元素，即更新函数和评分函数，\
                            分别表示如何在训练过程中更新这些目标以及获得最终分数。\
                            **loss_fn** 列表有 ``m`` 个损失函数，分别对应每个评估指标。\
                            **weight** 列表有 ``m`` 个二进制整数，分别对应每个评估指标，\
                            其中 ``1`` 表示分数越高性能越好，``0`` 表示相反。
        weighting (class): 基于 :class:`LibMTL.weighting.abstract_weighting.AbsWeighting` 的加权策略类。
        architecture (class): 基于 :class:`LibMTL.architecture.abstract_arch.AbsArchitecture` 的架构类。
        encoder_class (class): 神经网络类。
        decoders (dict): 一个由名称-解码器对组成的字典，类型为(:class:`str`, :class:`torch.nn.Module`)。
        rep_grad (bool): 如果为 ``True``，可以计算每个任务的表示梯度。
        multi_input (bool): 如果每个任务有自己的输入数据则为 ``True``，否则为 ``False``。
        optim_param (dict): 优化器的配置字典。
        scheduler_param (dict): 学习率调度器的配置字典。\
                                 如果不使用学习率调度器，请设置为 ``None``。
        kwargs (dict): 加权和架构方法的超参数字典。

    .. note::
            建议使用 :func:`LibMTL.config.prepare_args` 返回 ``optim_param``、\
            ``scheduler_param`` 和 ``kwargs`` 的字典。

    Examples::

        import torch.nn as nn
        from LibMTL import Trainer
        from LibMTL.loss import CE_loss_fn
        from LibMTL.metrics import acc_update_fun, acc_score_fun
        from LibMTL.weighting import EW
        from LibMTL.architecture import HPS
        from LibMTL.model import ResNet18
        from LibMTL.config import prepare_args

        task_dict = {'A': {'metrics': ['Acc'],
                           'metrics_fn': [acc_update_fun, acc_score_fun],
                           'loss_fn': [CE_loss_fn],
                           'weight': [1]}}

        decoders = {'A': nn.Linear(512, 31)}

        # 你可以使用命令行参数并通过 ``prepare_args`` 返回配置。
        # kwargs, optim_param, scheduler_param = prepare_args(params)
        optim_param = {'optim': 'adam', 'lr': 1e-3, 'weight_decay': 1e-4}
        scheduler_param = {'scheduler': 'step'}
        kwargs = {'weight_args': {}, 'arch_args': {}}

        trainer = Trainer(task_dict=task_dict,
                          weighting=EW,
                          architecture=HPS,
                          encoder_class=ResNet18,
                          decoders=decoders,
                          rep_grad=False,
                          multi_input=False,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)

    '''
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, 
                 rep_grad, multi_input, optim_param, scheduler_param, 
                 save_path=None, load_path=None, modelName='best', **kwargs):
        super(Trainer, self).__init__()
        
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

        weighting = weighting_method.__dict__[weighting] 
        architecture = architecture_method.__dict__[architecture]
        
        class MTLmodel(architecture, weighting):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()
                
        self.model = MTLmodel(task_name=self.task_name, 
                              encoder_class=encoder_class, 
                              decoders=decoders, 
                              rep_grad=self.rep_grad, 
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                self.load_path = os.path.join(self.load_path, 'best.pt')
            self.model.load_state_dict(torch.load(self.load_path), strict=False)
            print('Load Model from - {}'.format(self.load_path))
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
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
        else:
            self.scheduler = None

    def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except:
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
        r'''对每个任务的预测进行处理。

        - 默认不进行任何处理。如有需要，可以重写此函数。
        - 如果 ``multi_input`` 为 ``True``，``task_name`` 有效，且类型为 :class:`torch.Tensor` 的 ``preds`` 是该任务的预测结果。
        - 否则，``task_name`` 无效，且 ``preds`` 是所有任务的名称-预测对组成的 :class:`dict`。

        Args:
            preds (dict or torch.Tensor): ``task_name`` 或所有任务的预测结果。
            task_name (str): 任务名称字符串。
        '''
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

    def train(self, train_dataloaders, test_dataloaders, epochs, 
              val_dataloaders=None, return_weight=False):
        r'''多任务学习的训练过程。

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): 用于训练的数据加载器。\
                            如果 ``multi_input`` 为 ``True``，它是一个名称-数据加载器对的字典。\
                            否则，它是一个单一的数据加载器，每次迭代返回数据和一个名称-标签对的字典。

            test_dataloaders (dict or torch.utils.data.DataLoader): 用于验证或测试的数据加载器。\
                            结构与 ``train_dataloaders`` 相同。
            epochs (int): 总的训练轮数。
            return_weight (bool): 如果为 ``True``，将返回损失权重。
        '''
        print(os.path.join(self.save_path, f'best_{self.modelName}.pt'))
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.test_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.metrics_buffer = np.zeros([self.task_num, len(self.task_dict), epochs])
        self.model.epochs = epochs
        for epoch in range(epochs):
            self.model.epoch = epoch
            # print('Epoch:', epoch)
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
                        train_pred = self.model(train_input, task)
                        train_pred = train_pred[task]
                        train_pred = self.process_preds(train_pred, task)
                        train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                        self.meter.update(train_pred, train_gt, task)
                        
                self.optimizer.zero_grad(set_to_none=False)
                w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()
            
            print(w,'\t')
            self.meter.record_time('end')
            self.meter.get_score()
            # for tn, task in enumerate(self.task_name):
            #     num = len(self.task_dict[task]['metrics'])
            #     self.model.metrics_buffer[tn, :num, epoch] = self.meter.results[task]
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            np.save(os.path.join(self.save_path, f'best_{self.modelName}_train.npy'), self.model.train_loss_buffer)
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            self.test(test_dataloaders, self.model.test_loss_buffer, epoch = epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, f'best_{self.modelName}.pt'))
                print('\033[31mBest Model\033[0m {} to {}'.format(epoch, os.path.join(self.save_path, f'best_{self.modelName}.pt')))
                # save the metrics
                np.save(os.path.join(self.save_path, f'best_{self.modelName}_best_result.npy'), self.meter.best_result)
            else:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, f'cur_{self.modelName}.pt'))
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight


    def test(self, test_dataloaders, test_loss_buffer=None, epoch=None, mode='test', return_improvement=False):
        r'''多任务学习的测试过程。

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): 如果 ``multi_input`` 为 ``True``，\
                            它是一个名称-数据加载器对的字典。否则，它是一个单一的数据加载器，\
                            每次迭代返回数据和一个名称-标签对的字典。
            epoch (int, default=None): 当前轮数。
        '''
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        
        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            if not self.multi_input:
                for batch_index in tqdm(range(test_batch)):
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_preds = self.process_preds(test_preds)
                    test_losses = self._compute_loss(test_preds, test_gts)
                    self.meter.update(test_preds, test_gts)
            else:
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):
                        test_input, test_gt = self._process_data(test_loader[task])
                        test_pred = self.model(test_input, task)
                        test_pred = test_pred[task]
                        test_pred = self.process_preds(test_pred)
                        test_loss = self._compute_loss(test_pred, test_gt, task)
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
        

    def test_visiual(self, test_dataloaders, epoch=None, mode='test'):
        r'''用于多任务学习的可视化和评估。
        '''
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        Inputs = []
        Preds = []
        Gts = []
        Losses = []
        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            for batch_index in tqdm(range(test_batch)):
                test_inputs, test_gts = self._process_data(test_loader)
                test_preds = self.model(test_inputs)
                test_preds = self.process_preds(test_preds)
                test_losses = self._compute_loss(test_preds, test_gts)
                Inputs.append(test_inputs)
                Preds.append(test_preds)
                Gts.append(test_gts)
                Losses.append(test_losses)
                self.meter.update(test_preds, test_gts)
        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode)
        improvement = self.meter.improvement
        self.meter.reinit()
        return Inputs, Preds, Gts, Losses, improvement
