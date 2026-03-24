import os
import types
import numpy as np
import torch
import yaml


def load_config(config_path=None):
    """从 YAML 配置文件加载所有配置项，返回 SimpleNamespace。

    Args:
        config_path (str | None): YAML 文件路径。为 None 时自动在项目根目录查找 config.yaml。
    """
    if config_path is None:
        # 向上查找至包含 config.yaml 的目录
        search_dir = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            candidate = os.path.join(search_dir, 'config.yaml')
            if os.path.isfile(candidate):
                config_path = candidate
                break
            search_dir = os.path.dirname(search_dir)
        if config_path is None:
            raise FileNotFoundError(
                '未找到 config.yaml，请在项目根目录创建该文件或通过 load_config(path) 指定路径。'
            )

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    g   = cfg.get('general', {})
    dat = cfg.get('data', {})
    tr  = cfg.get('training', {})
    w   = cfg.get('weighting', {})
    ar  = cfg.get('architecture', {})
    op  = cfg.get('optimizer', {})
    sc  = cfg.get('scheduler', {})

    params = types.SimpleNamespace(
        # general
        mode        = g.get('mode', 'train'),
        seed        = int(g.get('seed', 777)),
        gpu_id      = str(g.get('gpu_id', '0')),
        save_path   = g.get('save_path', './Model_saved'),
        load_path   = g.get('load_path', None),
        save_name   = g.get('save_name', 'EGA'),
        # data
        dataset_path  = dat.get('dataset_path', './Dataset'),
        train_bs      = int(dat.get('train_bs', 22)),
        test_bs       = int(dat.get('test_bs', 22)),
        num_workers   = int(dat.get('num_workers', 8)),
        aug_snr       = int(dat.get('aug_snr', 100)),
        select_sample = bool(dat.get('select_sample', False)),
        id_train_max  = int(dat.get('id_train_max', 88)),
        test_ids      = str(dat.get('test_ids', '75,76,77,78,79,80,81,82,83,84,85')),
        # training
        epochs      = int(tr.get('epochs', 200)),
        rep_grad    = bool(tr.get('rep_grad', False)),
        multi_input = bool(tr.get('multi_input', False)),
        # weighting
        weighting          = w.get('method', 'EW'),
        T                  = float(w.get('T', 2.0)),
        alpha              = float(w.get('alpha', 1.5)),
        mgda_gn            = w.get('mgda_gn', 'none'),
        GradVac_beta       = float(w.get('GradVac_beta', 0.5)),
        GradVac_group_type = int(w.get('GradVac_group_type', 0)),
        leak               = float(w.get('leak', 0.0)),
        calpha             = float(w.get('calpha', 0.5)),
        rescale            = int(w.get('rescale', 1)),
        update_weights_every = int(w.get('update_weights_every', 1)),
        optim_niter        = int(w.get('optim_niter', 20)),
        max_norm           = float(w.get('max_norm', 1.0)),
        MoCo_beta          = float(w.get('MoCo_beta', 0.5)),
        MoCo_beta_sigma    = float(w.get('MoCo_beta_sigma', 0.5)),
        MoCo_gamma         = float(w.get('MoCo_gamma', 0.1)),
        MoCo_gamma_sigma   = float(w.get('MoCo_gamma_sigma', 0.5)),
        MoCo_rho           = float(w.get('MoCo_rho', 0)),
        DB_beta            = float(w.get('DB_beta', 0.9)),
        DB_beta_sigma      = float(w.get('DB_beta_sigma', 0)),
        EGA_temp           = float(w.get('EGA_temp', 1.0)),
        STCH_mu            = float(w.get('STCH_mu', 1.0)),
        STCH_warmup_epoch  = int(w.get('STCH_warmup_epoch', 4)),
        # architecture
        arch        = ar.get('method', 'HPS'),
        img_size    = ar.get('img_size', None),
        num_experts = ar.get('num_experts', None),
        num_nonzeros = int(ar.get('num_nonzeros', 2)),
        kgamma      = float(ar.get('kgamma', 1.0)),
        # optimizer
        optim        = op.get('optim', 'adam'),
        lr           = float(op.get('lr', 1e-4)),
        weight_decay = float(op.get('weight_decay', 1e-5)),
        momentum     = float(op.get('momentum', 0.9)),
        # scheduler
        scheduler  = sc.get('method', None),
        step_size  = int(sc.get('step_size', 100)),
        gamma      = float(sc.get('gamma', 0.5)),
        eta_min    = float(sc.get('eta_min', 5e-5)),
        T_max      = int(sc.get('T_max', 100)),
    )
    return params


# 加权方法的参数映射：方法名 -> [(参数名, 参数属性名), ...]
WEIGHTING_PARAMS = {
    'DWA': [('T', 'T')],
    'GradNorm': [('alpha', 'alpha')],
    'MGDA': [('mgda_gn', 'mgda_gn')],
    'GradVac': [('GradVac_beta', 'GradVac_beta'), ('GradVac_group_type', 'GradVac_group_type')],
    'GradDrop': [('leak', 'leak')],
    'CAGrad': [('calpha', 'calpha'), ('rescale', 'rescale')],
    'Nash_MTL': [('update_weights_every', 'update_weights_every'),
                 ('optim_niter', 'optim_niter'), ('max_norm', 'max_norm')],
    'MoCo': [('MoCo_beta', 'MoCo_beta'), ('MoCo_beta_sigma', 'MoCo_beta_sigma'),
             ('MoCo_gamma', 'MoCo_gamma'), ('MoCo_gamma_sigma', 'MoCo_gamma_sigma'),
             ('MoCo_rho', 'MoCo_rho')],
    'DB_MTL': [('DB_beta', 'DB_beta'), ('DB_beta_sigma', 'DB_beta_sigma')],
    'EGA': [('EGA_temp', 'EGA_temp')],
    'STCH': [('STCH_mu', 'STCH_mu'), ('STCH_warmup_epoch', 'STCH_warmup_epoch')],
}

SUPPORTED_WEIGHTING = ['EW', 'UW', 'GradNorm', 'GLS', 'RLW', 'MGDA', 'IMTL',
                       'PCGrad', 'GradVac', 'CAGrad', 'GradDrop', 'DWA',
                       'Nash_MTL', 'MoCo', 'Aligned_MTL', 'DB_MTL', 'Given_weight', 'EGA', 'STCH']

SUPPORTED_ARCH = ['HPS', 'Cross_stitch', 'MTAN', 'CGC', 'PLE', 'MMoE', 'DSelect_k', 'DIY', 'LTB']


def prepare_args(params):
    """返回超参数、优化器和学习率调度器的配置。

    Args:
        params (SimpleNamespace): 配置参数对象。
    """
    kwargs = {'weight_args': {}, 'arch_args': {}}

    # 处理加权方法参数
    if params.weighting not in SUPPORTED_WEIGHTING:
        raise ValueError(f'不支持的加权方法: {params.weighting}')

    weight_args = WEIGHTING_PARAMS.get(params.weighting, [])
    for param_name, attr_name in weight_args:
        value = getattr(params, attr_name, None)
        if value is None:
            raise ValueError(f'{params.weighting} 需要参数 {param_name}')
        if param_name == 'mgda_gn' and value not in ['none', 'l2', 'loss', 'loss+']:
            raise ValueError(f'MGDA 不支持 mgda_gn={value}')
        kwargs['weight_args'][param_name] = value

    # 处理架构参数
    if params.arch not in SUPPORTED_ARCH:
        raise ValueError(f'不支持的架构方法: {params.arch}')

    if params.arch in ['CGC', 'PLE', 'MMoE', 'DSelect_k']:
        if params.img_size is None:
            raise ValueError(f'{params.arch} 需要参数 img_size')
        kwargs['arch_args']['img_size'] = tuple(params.img_size)
        kwargs['arch_args']['num_experts'] = [int(num) for num in params.num_experts]
    if params.arch == 'DSelect_k':
        kwargs['arch_args']['kgamma'] = params.kgamma
        kwargs['arch_args']['num_nonzeros'] = params.num_nonzeros

    # 处理优化器参数
    if params.optim not in ['adam', 'sgd', 'adagrad', 'rmsprop']:
        raise ValueError(f'不支持的优化器: {params.optim}')

    optim_param = {'optim': params.optim, 'lr': params.lr, 'weight_decay': params.weight_decay}
    if params.optim == 'sgd':
        optim_param['momentum'] = params.momentum

    # 处理学习率调度器参数
    scheduler_param = None
    if params.scheduler is not None:
        if params.scheduler not in ['step', 'cos', 'exp']:
            raise ValueError(f'不支持的调度器: {params.scheduler}')
        if params.scheduler == 'step':
            scheduler_param = {'scheduler': 'step', 'step_size': params.step_size, 'gamma': params.gamma}
        elif params.scheduler == 'cos':
            scheduler_param = {'scheduler': 'cos', 'eta_min': params.eta_min, 'T_max': params.T_max}
        else:
            scheduler_param = {'scheduler': 'exp'}

    _display(params, kwargs, optim_param, scheduler_param)

    return kwargs, optim_param, scheduler_param

def _display(params, kwargs, optim_param, scheduler_param):
    print('='*40)
    print('General Configuration:')
    print('\tMode:', params.mode)
    print('\tWeighting:', params.weighting)
    print('\tArchitecture:', params.arch)
    print('\tRep_Grad:', params.rep_grad)
    print('\tMulti_Input:', params.multi_input)
    print('\tSeed:', params.seed)
    print('\tSave Path:', params.save_path)
    print('\tLoad Path:', params.load_path)
    print('\tDevice: {}'.format('cuda:'+params.gpu_id if torch.cuda.is_available() else 'cpu'))
    for wa, p in zip(['weight_args', 'arch_args'], [params.weighting, params.arch]):
        if kwargs[wa] != {}:
            print('{} Configuration:'.format(p))
            for k, v in kwargs[wa].items():
                print('\t'+k+':', v)
    print('Optimizer Configuration:')
    for k, v in optim_param.items():
        print('\t'+k+':', v)
    if scheduler_param is not None:
        print('Scheduler Configuration:')
        for k, v in scheduler_param.items():
            print('\t'+k+':', v)
