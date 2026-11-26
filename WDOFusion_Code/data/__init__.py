# from __future__ import absolute_import

# import importlib
# import numpy as np
# import random
# import torch
# import torch.utils.data
# from copy import deepcopy
# from functools import partial
# from os import path as osp

# from .prefetch_dataloader import PrefetchDataLoader
# from utils import get_root_logger, scandir
# from utils.dist_util import get_dist_info
# from utils.registry import DATASET_REGISTRY
# __all__ = ['build_dataset', 'build_dataloader']

# # automatically scan and import dataset modules for registry
# # scan all the files that end with '_dataset.py' under the data folder
# data_folder = osp.dirname(osp.abspath(__file__))
# dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
# # import all the dataset modules
# _dataset_modules = [importlib.import_module(f'data.{file_name}') for file_name in dataset_filenames]

# def build_dataset(dataset_opt):
#     """Build dataset from options.

#     Args:
#         dataset_opt (dict): Configuration for dataset. It must contain:
#             name (str): Dataset name.
#             type (str): Dataset type.
#     """
#     dataset_opt = deepcopy(dataset_opt)
#     dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
#     logger = get_root_logger()
#     logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} ' 'is built.')
#     return dataset


# def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
#     """Build dataloader.

#     Args:
#         dataset (torch.utils.data.Dataset): Dataset.
#         dataset_opt (dict): Dataset options. It contains the following keys:
#             phase (str): 'train' or 'val'.
#             num_worker_per_gpu (int): Number of workers for each GPU.
#             batch_size_per_gpu (int): Training batch size for each GPU.
#         num_gpu (int): Number of GPUs. Used only in the train phase.
#             Default: 1.
#         dist (bool): Whether in distributed training. Used only in the train
#             phase. Default: False.
#         sampler (torch.utils.data.sampler): Data sampler. Default: None.
#         seed (int | None): Seed. Default: None
#     """
#     phase = dataset_opt['phase']
#     rank, _ = get_dist_info()
#     if phase == 'train':
#         if dist:  # distributed training
#             batch_size = dataset_opt['batch_size_per_gpu']
#             num_workers = dataset_opt['num_worker_per_gpu']
#         else:  # non-distributed training
#             multiplier = 1 if num_gpu == 0 else num_gpu
#             batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
#             num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
#         dataloader_args = dict(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             sampler=sampler,
#             drop_last=True)
#         if sampler is None:
#             dataloader_args['shuffle'] = True
#         dataloader_args['worker_init_fn'] = partial(
#             worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
#     elif phase in ['val', 'test']:  # validation
#         dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
#     else:
#         raise ValueError(f'Wrong dataset phase: {phase}. ' "Supported ones are 'train', 'val' and 'test'.")

#     dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
#     dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)

#     prefetch_mode = dataset_opt.get('prefetch_mode')
#     if prefetch_mode == 'cpu':  # CPUPrefetcher
#         num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
#         logger = get_root_logger()
#         logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
#         return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
#     else:
#         # prefetch_mode=None: Normal dataloader
#         # prefetch_mode='cuda': dataloader for CUDAPrefetcher
#         return torch.utils.data.DataLoader(**dataloader_args)


# def worker_init_fn(worker_id, num_workers, rank, seed):
#     # Set the worker seed to num_workers * rank + worker_id + seed
#     worker_seed = num_workers * rank + worker_id + seed
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)



from __future__ import absolute_import

import importlib
import numpy as np
import random
import torch
import torch.utils.data
from copy import deepcopy
from functools import partial
from os import path as osp

from .prefetch_dataloader import PrefetchDataLoader
from utils import get_root_logger, scandir
from utils.dist_util import get_dist_info
from utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset', 'build_dataloader']


# ============================================================
# 1. 自动扫描并导入数据集模块
# ============================================================

data_folder = osp.dirname(osp.abspath(__file__))

dataset_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(data_folder)
    if v.endswith('_dataset.py')
]

_dataset_modules = [
    importlib.import_module(f'data.{file_name}')
    for file_name in dataset_filenames
]


# ============================================================
# 2. 自动获取混合数据集及均衡采样器
# ============================================================

WAVFUMixedDataset = None
WAVFUBalancedBatchSampler = None

for module in _dataset_modules:

    mixed_cls = getattr(
        module,
        'WAVFUMixedDataset',
        None
    )

    sampler_cls = getattr(
        module,
        'WAVFUBalancedBatchSampler',
        None
    )

    if mixed_cls is not None:
        WAVFUMixedDataset = mixed_cls

    if sampler_cls is not None:
        WAVFUBalancedBatchSampler = sampler_cls


# ============================================================
# 3. 构建数据集
# ============================================================

def build_dataset(dataset_opt):

    """Build dataset from options."""

    dataset_opt = deepcopy(dataset_opt)

    dataset = DATASET_REGISTRY.get(
        dataset_opt['type']
    )(dataset_opt)

    logger = get_root_logger()

    logger.info(
        f'Dataset [{dataset.__class__.__name__}] - '
        f'{dataset_opt["name"]} is built.'
    )

    return dataset


# ============================================================
# 4. 构建 DataLoader
# ============================================================

def build_dataloader(
        dataset,
        dataset_opt,
        num_gpu=1,
        dist=False,
        sampler=None,
        seed=None):

    """Build dataloader."""

    phase = dataset_opt['phase']

    rank, _ = get_dist_info()

    logger = get_root_logger()

    # ========================================================
    # 训练阶段
    # ========================================================

    if phase == 'train':

        if dist:

            batch_size = dataset_opt['batch_size_per_gpu']

            num_workers = dataset_opt['num_worker_per_gpu']

        else:

            multiplier = (
                1 if num_gpu == 0 else num_gpu
            )

            batch_size = (
                dataset_opt['batch_size_per_gpu']
                * multiplier
            )

            num_workers = (
                dataset_opt['num_worker_per_gpu']
                * multiplier
            )

        # ----------------------------------------------------
        # 判断是否为 MSRS + LLVIP 混合数据集
        # ----------------------------------------------------

        is_mixed_dataset = (
            (
                WAVFUMixedDataset is not None
                and isinstance(dataset, WAVFUMixedDataset)
            )
            or dataset.__class__.__name__ == 'WAVFUMixedDataset'
        )

        # ----------------------------------------------------
        # 混合数据集专用均衡采样
        # ----------------------------------------------------

        if is_mixed_dataset:

            # 当前版本的均衡采样器针对单 GPU
            # 非分布式训练设计
            if dist:

                raise RuntimeError(
                    '\nWAVFUMixedDataset 当前使用的 '
                    'WAVFUBalancedBatchSampler '
                    '尚未实现分布式数据切分。'
                    '\n请使用单 GPU 非分布式训练。'
                )

            if WAVFUBalancedBatchSampler is None:

                raise ImportError(
                    '\n没有找到 WAVFUBalancedBatchSampler。'
                    '\n请确认该类定义在 data 目录下的 '
                    '*_dataset.py 文件中。'
                )

            if batch_size % 2 != 0:

                raise ValueError(
                    'MSRS + LLVIP 均衡采样要求 '
                    'batch_size 必须是偶数。'
                )

            # 替换原来的普通采样器
            if sampler is not None:

                logger.warning(
                    '检测到原有 sampler，'
                    '混合数据集将使用 '
                    'WAVFUBalancedBatchSampler 替换它。'
                )

            sampler = WAVFUBalancedBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                seed=(
                    int(seed)
                    if seed is not None
                    else 42
                )
            )

            logger.info(
                '\n'
                + '=' * 55
                + '\nMSRS + LLVIP Balanced Sampling Enabled'
                + '\nBatch size: '
                + str(batch_size)
                + '\nMSRS per batch: '
                + str(batch_size // 2)
                + '\nLLVIP per batch: '
                + str(batch_size // 2)
                + '\nBatches per epoch: '
                + str(sampler.num_batches)
                + '\n'
                + '=' * 55
            )

        # ----------------------------------------------------
        # 构建 DataLoader 参数
        # ----------------------------------------------------

        dataloader_args = dict(

            dataset=dataset,

            batch_size=batch_size,

            shuffle=False,

            num_workers=num_workers,

            sampler=sampler,

            drop_last=True
        )

        # 普通数据集没有指定 sampler 时
        # 继续采用原来的随机打乱方式

        if sampler is None:

            dataloader_args['shuffle'] = True

        dataloader_args['worker_init_fn'] = (

            partial(
                worker_init_fn,
                num_workers=num_workers,
                rank=rank,
                seed=seed
            )

            if seed is not None else None
        )

    # ========================================================
    # 验证 / 测试阶段
    # ========================================================

    elif phase in ['val', 'test']:

        dataloader_args = dict(

            dataset=dataset,

            batch_size=1,

            shuffle=False,

            num_workers=0
        )

    else:

        raise ValueError(
            f'Wrong dataset phase: {phase}. '
            "Supported ones are 'train', 'val' and 'test'."
        )

    # ========================================================
    # 通用 DataLoader 参数
    # ========================================================

    dataloader_args['pin_memory'] = (
        dataset_opt.get('pin_memory', False)
    )

    dataloader_args['persistent_workers'] = (
        dataset_opt.get('persistent_workers', False)
    )

    prefetch_mode = dataset_opt.get('prefetch_mode')

    # ========================================================
    # CPU Prefetch
    # ========================================================

    if prefetch_mode == 'cpu':

        num_prefetch_queue = dataset_opt.get(
            'num_prefetch_queue',
            1
        )

        logger.info(
            f'Use {prefetch_mode} prefetch dataloader: '
            f'num_prefetch_queue = {num_prefetch_queue}'
        )

        return PrefetchDataLoader(
            num_prefetch_queue=num_prefetch_queue,
            **dataloader_args
        )

    # ========================================================
    # 普通 DataLoader / CUDA Prefetch
    # ========================================================

    else:

        return torch.utils.data.DataLoader(
            **dataloader_args
        )


# ============================================================
# 5. DataLoader Worker 初始化
# ============================================================

def worker_init_fn(
        worker_id,
        num_workers,
        rank,
        seed):

    """Initialize dataloader workers."""

    worker_seed = (
        num_workers * rank
        + worker_id
        + seed
    )

    np.random.seed(worker_seed)

    random.seed(worker_seed)
