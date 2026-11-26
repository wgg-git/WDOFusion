# import datetime
# import logging
# import math
# import time
# import torch
# from os import path as osp
# import torchvision

# from data import build_dataloader, build_dataset
# from data.data_sampler import EnlargedSampler
# from data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
# from models import build_model
# from utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
#                            init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
# from utils.options import copy_opt_file, dict2str, parse_options
# from archs import build_network
# import torch.nn as nn


# def init_tb_loggers(opt):
#     # initialize wandb logger before tensorboard logger to allow proper sync
#     if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
#                                                      is not None) and ('debug' not in opt['name']):
#         assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
#         init_wandb_logger(opt)
#     tb_logger = None
#     if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
#         tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
#     return tb_logger


# def create_train_val_dataloader(opt, logger):
#     # create train and val dataloaders
#     train_loader, val_loaders = None, []
#     for phase, dataset_opt in opt['datasets'].items():
#         if phase == 'train':
#             dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
#             train_set = build_dataset(dataset_opt)
#             train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
#             train_loader = build_dataloader(
#                 train_set,
#                 dataset_opt,
#                 num_gpu=opt['num_gpu'],
#                 dist=opt['dist'],
#                 sampler=train_sampler,
#                 seed=opt['manual_seed'])

#             num_iter_per_epoch = math.ceil(
#                 len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
#             phase1_iters = int(opt['train']['phase1_iter'])
#             total_iters = int(opt['train']['total_iter'])
#             phase1_epochs = math.ceil(phase1_iters / (num_iter_per_epoch))
#             total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
#             logger.info('Training statistics:'
#                         f'\n\tNumber of train images: {len(train_set)}'
#                         f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
#                         f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
#                         f'\n\tWorld size (gpu number): {opt["world_size"]}'
#                         f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
#                         f'\n\tTotal epochs: {phase1_epochs}; iters: {phase1_iters}'
#                         f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
#         elif phase.split('_')[0] == 'val':
#             val_set = build_dataset(dataset_opt)
#             val_loader = build_dataloader(
#                 val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
#             logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
#             val_loaders.append(val_loader)
#         else:
#             raise ValueError(f'Dataset phase {phase} is not recognized.')

#     return train_loader, train_sampler, val_loaders, phase1_epochs, total_epochs, total_iters


# def load_resume_state(opt):
#     resume_state_path = None
#     if opt['auto_resume']:
#         state_path = osp.join('experiments', opt['name'], 'training_states')
#         if osp.isdir(state_path):
#             states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
#             if len(states) != 0:
#                 states = [float(v.split('.state')[0]) for v in states]
#                 resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
#                 opt['path']['resume_state'] = resume_state_path
#     else:
#         if opt['path'].get('resume_state'):
#             resume_state_path = opt['path']['resume_state']

#     if resume_state_path is None:
#         resume_state = None
#     else:
#         device_id = torch.cuda.current_device()
#         resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
#         check_resume(opt, resume_state['iter'])
#     return resume_state


# def train_pipeline(root_path):
#     # parse options, set distributed setting, set random seed
#     opt, args = parse_options(root_path, is_train=True)
#     opt['root_path'] = root_path
#     is_debug = 'debug' in opt['name']

#     torch.backends.cudnn.benchmark = True
#     # torch.backends.cudnn.deterministic = True

#     # load resume states if necessary
#     resume_state = load_resume_state(opt)
#     # mkdir for experiments and logger
#     if resume_state is None:
#         make_exp_dirs(opt)
#         if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
#             mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

#     # copy the yml file to the experiment root
#     copy_opt_file(args.opt, opt['path']['experiments_root'])

#     # WARNING: should not use get_root_logger in the above codes, including the called functions
#     # Otherwise the logger will not be properly initialized
#     log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
#     logger = get_root_logger(logger_name='basicIF', log_level=logging.INFO, log_file=log_file)
#     logger.info(get_env_info())
#     logger.info(dict2str(opt))
#     # initialize wandb and tb loggers
#     tb_logger = init_tb_loggers(opt)

#     # create train and validation dataloaders
#     result = create_train_val_dataloader(opt, logger)
#     train_loader, train_sampler, val_loaders, phase1_epochs, total_epochs, total_iters = result

#     # create model
#     model = build_model(opt)
#     if resume_state:  # resume training
#         model.resume_training(resume_state)  # handle optimizers and schedulers
#         logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
#         start_epoch = resume_state['epoch']
#         current_iter = resume_state['iter']
#     else:
#         start_epoch = 0
#         current_iter = 0

#     # create message logger (formatted outputs)
#     msg_logger = MessageLogger(opt, current_iter, tb_logger)

#     # dataloader prefetcher
#     prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
#     if prefetch_mode is None or prefetch_mode == 'cpu':
#         prefetcher = CPUPrefetcher(train_loader)
#     elif prefetch_mode == 'cuda':
#         prefetcher = CUDAPrefetcher(train_loader, opt)
#         logger.info(f'Use {prefetch_mode} prefetch dataloader')
#         if opt['datasets']['train'].get('pin_memory') is not True:
#             raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
#     else:
#         raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

#     # training
#     logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
#     data_timer, iter_timer = AvgTimer(), AvgTimer()
#     start_time = time.time()

#     for epoch in range(start_epoch, total_epochs + 1):
#         train_sampler.set_epoch(epoch)
#         prefetcher.reset()
#         train_data = prefetcher.next()

#         while train_data is not None:
#             data_timer.record()

#             current_iter += 1
#             if current_iter > total_iters:
#                 break
#             # update learning rate
#             model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
#             # training
#             model.feed_data(train_data)
#             model.optimize_parameters(current_iter)

#             iter_timer.record()
#             if current_iter == 1:
#                 # reset start time in msg_logger for more accurate eta_time
#                 # not work in resume mode
#                 msg_logger.reset_start_time()
#             # log
#             if current_iter % opt['logger']['print_freq'] == 0:
#                 log_vars = {'epoch': epoch, 'iter': current_iter}
#                 log_vars.update({'lrs': model.get_current_learning_rate()})
#                 log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
#                 log_vars.update(model.get_current_log())
#                 msg_logger(log_vars)

#             # save models and training states
#             if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
#                 logger.info('Saving models and training states.')
#                 model.save(epoch, current_iter)

#             # validation
#             val_start_iter = opt['val'].get('val_start_iter', 0)  # 获取配置的起始验证迭代，如果没有，则默认为0

#             if is_debug or current_iter >= val_start_iter:  # 只有在超过或等于val_start_iter时才进行验证
#                 if opt.get('val') is not None and (current_iter  % opt['val']['val_freq'] == 0):
#                     if len(val_loaders) > 1:
#                         logger.warning('Multiple validation datasets are *only* supported by SRModel.')
#                     for val_loader in val_loaders:
#                         model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])


#             data_timer.start()
#             iter_timer.start()
#             train_data = prefetcher.next()
#         # end of iter

#     # end of epoch

#     consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
#     logger.info(f'End of training. Time consumed: {consumed_time}')
#     logger.info('Save the latest model.')
#     model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
#     if opt.get('val') is not None:
#         for val_loader in val_loaders:
#             model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
#     if tb_logger:
#         tb_logger.close()


# if __name__ == '__main__':
#     #root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
#     root_path = osp.abspath(osp.join(__file__, osp.pardir))
#     print(root_path)
#     train_pipeline(root_path)
# import datetime
# import logging
# import math
# import time
# import torch
# from os import path as osp
# import torchvision

# from data import build_dataloader, build_dataset
# from data.data_sampler import EnlargedSampler
# from data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
# from models import build_model
# from utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
#                    init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
# from utils.options import copy_opt_file, dict2str, parse_options
# from archs import build_network
# import torch.nn as nn

# # =====================================================================
# # 🚀 硬件底层优化：为 RTX 4090 强制开启 TF32 混合精度加速
# # =====================================================================
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# def init_tb_loggers(opt):
#     if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
#                                                      is not None) and ('debug' not in opt['name']):
#         assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
#         init_wandb_logger(opt)
#     tb_logger = None
#     if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
#         tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
#     return tb_logger

# def create_train_val_dataloader(opt, logger):
#     train_loader, val_loaders = None, []
#     for phase, dataset_opt in opt['datasets'].items():
#         if phase == 'train':
#             dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
#             train_set = build_dataset(dataset_opt)
#             train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
#             train_loader = build_dataloader(
#                 train_set,
#                 dataset_opt,
#                 num_gpu=opt['num_gpu'],
#                 dist=opt['dist'],
#                 sampler=train_sampler,
#                 seed=opt['manual_seed'])

#             num_iter_per_epoch = math.ceil(
#                 len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
#             phase1_iters = int(opt['train']['phase1_iter'])
#             total_iters = int(opt['train']['total_iter'])
#             phase1_epochs = math.ceil(phase1_iters / (num_iter_per_epoch))
#             total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
#             logger.info('Training statistics:'
#                         f'\n\tNumber of train images: {len(train_set)}'
#                         f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
#                         f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
#                         f'\n\tWorld size (gpu number): {opt["world_size"]}'
#                         f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
#                         f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
#         elif phase.split('_')[0] == 'val':
#             val_set = build_dataset(dataset_opt)
#             val_loader = build_dataloader(
#                 val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
#             logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
#             val_loaders.append(val_loader)
#         else:
#             raise ValueError(f'Dataset phase {phase} is not recognized.')

#     return train_loader, train_sampler, val_loaders, phase1_epochs, total_epochs, total_iters

# def load_resume_state(opt):
#     resume_state_path = None
#     if opt['auto_resume']:
#         state_path = osp.join('experiments', opt['name'], 'training_states')
#         if osp.isdir(state_path):
#             states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
#             if len(states) != 0:
#                 states = [float(v.split('.state')[0]) for v in states]
#                 resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
#                 opt['path']['resume_state'] = resume_state_path
#     else:
#         if opt['path'].get('resume_state'):
#             resume_state_path = opt['path']['resume_state']

#     if resume_state_path is None:
#         return None
#     else:
#         device_id = torch.cuda.current_device()
#         resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
#         check_resume(opt, resume_state['iter'])
#         return resume_state

# def train_pipeline(root_path):
#     opt, args = parse_options(root_path, is_train=True)
#     opt['root_path'] = root_path
#     is_debug = 'debug' in opt['name']

#     resume_state = load_resume_state(opt)

#     # =====================================================================
#     # 🌟 修改1：按照当前时间生成唯一的实验文件夹，避免权重覆盖
#     # =====================================================================
#     if resume_state is None and not is_debug:
#         time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#         opt['name'] = f"{opt['name']}_{time_str}"
#         opt['path']['experiments_root'] = osp.join(opt['root_path'], 'experiments', opt['name'])
#         opt['path']['models'] = osp.join(opt['path']['experiments_root'], 'models')
#         opt['path']['training_states'] = osp.join(opt['path']['experiments_root'], 'training_states')
#         opt['path']['log'] = opt['path']['experiments_root']
#         opt['path']['visualization'] = osp.join(opt['path']['experiments_root'], 'visualization')

#     if resume_state is None:
#         make_exp_dirs(opt)
#         if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
#             mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

#     copy_opt_file(args.opt, opt['path']['experiments_root'])

#     log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
#     logger = get_root_logger(logger_name='basicIF', log_level=logging.INFO, log_file=log_file)
#     logger.info(get_env_info())
#     logger.info(dict2str(opt))
#     tb_logger = init_tb_loggers(opt)

#     result = create_train_val_dataloader(opt, logger)
#     train_loader, train_sampler, val_loaders, phase1_epochs, total_epochs, total_iters = result

#     model = build_model(opt)
#     if resume_state:
#         model.resume_training(resume_state)
#         logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
#         start_epoch = resume_state['epoch']
#         current_iter = resume_state['iter']
#     else:
#         start_epoch = 0
#         current_iter = 0

#     msg_logger = MessageLogger(opt, current_iter, tb_logger)

#     prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
#     if prefetch_mode is None or prefetch_mode == 'cpu':
#         prefetcher = CPUPrefetcher(train_loader)
#     elif prefetch_mode == 'cuda':
#         prefetcher = CUDAPrefetcher(train_loader, opt)
#         logger.info(f'Use {prefetch_mode} prefetch dataloader')
#     else:
#         raise ValueError(f"Wrong prefetch_mode {prefetch_mode}.")

#     logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
#     data_timer, iter_timer = AvgTimer(), AvgTimer()
#     start_time = time.time()

#     # =====================================================================
#     # 🌟 修改2：废弃原有的冗余嵌套，使用清晰的纯步数(Iter)单循环
#     # =====================================================================
#     epoch = start_epoch
#     train_sampler.set_epoch(epoch)
#     prefetcher.reset()
#     train_data = prefetcher.next()

#     while current_iter < total_iters:
#         # 当取不到数据时，说明这一轮Epoch结束了，后台自动重置数据流
#         if train_data is None:
#             epoch += 1
#             train_sampler.set_epoch(epoch)
#             prefetcher.reset()
#             train_data = prefetcher.next()
#             continue 

#         data_timer.record()
#         current_iter += 1

#         # 更新学习率、投喂数据、反向传播
#         model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
#         model.feed_data(train_data)
#         model.optimize_parameters(current_iter)

#         iter_timer.record()
#         if current_iter == 1:
#             msg_logger.reset_start_time()

#         # 打印日志与计算精准剩余时间
#         if current_iter % opt['logger']['print_freq'] == 0:
#             log_vars = {'epoch': epoch, 'iter': current_iter}
#             log_vars.update({'lrs': model.get_current_learning_rate()})
#             log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
#             log_vars.update(model.get_current_log())
            
#             # 精确计算 ETA
#             remain_iters = total_iters - current_iter
#             eta_seconds = int(remain_iters * iter_timer.get_avg_time())
#             eta_string = str(datetime.timedelta(seconds=eta_seconds))
#             logger.info(f"⏳ [精准进度] 预计剩余训练时间 (ETA): {eta_string}")

#             msg_logger(log_vars)

#         # 保存权重
#         if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
#             logger.info(f'Saving models and training states at iter {current_iter}.')
#             model.save(epoch, current_iter)

#         # 验证集打分存图
#         val_start_iter = opt['val'].get('val_start_iter', 0)
#         if (not is_debug) and (current_iter >= val_start_iter) and (opt.get('val') is not None) and (current_iter % opt['val']['val_freq'] == 0):
#             for val_loader in val_loaders:
#                 model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

#         # 预取下一步数据
#         data_timer.start()
#         iter_timer.start()
#         train_data = prefetcher.next()

#     # 循环结束
#     consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
#     logger.info(f'End of training. Time consumed: {consumed_time}')
#     logger.info('Save the latest model.')
#     model.save(epoch=-1, current_iter=-1)
    
#     if opt.get('val') is not None:
#         for val_loader in val_loaders:
#             model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
#     if tb_logger:
#         tb_logger.close()

# if __name__ == '__main__':
#     root_path = osp.abspath(osp.join(__file__, osp.pardir))
#     print(f"Project Root Path: {root_path}")
#     train_pipeline(root_path)

#上面是常用的，下面的加了输出总损失
import datetime
import logging
import math
import time
import torch
from os import path as osp
import torchvision

from data import build_dataloader, build_dataset
from data.data_sampler import EnlargedSampler
from data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from models import build_model
from utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                   init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from utils.options import copy_opt_file, dict2str, parse_options
from archs import build_network
import torch.nn as nn

# =====================================================================
# 🚀 硬件底层优化：为 RTX 4090 强制开启 TF32 混合精度加速
# =====================================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def init_tb_loggers(opt):
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger

def create_train_val_dataloader(opt, logger):
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            phase1_iters = int(opt['train']['phase1_iter'])
            total_iters = int(opt['train']['total_iter'])
            phase1_epochs = math.ceil(phase1_iters / (num_iter_per_epoch))
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, phase1_epochs, total_epochs, total_iters

def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        return None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
        return resume_state

def train_pipeline(root_path):
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    is_debug = 'debug' in opt['name']

    resume_state = load_resume_state(opt)

    # =====================================================================
    # 🌟 修改1：按照当前时间生成唯一的实验文件夹，避免权重覆盖
    # =====================================================================
    if resume_state is None and not is_debug:
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        opt['name'] = f"{opt['name']}_{time_str}"
        opt['path']['experiments_root'] = osp.join(opt['root_path'], 'experiments', opt['name'])
        opt['path']['models'] = osp.join(opt['path']['experiments_root'], 'models')
        opt['path']['training_states'] = osp.join(opt['path']['experiments_root'], 'training_states')
        opt['path']['log'] = opt['path']['experiments_root']
        opt['path']['visualization'] = osp.join(opt['path']['experiments_root'], 'visualization')

    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    copy_opt_file(args.opt, opt['path']['experiments_root'])

    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicIF', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_loggers(opt)

    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, phase1_epochs, total_epochs, total_iters = result

    model = build_model(opt)
    if resume_state:
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}.")

    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    # =====================================================================
    # 🌟 修改2：废弃原有的冗余嵌套，使用清晰的纯步数(Iter)单循环
    # =====================================================================
    epoch = start_epoch
    train_sampler.set_epoch(epoch)
    prefetcher.reset()
    train_data = prefetcher.next()

    while current_iter < total_iters:
        # 当取不到数据时，说明这一轮Epoch结束了，后台自动重置数据流
        if train_data is None:
            epoch += 1
            train_sampler.set_epoch(epoch)
            prefetcher.reset()
            train_data = prefetcher.next()
            continue 

        data_timer.record()
        current_iter += 1

        # 更新学习率、投喂数据、反向传播
        model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
        model.feed_data(train_data)
        model.optimize_parameters(current_iter)

        iter_timer.record()
        if current_iter == 1:
            msg_logger.reset_start_time()

        # =====================================================================
        # 🌟 强化日志与监控模块：输出 Total Loss 和 Mask Alpha
        # =====================================================================
        # if current_iter % opt['logger']['print_freq'] == 0:
        #     log_vars = {'epoch': epoch, 'iter': current_iter}
        #     log_vars.update({'lrs': model.get_current_learning_rate()})
        #     log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
            
        #     # 获取当前所有独立 Loss 的字典
        #     current_logs = model.get_current_log()
        #     log_vars.update(current_logs)
            
        #     # 1. 动态计算 Total Loss (将字典中所有的数值型 loss 累加)
        #     total_loss = sum([v for k, v in current_logs.items() if isinstance(v, (int, float))])
            
        #     # 2. 安全提取架构中的 mask_alpha
        #     try:
        #         # 兼容分布式训练(DDP)带有 .module 包装的情况和单卡模式
        #         net = model.netG.module if hasattr(model.netG, 'module') else model.netG
        #         current_alpha = net.mask_alpha.item()
        #         # 将 alpha 塞进 log_vars，TensorBoard/WandB 会自动帮你画出它的曲线
        #         log_vars['alpha_val'] = current_alpha 
        #     except AttributeError:
        #         current_alpha = -999.0 # 如果找不到参数，给一个明显的异常值防止报错

        #     # 3. 精确计算 ETA
        #     remain_iters = total_iters - current_iter
        #     eta_seconds = int(remain_iters * iter_timer.get_avg_time())
        #     eta_string = str(datetime.timedelta(seconds=eta_seconds))
            
        #     # 4. 高亮醒目地打印到终端
        #     logger.info(f"⏳ [进度] ETA: {eta_string} | 📉 Total Loss: {total_loss:.4f} | 🌟 Mask Alpha: {current_alpha:.4f}")

        #     # 原有的详细日志处理器（会将各项 loss 和 alpha 写入 log 文件和 TB）
        #     msg_logger(log_vars)
        # =====================================================================
        # 🌟 强化日志与监控模块：定制化输出核心 Loss (严谨版)
        # =====================================================================
        if current_iter % opt['logger']['print_freq'] == 0:
            log_vars = {'epoch': epoch, 'iter': current_iter}
            log_vars.update({'lrs': model.get_current_learning_rate()})
            log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
            
            # 获取当前所有独立 Loss 的字典
            current_logs = model.get_current_log()
            log_vars.update(current_logs)
            
            # 1. 动态计算 Total Loss (将字典中所有的数值型 loss 累加)
            total_loss = sum([v for k, v in current_logs.items() if isinstance(v, (int, float))])
            
            # 2. 安全提取架构中的 mask_alpha
            try:
                net = model.netG.module if hasattr(model.netG, 'module') else model.netG
                current_alpha = net.mask_alpha.item()
                log_vars['alpha_val'] = current_alpha 
            except AttributeError:
                current_alpha = -999.0 

            # 3. 🎯 严谨提取指定的 Loss 数值
            # 采用遍历字典累加的策略，防止 KeyError 导致训练中断
            val_int = sum([v for k, v in current_logs.items() if 'int' in k.lower()])
            val_ssim = sum([v for k, v in current_logs.items() if 'ssim' in k.lower()])
            val_sobel = sum([v for k, v in current_logs.items() if 'sobel' in k.lower()])
            # 因为你有 percep_vis 和 percep_ir 两个感知损失，这里将它们合二为一显示总 VGG 压力
            val_vgg = sum([v for k, v in current_logs.items() if 'percep' in k.lower() or 'vgg' in k.lower()])

            # 4. 精确计算 ETA
            remain_iters = total_iters - current_iter
            eta_seconds = int(remain_iters * iter_timer.get_avg_time())
            eta_string = str(datetime.timedelta(seconds=eta_seconds))
            
            # 5. 🔥 定制化高亮终端打印 (隐藏无关项)
            logger.info(
                f"⏳ ETA: {eta_string} | 📉 Total: {total_loss:.4f} | "
                f"Int: {val_int:.4f} | SSIM: {val_ssim:.4f} | Sobel: {val_sobel:.4f} | VGG: {val_vgg:.4f} | "
                f"🌟 Alpha: {current_alpha:.4f}"
            )

            # 写入 log 文件和 TB
            msg_logger(log_vars)
        # =====================================================================
        # =====================================================================

        # 保存权重
        if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
            logger.info(f'Saving models and training states at iter {current_iter}.')
            model.save(epoch, current_iter)

        # 验证集打分存图
        val_start_iter = opt['val'].get('val_start_iter', 0)
        if (not is_debug) and (current_iter >= val_start_iter) and (opt.get('val') is not None) and (current_iter % opt['val']['val_freq'] == 0):
            for val_loader in val_loaders:
                model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

        # 预取下一步数据
        data_timer.start()
        iter_timer.start()
        train_data = prefetcher.next()

    # 循环结束
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)
    
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    print(f"Project Root Path: {root_path}")
    train_pipeline(root_path)