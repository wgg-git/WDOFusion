#跟小波分解++
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from archs import build_network
from losses import build_loss
from metrics import calculate_metric
from utils import get_root_logger, imwrite, tensor2img
from utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import torch.nn as nn
from utils.utils import *
import cv2

def ycrcb_to_bgr(one):
    one = one.astype('float32')
    Y, Cr, Cb = cv2.split(one)
    B = (Cb - 0.5) * 1. / 0.564 + Y
    R = (Cr - 0.5) * 1. / 0.713 + Y
    G = 1. / 0.587 * (Y - 0.299 * R - 0.114 * B)
    return cv2.merge([B, G, R])

def RGB2GRAY(rgb_image):
    """将RGB格式转换为GRAY格式。"""
    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return gray

def RGB2YCrCb(input_im):
    """将RGB图像转换为YCrCb颜色空间"""
    device = input_im.device
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out[:, :1], out[:, 1:2], out[:, 2:3]

def merge_channels(Y, Cr, Cb):
    """Merge separate Y, Cr, Cb tensors into a single tensor."""
    return torch.cat((Y, Cr, Cb), dim=1)

def YCrCb2RGB(Y, Cr, Cb):
    """将YCrCb颜色空间转换回RGB"""
    input_im = torch.cat((Y, Cr, Cb), dim=1)
    device = input_im.device
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def compute_class_distribution(seg_result):
    B, _, H, W = seg_result.shape
    total_pixels = H * W
    class_distribution = []

    for b in range(B):
        img = seg_result[b, 0]
        class_counts = torch.bincount(img.view(-1))
        class_proportions = class_counts.float() / total_pixels
        class_distribution.append(class_proportions)

    return torch.stack(class_distribution)

@MODEL_REGISTRY.register()
class WAVFUModel(BaseModel):
    """Base IF model for infrares-visible image fusion."""

    def __init__(self, opt):
        super(WAVFUModel, self).__init__(opt)
        torch.set_float32_matmul_precision('high')
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.clip_grad_norm_value = 0.1
        self.clip_grad = opt['train'].get('clip_grad', True) if self.is_train else False

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        """初始化训练设置"""
        train_opt = self.opt['train']
        self.clip_grad_norm_value = 0.01

        # EMA设置
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'使用指数移动平均，衰减率: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        self.net_g.train()

        self.losses = ["intensity", "edge", "percep_vis", "percep_ir", "content_mask", "pixel"]
        
        for loss_name in self.losses:
            if train_opt.get(f'{loss_name}_opt'):
                loss_criterion = build_loss(train_opt[f'{loss_name}_opt']).to(self.device)
                setattr(self, f'cri_{loss_name}', loss_criterion)
            else:
                setattr(self, f'cri_{loss_name}', None)

        # 小波损失设置
        if train_opt.get("wavelet_opt"):
            self.cri_wavelet = train_opt['wavelet_opt'].get('use_loss', False)
        else:
            self.cri_wavelet = False

        # 确保至少有一个损失函数
        if not any(getattr(self, f'cri_{loss_name}') for loss_name in self.losses):
            raise ValueError('所有损失函数都未定义')

        # 设置优化器和调度器
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        """加载数据"""
        self.VIS = data['VIS'].to(self.device)

        #这里
        self.IR = data['IR'].to(self.device)
        
        # 🌟 修复脏数据泄露 Bug：强制赋予 None，确保跨数据集时不受上一批数据污染
        self.CBCR = data['CBCR'].to(self.device) if 'CBCR' in data else None
        self.MASK = data['MASK'].to(self.device) if 'MASK' in data else None

    # def optimize_parameters(self, current_iter):
    #     """优化参数"""
    #     self.optimizer_g.zero_grad()
    #     self.data_fusion = self.net_g(self.VIS, self.IR)
        
        
    #     loss_conditions = [
    #         ("cri_intensity", "l_intensity", self.IR, self.VIS, self.data_fusion),
    #         ("cri_edge", "l_edge", self.IR, self.VIS, self.data_fusion),
    #         ("cri_percep_vis", "l_p_vis", self.VIS, self.data_fusion),
    #         ("cri_percep_ir", "l_p_ir", self.IR, self.data_fusion),
    #         ("cri_content_mask", "l_mask", self.IR, self.VIS, self.data_fusion, self.MASK),
    #         ("cri_pixel", "l_pixel", self.IR, self.VIS, self.data_fusion),
    #     ]
        
    #     l_total, loss_dict = self.compute_losses(loss_conditions)

    #     # 🌟 修复 float 属性报错：保留 Tensor 状态传入 dict
    #     if self.cri_wavelet:
    #         net = self.net_g.module if hasattr(self.net_g, 'module') else self.net_g
    #         l_wavelet = net.get_wavelet_loss()
    #         l_total += l_wavelet
    #         loss_dict['l_wavelet'] = l_wavelet  # 🚀 去掉 .item()，确保兼容 reduce_loss_dict 里的 .mean()

    #     # 防止 DDP 报错，挂载一个计算图中所有参数的空梯度 (0 * sum)
    #     l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

    #     # 反向传播和优化
    #     l_total.backward()
    #     if getattr(self, 'clip_grad', False):
    #         nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=self.clip_grad_norm_value, norm_type=2)
    #     self.optimizer_g.step()

    #     self.log_dict = self.reduce_loss_dict(loss_dict)
    #     if self.ema_decay > 0:
    #         self.model_ema(decay=self.ema_decay)
    def optimize_parameters(self, current_iter):
            """优化参数 - 终极监控版"""
            self.optimizer_g.zero_grad()
            self.data_fusion = self.net_g(self.VIS, self.IR)
            
            # 1. 🌟 拆分计算：先处理通用的感知损失和掩码损失
            # 移除原有的 cri_intensity, cri_edge 和 cri_pixel，避免冲突
            base_loss_conditions = [
                ("cri_percep_vis", "l_p_vis", self.VIS, self.data_fusion),
                ("cri_percep_ir", "l_p_ir", self.IR, self.data_fusion),
                ("cri_content_mask", "l_mask", self.IR, self.VIS, self.data_fusion, self.MASK),
            ]
            l_total, loss_dict = self.compute_losses(base_loss_conditions)

            # 2. 🌟 核心修改：手动调用 Ultimate_PA_Loss 并解包子项
            # 确保 cri_pixel 对应的是你的 Ultimate_PA_Loss
            if hasattr(self, 'cri_pixel'):
                # 解包：l_pixel 是用于反向传播的 Tensor，pixel_sub_dict 包含 int, ssim, sobel 的数值
                l_pixel, pixel_sub_dict = self.cri_pixel(self.IR, self.VIS, self.data_fusion)
                
                l_total += l_pixel
                # 将拆解后的子项塞进 loss_dict，这样 train.py 就能打印出 Int, SSIM, Sobel 了
                loss_dict.update(pixel_sub_dict)
                loss_dict['l_pixel_total'] = l_pixel # 记录一个物理损失的总计

            # 3. 处理小波损失 (保持你原有的 DDP 兼容逻辑)
            if self.cri_wavelet:
                net = self.net_g.module if hasattr(self.net_g, 'module') else self.net_g
                l_wavelet = net.get_wavelet_loss()
                l_total += l_wavelet
                loss_dict['l_wavelet'] = l_wavelet 

            # 4. 强制梯度挂载与反向传播
            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())
            l_total.backward()

            if getattr(self, 'clip_grad', False):
                nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=self.clip_grad_norm_value, norm_type=2)
            
            self.optimizer_g.step()

            # 5. 更新日志字典
            self.log_dict = self.reduce_loss_dict(loss_dict)
            
            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)
    def test(self):
        """测试模型"""
        self.VIS, pads = self.pad_to(self.VIS, stride=32, scale=1)
        self.IR, pads = self.pad_to(self.IR, stride=32, scale=1)

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.data_fusion = self.net_g_ema(self.VIS, self.IR)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.data_fusion = self.net_g(self.VIS, self.IR)
            self.net_g.train()

        self.data_fusion = self.unpad(self.data_fusion, pads)
        self.VIS = self.unpad(self.VIS, pads)
        self.IR = self.unpad(self.IR, pads)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics')
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'): 
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)

        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        compute_miou = False  
        
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['IR_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            
            VIS_img = tensor2img(self.VIS.detach().cpu()) 
            IR_img = tensor2img([visuals['IR']]) 
            
            fusion_img = tensor2img([visuals['result']])
            fusion_img=(fusion_img-np.min(fusion_img))/(np.max(fusion_img)-np.min(fusion_img) + 1e-6)
            fusion_img, VIS_img, IR_img = fusion_img * 255, VIS_img * 255, IR_img * 255
            metric_data['img_fusion'] = fusion_img.squeeze(0)
            metric_data['img_A'] = VIS_img.squeeze(0)
            metric_data['img_B'] = IR_img.squeeze(0)
            
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = self._get_save_path(dataset_name, img_name, current_iter)
                else:
                    save_img_path = self._get_save_path(dataset_name, img_name, current_iter)
                
                vi_filepath = val_data['VIS_path'][0]
                fuse_y = fusion_img.squeeze(0)
                img_vi  = cv2.imread(vi_filepath, flags=cv2.IMREAD_COLOR)
                
                if img_vi is not None:
                    vi_ycbcr = cv2.cvtColor(img_vi, cv2.COLOR_BGR2YCrCb)
                    vi_cb = vi_ycbcr[:, :, 1]
                    vi_cr = vi_ycbcr[:, :, 2]
                    fused_ycbcr = np.stack([fuse_y, vi_cb, vi_cr], axis=2).astype(np.uint8)
                    fused_bgr = cv2.cvtColor(fused_ycbcr, cv2.COLOR_YCrCb2BGR)
                    imwrite(save_img_path, fused_bgr)
                else:
                    imwrite(save_img_path, fuse_y)
                
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            best_metric_value = self.best_metric_results[dataset_name][metric]["val"]
            best_metric_iter = self.best_metric_results[dataset_name][metric]["iter"]
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
            if (value==best_metric_value) and (self.is_train==True):
                print(f'Saving best %s models and training states.' % metric)
                self.save_best(metric, best_metric_iter)

        logger = get_root_logger()
        if not self.is_train:
            print(log_str) 
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['VIS'] = (self.VIS.detach().cpu())
        out_dict['IR'] = self.IR.detach().cpu()
        if hasattr(self, 'CBCR') and self.CBCR is not None:
            out_dict['CBCR'] = self.CBCR.detach().cpu().squeeze(0).numpy()
        if hasattr(self, 'data_fusion'):
            out_dict['result'] = self.data_fusion.detach().cpu()

        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def _get_save_path(self, dataset_name, img_name, current_iter):
        """获取保存路径"""
        if self.opt['is_train']:
            return osp.join(self.opt['path']['visualization'], 'train',
                          f"{current_iter:06d}_{img_name}.png")
        else:
            suffix = self.opt['val'].get('suffix', '')
            return osp.join(self.opt['path']['visualization'], dataset_name,
                          f"{img_name}_{suffix}.png" if suffix else f"{img_name}.png")

#进行可见光单输入
# import logging
# import torch
# from collections import OrderedDict
# from os import path as osp
# from tqdm import tqdm
# import numpy as np
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# from archs import build_network
# from losses import build_loss
# from metrics import calculate_metric
# from utils import get_root_logger, imwrite, tensor2img
# from utils.registry import MODEL_REGISTRY
# from .base_model import BaseModel
# import torch.nn as nn
# from utils.utils import *
# import cv2

# @MODEL_REGISTRY.register()
# class WAVFUModel(BaseModel):
#     def __init__(self, opt):
#         super(WAVFUModel, self).__init__(opt)
#         torch.set_float32_matmul_precision('high')
        
#         # 定义网络
#         self.net_g = build_network(opt['network_g'])
#         self.net_g = self.model_to_device(self.net_g)
#         self.clip_grad_norm_value = 0.1
#         self.clip_grad = opt['train'].get('clip_grad', True) if self.is_train else False

#         # 加载权重
#         load_path = self.opt['path'].get('pretrain_network_g', None)
#         if load_path is not None:
#             param_key = self.opt['path'].get('param_key_g', 'params')
#             self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

#         if self.is_train:
#             self.init_training_settings()

#     def init_training_settings(self):
#         train_opt = self.opt['train']
#         self.net_g.train()
#         self.setup_optimizers()
#         self.setup_schedulers()

#     def setup_optimizers(self):
#         train_opt = self.opt['train']
#         optim_params = []
#         for k, v in self.net_g.named_parameters():
#             if v.requires_grad:
#                 optim_params.append(v)
#         optim_type = train_opt['optim_g'].pop('type')
#         self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
#         self.optimizers.append(self.optimizer_g)

#     def feed_data(self, data):
#         """🌟 物理隔离：强制将 IR 设为 VIS，确保模型输入完全一致"""
#         self.VIS = data['VIS'].to(self.device)
#         # 即使配置文件给出了红外路径，我们也直接用 VIS 覆盖，实现纯重构测试
#         self.IR = self.VIS 
        
#         self.CBCR = data['CBCR'].to(self.device) if 'CBCR' in data else None
#         self.MASK = data['MASK'].to(self.device) if 'MASK' in data else None

#     def optimize_parameters(self, current_iter):
#         """训练逻辑：直通模式通常只计算可见光重构 Loss"""
#         self.optimizer_g.zero_grad()
#         self.data_fusion = self.net_g(self.VIS, self.IR)
        
#         # 仅用于诊断时的 Loss 计算
#         l_total = torch.nn.functional.mse_loss(self.data_fusion, self.VIS)
#         l_total.backward()
#         self.optimizer_g.step()

#     def test(self):
#         """🌟 诊断测试：重构逻辑"""
#         # 记录原始 Padding 信息
#         self.VIS, pads = self.pad_to(self.VIS, stride=32, scale=1)
#         # self.IR 已经指向 self.VIS，无需重复 padding

#         self.net_g.eval()
#         with torch.no_grad():
#             # 这里的两个参数实际上都是可见光
#             self.data_fusion = self.net_g(self.VIS, self.IR)
#         self.net_g.train()

#         # 还原 Padding 尺寸
#         self.data_fusion = self.unpad(self.data_fusion, pads)
#         self.VIS = self.unpad(self.VIS, pads)
#         # 此时 self.IR 也需要对应还原，以供后续 visuals 使用
#         self.IR = self.unpad(self.IR, pads)

#     def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
#         dataset_name = dataloader.dataset.opt['name']
#         with_metrics = self.opt['val'].get('metrics')
        
#         if with_metrics:
#             if not hasattr(self, 'metric_results'): 
#                 self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
#             self._initialize_best_metric_results(dataset_name)
#             self.metric_results = {metric: 0 for metric in self.metric_results}

#         metric_data = dict()
#         pbar = tqdm(total=len(dataloader), unit='image')
        
#         for idx, val_data in enumerate(dataloader):
#             img_name = osp.splitext(osp.basename(val_data['VIS_path'][0]))[0]
#             self.feed_data(val_data)
#             self.test()
            
#             # 1. 提取重构结果并强制维度对齐
#             # .squeeze() 确保没有多余的 (H, W, 1) 维度
#             fusion_img = tensor2img(self.data_fusion.detach().cpu()).squeeze()
#             VIS_img = tensor2img(self.VIS.detach().cpu()).squeeze()
            
#             # 归一化并转为 uint8
#             fusion_img = (fusion_img - np.min(fusion_img)) / (np.max(fusion_img) - np.min(fusion_img) + 1e-6)
#             fusion_img = (fusion_img * 255.0).astype(np.uint8)
#             VIS_img = (VIS_img * 255.0).astype(np.uint8)
            
#             if save_img:
#                 save_img_path = self._get_save_path(dataset_name, img_name, current_iter)
#                 # 2. 读取彩色原图进行 YCrCb 恢复
#                 img_vi_color = cv2.imread(val_data['VIS_path'][0], cv2.IMREAD_COLOR)
                
#                 if img_vi_color is not None:
#                     vi_ycbcr = cv2.cvtColor(img_vi_color, cv2.COLOR_BGR2YCrCb)
                    
#                     # 🌟 [修复核心]：如果模型输出与原图尺寸不一致，强制缩放
#                     if fusion_img.shape != vi_ycbcr.shape[:2]:
#                         fusion_img = cv2.resize(fusion_img, (vi_ycbcr.shape[1], vi_ycbcr.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    
#                     # 此时 stack 的所有数组形状绝对一致
#                     fused_ycbcr = np.stack([fusion_img, vi_ycbcr[:, :, 1], vi_ycbcr[:, :, 2]], axis=2).astype(np.uint8)
#                     imwrite(save_img_path, cv2.cvtColor(fused_ycbcr, cv2.COLOR_YCrCb2BGR))
#                 else:
#                     imwrite(save_img_path, fusion_img)
            
#             # 指标计算：重构测试，参考图 A 和 B 均设为 VIS
#             metric_data['img_fusion'] = fusion_img
#             metric_data['img_A'] = VIS_img
#             metric_data['img_B'] = VIS_img

#             if with_metrics:
#                 for name, opt_ in self.opt['val']['metrics'].items():
#                     self.metric_results[name] += calculate_metric(metric_data, opt_)
            
#             pbar.update(1)
#             pbar.set_description(f'Test {img_name}')
#         pbar.close()

#         if with_metrics:
#             for metric in self.metric_results.keys():
#                 self.metric_results[metric] /= (idx + 1)
#             self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

#     def get_current_visuals(self):
#         out_dict = OrderedDict()
#         out_dict['VIS'] = self.VIS.detach().cpu()
#         out_dict['result'] = self.data_fusion.detach().cpu()
#         return out_dict

#     def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
#         log_str = f'Validation {dataset_name}\n'
#         for metric, value in self.metric_results.items():
#             log_str += f'\t # {metric}: {value:.4f}\n'
#         get_root_logger().info(log_str)

#     def save(self, epoch, current_iter):
#         self.save_network(self.net_g, 'net_g', current_iter)

#     def _get_save_path(self, dataset_name, img_name, current_iter):
#         # 确保目录存在
#         save_dir = osp.join(self.opt['path']['visualization'], dataset_name)
#         if not osp.exists(save_dir):
#             os.makedirs(save_dir, exist_ok=True)
#         return osp.join(save_dir, f"{img_name}.png")