
import torch
from torch import nn as nn
from torch.nn import functional as F

from archs.vgg_arch import VGGFeatureExtractor
from utils.registry import LOSS_REGISTRY
import kornia
import kornia.losses
from .loss_util import weighted_loss
import numpy as np
import math
from math import exp

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

# =============================================================================
# 1. 基础损失模块
# =============================================================================

@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)

@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)

# =============================================================================
# 2. 核心物理约束模块 (Sobel, Intensity, SSIM)
# =============================================================================

@LOSS_REGISTRY.register()
class SobelLoss2(nn.Module):
    """
    梯度极大值损失：强迫融合图像保留源图像中最锐利的边缘。
    这是保住 Qabf 和 SF 指标的核心。
    """
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(SobelLoss2, self).__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        self.register_buffer('weightx', torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0))
        self.register_buffer('weighty', torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0))
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, img_vi, img_ir, img_fu):
        wx, wy = self.weightx, self.weighty
        
        vi_grad_x = torch.abs(F.conv2d(img_vi, wx, padding=1))
        vi_grad_y = torch.abs(F.conv2d(img_vi, wy, padding=1))
        ir_grad_x = torch.abs(F.conv2d(img_ir, wx, padding=1))
        ir_grad_y = torch.abs(F.conv2d(img_ir, wy, padding=1))
        fu_grad_x = torch.abs(F.conv2d(img_fu, wx, padding=1))
        fu_grad_y = torch.abs(F.conv2d(img_fu, wy, padding=1))
        
        grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
        grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
        
        loss = F.l1_loss(fu_grad_x, grad_joint_x, reduction=self.reduction) + \
               F.l1_loss(fu_grad_y, grad_joint_y, reduction=self.reduction)
        
        return loss * self.loss_weight

# @LOSS_REGISTRY.register()
# class IntensityLoss(nn.Module):
#     def __init__(self, loss_weight=1.0, reduction='mean'):
#         super(IntensityLoss, self).__init__()
#         self.loss_weight = loss_weight
#         self.reduction = reduction

#     def forward(self, img_ir, img_vi, img_fu):
#         target_max = torch.max(img_ir, img_vi)
#         return F.l1_loss(img_fu, target_max, reduction=self.reduction) * self.loss_weight


# @LOSS_REGISTRY.register()
# class IntensityLoss(nn.Module):
#     """
#     小波增强版强度损失 (等效于原 IntensityLoss 的逻辑，但具备频域解耦能力)
#     专门解决 MSRS 数据集 SD 崩盘问题，同时适配 Wave-PA 架构。
#     """
#     def __init__(self, loss_weight=2.0):
#         super(IntensityLoss, self).__init__()
#         self.loss_weight = loss_weight
        
#         # 定义固定权重的 Haar LL 提取算子 (2x2 均值)
#         # 设置 requires_grad=False 确保它只作为物理标尺，不参与参数更新
#         kernel = torch.tensor([[[[0.5, 0.5], 
#                                  [0.5, 0.5]]]], dtype=torch.float32)
#         self.register_buffer('filter', kernel)

#     def forward(self, img_ir, img_vi, img_fu):
#         """
#         参数说明:
#         - img_ir: 红外图像 (B, 1, H, W)
#         - img_vi: 可见光图像 (B, 1, H, W)
#         - img_fu: 融合结果 (B, 1, H, W)
#         """
#         # 1. 物理频域剥离：提取 1/2 尺度的低频分量 (LL)
#         # 这一步会自动抹平 MSRS 这种夜间高频 ISO 噪点
#         ll_ir = F.conv2d(img_ir, self.filter, stride=2, padding=0)
#         ll_vi = F.conv2d(img_vi, self.filter, stride=2, padding=0)
#         ll_fu = F.conv2d(img_fu, self.filter, stride=2, padding=0)
        
#         # 2. 计算强度目标：采用经典 torch.max 逻辑
#         # 此时的 max 是在纯净的物理能量场上计算的，不会被噪点垫高背景
#         target = torch.max(ll_ir, ll_vi)
        
#         # 3. 计算 L1 损失并应用权重
#         loss = F.l1_loss(ll_fu, target)
        
#         return loss * self.loss_weight
@LOSS_REGISTRY.register()
class IntensityLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(IntensityLoss, self).__init__()
        self.loss_weight = loss_weight
        # 物理标尺：使用 0.25 保证亮度守恒，或根据你的实验使用 0.5 作为增益
        kernel = torch.tensor([[[[0.25, 0.25], [0.25, 0.25]]]], dtype=torch.float32)
        self.register_buffer('filter', kernel)

    def forward(self, img_ir, img_vi, img_fu):
        # 1. 预计算全分辨率目标
        target_pixel = torch.max(img_ir, img_vi)

        # 2. 宏观项：下采样监督 (主导 EN, SD)
        # 这里可以使用你发现效果更好的 0.5 系数逻辑，或者增加该项权重
        ll_fu = F.conv2d(img_fu, self.filter, stride=2)
        target_ll = F.conv2d(target_pixel, self.filter, stride=2)
        loss_macro = F.l1_loss(ll_fu, target_ll)

        # 3. 微观项：像素级约束 (消除方块，提升精度)
        # 权重设为 0.05，作为“修正梯度”存在
        loss_micro = F.l1_loss(img_fu, target_pixel)

        # 联合输出：确保小波 LL 域占据绝对主导权
        return self.loss_weight * (loss_macro + 0.05* loss_micro)
@LOSS_REGISTRY.register()
class StructureSSIMLoss(nn.Module):
    def __init__(self, loss_weight=2.0, window_size=11):
        super(StructureSSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.window_size = window_size

    def forward(self, img_ir, img_vi, img_fu):
        loss_ir = kornia.losses.ssim_loss(img_fu, img_ir, window_size=self.window_size)
        loss_vi = kornia.losses.ssim_loss(img_fu, img_vi, window_size=self.window_size)
        return (loss_ir + loss_vi) / 2.0 * self.loss_weight



# =============================================================================
# 4. 🏆 Wave-PA 终极刷榜损失联合体 (升级版)
# =============================================================================

@LOSS_REGISTRY.register()
class Ultimate_PA_Loss(nn.Module):
    """
    Wave-PA 终极刷榜损失联合体 V2：
    1. Intensity (w_int): 保亮度基调。
    2. SSIM (w_ssim): 保结构感知。
    3. Sobel (w_sobel): 保边缘梯度 (Qabf 的核心)。
    """
    def __init__(self, w_int=0.0, w_ssim=0.0, w_sobel=1.0):
        super(Ultimate_PA_Loss, self).__init__()
        self.loss_int = IntensityLoss(loss_weight=w_int)
        self.loss_ssim = StructureSSIMLoss(loss_weight=w_ssim)
        self.loss_sobel = SobelLoss2(loss_weight=w_sobel)
    # def forward(self, img_ir, img_vi, img_fu):
    #     # A. 基础亮度与结构保全
    #     l_int = self.intensity_loss_call(img_ir, img_vi, img_fu)
    #     l_ssim = self.loss_ssim(img_ir, img_vi, img_fu)
        
    #     # B. 梯度保全 (针对 Qabf, SF)
    #     # 注意顺序：SobelLoss2 内部定义为 (vi, ir, fu)
    #     l_sobel = self.loss_sobel(img_vi, img_ir, img_fu)
        
        
        
    #     # 返回加权总和
    #     return l_int + l_ssim + l_sobel 
    def forward(self, img_ir, img_vi, img_fu):
        # A. 基础亮度与结构保全
        l_int = self.intensity_loss_call(img_ir, img_vi, img_fu)
        l_ssim = self.loss_ssim(img_ir, img_vi, img_fu)
        
        # B. 梯度保全 (针对 Qabf, SF)
        l_sobel = self.loss_sobel(img_vi, img_ir, img_fu)
        
        # 计算总和，确保与之前的训练逻辑 100% 一致
        loss_total = l_int + l_ssim + l_sobel
        
        # 🌟 关键修改：除了返回总和，还返回一个包含子项的字典
        # 这里的键名必须包含 'int', 'ssim', 'sobel'，以便 train.py 匹配
        loss_dict = {
            'l_int': l_int,
            'l_ssim': l_ssim,
            'l_sobel': l_sobel
        }
        
        return loss_total, loss_dict

    def intensity_loss_call(self, img_ir, img_vi, img_fu):
        # 内部封装，确保逻辑清晰
        return self.loss_int(img_ir, img_vi, img_fu)

#下面一步步修改损失函数，上面的是最常用的
