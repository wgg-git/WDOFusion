# 协方差中加上HH
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys, os

# 保持对 WaveFusion 官方框架的兼容
# 请确保你的项目中包含这些注册器和基础工具类
from utils.registry import ARCH_REGISTRY
from archs.arch_utils import LayerNorm, FeedForward


# =============================================================================
# 1. 核心算子库 (V2版的无损 LiftingCore)
# =============================================================================
class LiftingCore(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.update_op = nn.Conv2d(channels * 3, channels, 3, padding=1, groups=channels, bias=False)
        self.predict_hl = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.predict_lh = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.predict_hh = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

    def forward(self, x):
        # 棋盘格切分
        x_e = x[:, :, 0::2, 0::2]
        x_hl = x[:, :, 0::2, 1::2]
        x_lh = x[:, :, 1::2, 0::2]
        x_hh = x[:, :, 1::2, 1::2]

        # 预测与更新逻辑，确保可逆性
        HL = x_hl - self.predict_hl(x_e)
        LH = x_lh - self.predict_lh(x_e)
        HH = x_hh - self.predict_hh(x_e)

        hf_cat = torch.cat([HL, LH, HH], dim=1)
        LL = x_e + self.update_op(hf_cat)

        return LL, hf_cat

    def inverse(self, LL, HF):
        C = LL.shape[1]
        HL, LH, HH = torch.split(HF, C, dim=1)

        x_e = LL - self.update_op(HF)
        x_hl = HL + self.predict_hl(x_e)
        x_lh = LH + self.predict_lh(x_e)
        x_hh = HH + self.predict_hh(x_e)

        B, C, H, W = x_e.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=x_e.device)
        out[:, :, 0::2, 0::2] = x_e
        out[:, :, 0::2, 1::2] = x_hl
        out[:, :, 1::2, 0::2] = x_lh
        out[:, :, 1::2, 1::2] = x_hh

        return out


# =============================================================================
# 2. V1 版高频增强模块 (物理能量逻辑)
# =============================================================================
class Lipschitz_Dynamic_Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.local_mean = nn.AvgPool2d(3, 1, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 3, 1),
            nn.GELU(),
            nn.Conv2d(dim * 3, dim * 3, 1),
            nn.Sigmoid()
        )

    def compute_structural_regularity(self, hf):
        hl, lh, hh = torch.split(hf, self.dim, dim=1)

        # --------------------------------------------------
        # 1. 三个高频子带聚合
        #    保持原始方差计算方式不变
        # --------------------------------------------------
        hf_sum = hl + lh + hh
        mu = self.local_mean(hf_sum)
        variance = self.local_mean(hf_sum * hf_sum) - mu * mu

        # --------------------------------------------------
        # 2. 局部协方差计算
        #    统一计算三个高频子带之间的两两协方差
        # --------------------------------------------------
        def local_covariance(x, y):
            return self.local_mean(x * y) - self.local_mean(x) * self.local_mean(y)

        # HL 与 LH 的协方差
        cov_hl_lh = local_covariance(hl, lh)
        # HL 与 HH 的协方差
        cov_hl_hh = local_covariance(hl, hh)
        # LH 与 HH 的协方差
        cov_lh_hh = local_covariance(lh, hh)

        # --------------------------------------------------
        # 3. 三组协方差进行对称等权聚合
        # --------------------------------------------------
        covar = (torch.abs(cov_hl_lh) + torch.abs(cov_hl_hh) + torch.abs(cov_lh_hh)) / 3.0

        # --------------------------------------------------
        # 4. 结构规律性响应
        #    保持原始响应形式不变
        # --------------------------------------------------
        return torch.abs(variance) + covar

    def forward(self, vi_hf, ir_hf):
        reg_vi = self.compute_structural_regularity(vi_hf)
        reg_ir = self.compute_structural_regularity(ir_hf)

        # 拼接分布，通过门控决定物理权重分配
        w_str = self.gate(torch.cat([reg_vi, reg_ir], dim=1))
        return 1.0 - w_str, w_str


class Orthogonal_HF_Fusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1.0)

    def forward(self, vi_hf, ir_hf, w_str, w_thm):
        # 正交投影融合逻辑
        base_feat = vi_hf * w_str + ir_hf * w_thm
        aux_feat = vi_hf * w_thm + ir_hf * w_str

        dot_base_base = torch.clamp(torch.sum(base_feat * base_feat, dim=1, keepdim=True), min=1e-4)
        proj_aux_on_base = (torch.sum(aux_feat * base_feat, dim=1, keepdim=True) / dot_base_base) * base_feat
        orthogonal_feat = aux_feat - proj_aux_on_base

        detail_boost = self.gamma * F.leaky_relu(orthogonal_feat, negative_slope=0.1)
        return base_feat + detail_boost


# =============================================================================
# 3. 跨频段引导注意力 (HF-Guided DAFF)
# =============================================================================
class DualModalAttention_HF_Guided(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
        super(DualModalAttention_HF_Guided, self).__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.q_conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qk_dwconv1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.q_conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qk_dwconv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.v_conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.v_conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, F1, F2, LL1, LL2):
        b, c, h, w = LL1.shape

        qk1 = self.qk_dwconv1(torch.cat([self.q_conv1(F1), self.k_conv1(F1)], dim=1))
        q1, k1 = qk1.chunk(2, dim=1)
        v1 = self.v_dwconv1(self.v_conv1(LL1))

        qk2 = self.qk_dwconv2(torch.cat([self.q_conv2(F2), self.k_conv2(F2)], dim=1))
        q2, k2 = qk2.chunk(2, dim=1)
        v2 = self.v_dwconv2(self.v_conv2(LL2))

        enhanced_LL1 = self.compute_attention(q1, k1, v1, k2, v2, b, h, w)
        enhanced_LL2 = self.compute_attention(q2, k2, v2, k1, v1, b, h, w)

        return self.project_out(enhanced_LL1), self.project_out(enhanced_LL2)

    def compute_attention(self, q, k_self, v_self, k_cross, v_cross, b, h_v, w_v):
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_self = rearrange(k_self, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_self = rearrange(v_self, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_cross = rearrange(k_cross, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_cross = rearrange(v_cross, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k_self = F.normalize(k_self, dim=-1)
        k_cross = F.normalize(k_cross, dim=-1)

        self_attn = (q @ k_self.transpose(-2, -1)) * self.temperature
        self_attn = self_attn.softmax(dim=-1)
        self_out = self_attn @ v_self

        cross_attn = (q @ k_cross.transpose(-2, -1)) * self.temperature
        cross_attn = (-cross_attn).softmax(dim=-1)
        cross_out = cross_attn @ v_cross

        beta = torch.sigmoid(self.beta)
        out_combined = beta * self_out + cross_out

        out = rearrange(out_combined, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h_v, w=w_v)
        return out


class DAIMBlock_HF_Guided(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias"):
        super(DAIMBlock_HF_Guided, self).__init__()
        self.norm_F1 = LayerNorm(dim, LayerNorm_type)
        self.norm_F2 = LayerNorm(dim, LayerNorm_type)
        self.norm_LL1 = LayerNorm(dim, LayerNorm_type)
        self.norm_LL2 = LayerNorm(dim, LayerNorm_type)

        self.cross_attn = DualModalAttention_HF_Guided(dim, num_heads, bias)

        self.norm_out1 = LayerNorm(dim, LayerNorm_type)
        self.norm_out2 = LayerNorm(dim, LayerNorm_type)

        self.ffn_1 = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn_2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, F1, F2, LL1, LL2):
        enhanced_LL1, enhanced_LL2 = self.cross_attn(
            self.norm_F1(F1), self.norm_F2(F2), self.norm_LL1(LL1), self.norm_LL2(LL2)
        )
        LL1 = LL1 + enhanced_LL1
        LL2 = LL2 + enhanced_LL2

        LL1 = LL1 + self.ffn_1(self.norm_out1(LL1))
        LL2 = LL2 + self.ffn_2(self.norm_out2(LL2))
        return LL1, LL2


class WavePA_DAFF_LL_Fusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.daff = DAIMBlock_HF_Guided(dim=dim, num_heads=num_heads, LayerNorm_type="WithBias")
        self.fuse_out = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )

    def forward(self, F_vi, F_ir, vi_LL, ir_LL):
        vi_enhanced, ir_enhanced = self.daff(F_vi, F_ir, vi_LL, ir_LL)
        return self.fuse_out(torch.cat([vi_enhanced, ir_enhanced], dim=1))


# =============================================================================
# 4. 辅助模块及解码器
# =============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class WaveletSafeEncoder(nn.Module):
    def __init__(self, in_channels, dim):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.res_layers = nn.Sequential(
            ResidualBlock(dim),
            ResidualBlock(dim)
        )

    def forward(self, x):
        x = self.stem(x)
        return x + self.res_layers(x)


class RobustDecoder(nn.Module):
    def __init__(self, dim, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.out_conv = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x):
        res = self.act(self.conv1(x))
        res = self.conv2(res)
        x = x + res
        return torch.sigmoid(self.out_conv(x))


# =============================================================================
# 5. Wave-PA 主网络 (终极动态通道调制 DCM 版)
# =============================================================================
@ARCH_REGISTRY.register()
class LWAVFUNet(nn.Module):
    def __init__(self, inp_channels=1, dim=64, wavestlayer_configs={}, decoder_configs={}):
        super(LWAVFUNet, self).__init__()
        self.inp_channels = inp_channels
        self.dim = dim

        # 1. 编码嵌入
        self.VIS_embed = WaveletSafeEncoder(inp_channels, dim)
        self.IR_embed = WaveletSafeEncoder(inp_channels, dim)

        # 2. 双层提升核心
        self.lifting_l1 = LiftingCore(dim)
        self.lifting_l2 = LiftingCore(dim)

        # 3. 动态物理门控
        self.dmg_l1 = Lipschitz_Dynamic_Gate(dim)
        self.dmg_l2 = Lipschitz_Dynamic_Gate(dim)

        # 4. 低频融合核心
        self.ll_vit = WavePA_DAFF_LL_Fusion(dim, num_heads=4)

        # 5. 高频正交融合核心
        self.ohfi_l1 = Orthogonal_HF_Fusion(dim * 3)
        self.ohfi_l2 = Orthogonal_HF_Fusion(dim * 3)

        # 6. 空间掩膜：生成提亮位置引导 (B, 1, H, W)
        self.mask_conv = nn.Conv2d(dim * 2, 1, 3, padding=1)

        # 7. 动态通道调制器 (DCM)
        self.alpha_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.alpha_gain = nn.Parameter(torch.ones(1, dim, 1, 1) * 3.0)

        # 8. 解码输出
        self.decoder = RobustDecoder(dim, inp_channels)

    def forward(self, modality1, modality2):
        feat_vi = self.VIS_embed(modality1)
        feat_ir = self.IR_embed(modality2)

        # 双层小波无损分解
        vi_ll1, vi_hf1 = self.lifting_l1(feat_vi)
        ir_ll1, ir_hf1 = self.lifting_l1(feat_ir)

        vi_ll2, vi_hf2 = self.lifting_l2(vi_ll1)
        ir_ll2, ir_hf2 = self.lifting_l2(ir_ll1)

        # 物理能量占比计算
        w_thm1, w_str1 = self.dmg_l1(vi_hf1, ir_hf1)
        w_thm2, w_str2 = self.dmg_l2(vi_hf2, ir_hf2)

        # 空间重构：获取富含语义的探照灯特征 F2
        vi_F2 = self.lifting_l2.inverse(vi_ll2, vi_hf2)
        ir_F2 = self.lifting_l2.inverse(ir_ll2, ir_hf2)

        f2_cat = torch.cat([vi_F2, ir_F2], dim=1)

        # 动态计算本图的增强配置
        mask_l2 = torch.sigmoid(self.mask_conv(f2_cat))
        dynamic_alpha = self.alpha_generator(f2_cat) * self.alpha_gain

        # 正常的融合链路
        fused_ll2 = self.ll_vit(vi_F2, ir_F2, vi_ll2, ir_ll2)
        fused_hf2 = self.ohfi_l2(vi_hf2, ir_hf2, w_str2, w_thm2)

        fused_ll1 = self.lifting_l2.inverse(fused_ll2, fused_hf2)
        fused_hf1 = self.ohfi_l1(vi_hf1, ir_hf1, w_str1, w_thm1)
        fused_feat = self.lifting_l1.inverse(fused_ll1, fused_hf1)

        # 空域延后打击：Mask 放大到全图尺寸
        mask_up = F.interpolate(mask_l2, size=fused_feat.shape[2:], mode='bilinear', align_corners=False)
        safe_alpha = torch.clamp(dynamic_alpha, min=-0.99)

        # 终极增强
        fused_feat_enhanced = fused_feat * (1.0 + safe_alpha * mask_up)

        return self.decoder(fused_feat_enhanced)

    def get_wavelet_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)
