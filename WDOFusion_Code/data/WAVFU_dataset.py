import os
import copy
import math
import random

import cv2
import torch
import torch.utils.data as Data
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Sampler, ConcatDataset
from natsort import natsorted

from utils.registry import DATASET_REGISTRY


# ============================================================
# 1. 基础图像处理函数
# ============================================================

IMAGE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg',
    '.bmp', '.tif', '.tiff'
}


def min_max_normalize(tensor):
    """归一化张量到 [0,1] 范围。"""

    min_val = tensor.min()
    max_val = tensor.max()

    if (max_val - min_val).item() > 1e-12:
        return (tensor - min_val) / (max_val - min_val)

    return tensor


def rgb2y(img):
    """RGB 转 Y 通道。"""

    return (
        img[0:1] * 0.299
        + img[1:2] * 0.587
        + img[2:3] * 0.114
    )


def bgr_to_ycrcb(path):
    """BGR 图像转 YCrCb 颜色空间。"""

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f'无法读取图像: {path}')

    img = img.astype('float32')

    B, G, R = cv2.split(img)

    Y = 0.299 * R + 0.587 * G + 0.114 * B

    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    return Y, cv2.merge([Cr, Cb])


def get_image_names(folder):
    """获取文件夹内所有图像文件名。"""

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f'数据集文件夹不存在: {folder}'
        )

    return natsorted([
        name for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
        and os.path.splitext(name)[1].lower()
        in IMAGE_EXTENSIONS
    ])


def get_image_size(path):
    """读取图像实际尺寸，返回 (W,H)。"""

    try:
        with Image.open(path) as img:
            return img.size

    except Exception as e:
        raise RuntimeError(
            f'读取图像尺寸失败: {path}\n错误: {e}'
        )


# ============================================================
# 2. WAVFUDataset
#
# 功能：
# 1. 根据实际图像尺寸生成训练图像块。
# 2. 同步裁剪红外和可见光图像。
# 3. 自动覆盖图像右侧和底部边缘。
# 4. 测试模式支持 start_idx 断点续跑。
# ============================================================

@DATASET_REGISTRY.register()
class WAVFUDataset(Data.Dataset):

    def __init__(self, opt):

        super(WAVFUDataset, self).__init__()

        self.opt = opt

        self.is_train = bool(opt['is_train'])

        self.img_size = (
            int(opt.get('img_size', 128))
            if self.is_train else None
        )

        self.stride = (
            int(opt.get('stride', 64))
            if self.is_train else None
        )

        self.arg = opt.get('is_arg', True)

        self.is_RGB = opt.get('is_RGB', False)

        self.strict_pairing = opt.get(
            'strict_pairing', True
        )

        # ----------------------------------------------------
        # 数据路径
        # ----------------------------------------------------

        self.vis_folder = os.path.abspath(
            os.path.expanduser(opt['dataroot_source2'])
        )

        self.ir_folder = os.path.abspath(
            os.path.expanduser(opt['dataroot_source1'])
        )

        if not os.path.isdir(self.vis_folder):
            raise FileNotFoundError(self.vis_folder)

        if not os.path.isdir(self.ir_folder):
            raise FileNotFoundError(self.ir_folder)

        # ----------------------------------------------------
        # 检查红外与可见光图像配对
        # ----------------------------------------------------

        ir_names = get_image_names(self.ir_folder)
        vi_names = get_image_names(self.vis_folder)

        ir_set = set(ir_names)
        vi_set = set(vi_names)

        missing_vi = [
            name for name in ir_names
            if name not in vi_set
        ]

        missing_ir = [
            name for name in vi_names
            if name not in ir_set
        ]

        if self.strict_pairing and (
            missing_vi or missing_ir
        ):

            raise RuntimeError(
                f'\n数据集图像配对不完整！'
                f'\nIR 数量: {len(ir_names)}'
                f'\nVI 数量: {len(vi_names)}'
                f'\n缺少 VI: {missing_vi[:10]}'
                f'\n缺少 IR: {missing_ir[:10]}'
            )

        # 保留 IR 文件夹中的自然排序
        self.ir_list = [
            name for name in ir_names
            if name in vi_set
        ]

        if len(self.ir_list) == 0:
            raise RuntimeError(
                '没有找到有效的红外-可见光图像对。'
            )

        if missing_vi or missing_ir:
            print(
                f'[WAVFUDataset] 警告：'
                f'未匹配图像已排除，'
                f'缺少 VI: {len(missing_vi)}, '
                f'缺少 IR: {len(missing_ir)}'
            )

        # ----------------------------------------------------
        # 数据清洗黑名单
        # ----------------------------------------------------

        self.bad_images_blacklist = set(
            opt.get('bad_images_blacklist', [])
        )

        if self.is_train:

            original_count = len(self.ir_list)

            self.ir_list = [
                name for name in self.ir_list
                if name not in self.bad_images_blacklist
            ]

            print(
                f'[WAVFUDataset] 原始图像对: {original_count}, '
                f'屏蔽: {original_count - len(self.ir_list)}, '
                f'实际训练图像对: {len(self.ir_list)}'
            )

        # ====================================================
        # 新增：测试集断点续跑
        #
        # start_idx:
        # 0    = 从第一张开始
        # 3405 = 跳过前 3405 张，从第 3406 张开始
        #
        # 只对验证/测试模式生效，不影响训练模式。
        # ====================================================

        self.start_idx = 0

        if not self.is_train:

            original_pair_count = len(self.ir_list)

            self.start_idx = int(
                opt.get('start_idx', 0)
            )

            if self.start_idx < 0:
                raise ValueError(
                    f'start_idx 不能小于 0，'
                    f'当前值为 {self.start_idx}'
                )

            if self.start_idx > original_pair_count:
                raise ValueError(
                    f'start_idx 超出数据集范围！'
                    f'\n当前数据集共有 {original_pair_count} 对'
                    f'\nstart_idx = {self.start_idx}'
                )

            # 按原有自然排序跳过已经完成的图像
            self.ir_list = self.ir_list[self.start_idx:]

            print('\n' + '=' * 65)
            print('[WAVFUDataset] 测试集断点续跑设置')
            print(f'原始图像对数量: {original_pair_count}')
            print(f'跳过图像对数量: {self.start_idx}')
            print(f'剩余待处理数量: {len(self.ir_list)}')

            if len(self.ir_list) > 0:
                print(
                    f'第一张待处理图像: {self.ir_list[0]}'
                )
                print(
                    f'最后一张待处理图像: {self.ir_list[-1]}'
                )

            print('=' * 65)

        if len(self.ir_list) == 0:
            raise RuntimeError(
                '当前数据集没有剩余可用图像。'
                '请检查 start_idx 设置。'
            )

        # ----------------------------------------------------
        # 训练模式：根据实际图像尺寸建立图像块索引
        # ----------------------------------------------------

        self.patch_info = []

        self.image_patch_counts = {}

        if self.is_train:

            if self.img_size <= 0 or self.stride <= 0:
                raise ValueError(
                    'img_size 和 stride 必须大于 0。'
                )

            for image_name in self.ir_list:

                ir_path = os.path.join(
                    self.ir_folder, image_name
                )

                vi_path = os.path.join(
                    self.vis_folder, image_name
                )

                # 获取实际尺寸
                ir_size = get_image_size(ir_path)
                vi_size = get_image_size(vi_path)

                # 配准图像必须具有相同空间尺寸
                if ir_size != vi_size:

                    raise ValueError(
                        f'\n红外与可见光尺寸不一致: {image_name}'
                        f'\nIR: {ir_size}'
                        f'\nVI: {vi_size}'
                        f'\n请检查图像配准和数据配对。'
                    )

                width, height = ir_size

                # 根据实际宽度和高度生成裁剪位置
                x_positions = self._get_positions(width)
                y_positions = self._get_positions(height)

                patch_count = 0

                for top in y_positions:

                    for left in x_positions:

                        self.patch_info.append(
                            (image_name, top, left)
                        )

                        patch_count += 1

                self.image_patch_counts[image_name] = (
                    patch_count
                )

            if len(self.patch_info) == 0:
                raise RuntimeError(
                    '没有生成有效图像块，请检查图像尺寸。'
                )

            print('\n' + '=' * 65)
            print('[WAVFUDataset] 图像块生成完成')
            print(
                f'数据目录: {os.path.basename(self.ir_folder)}'
            )
            print(f'图像对数量: {len(self.ir_list)}')
            print(f'图像块总数: {len(self.patch_info)}')
            print(f'裁剪尺寸: {self.img_size}×{self.img_size}')
            print(f'滑动步长: {self.stride}')
            print('=' * 65)

        self.crop = None

    # --------------------------------------------------------
    # 根据图像实际尺寸计算滑动窗口位置
    # --------------------------------------------------------

    def _get_positions(self, length):

        if length < self.img_size:

            raise ValueError(
                f'图像尺寸 {length} 小于裁剪尺寸 {self.img_size}'
            )

        max_start = length - self.img_size

        positions = list(
            range(
                0,
                max_start + 1,
                self.stride
            )
        )

        # 确保最后一个裁剪窗口覆盖图像边缘
        if positions[-1] != max_start:
            positions.append(max_start)

        return positions

    # --------------------------------------------------------
    # 加载图像
    # --------------------------------------------------------

    def _load_image(self, path, is_visible=True):

        try:

            with Image.open(path) as img:

                if is_visible:
                    img = img.convert('RGB')
                else:
                    img = img.convert('L')

                img_tensor = TF.to_tensor(img).unsqueeze(0)

        except Exception as e:

            raise RuntimeError(
                f'图像加载失败: {path}\n错误: {e}'
            )

        return min_max_normalize(img_tensor)

    # --------------------------------------------------------
    # 数据增强
    # --------------------------------------------------------

    def _augment(self, img):

        # 随机旋转
        rot_times = random.randint(0, 3)

        img = torch.rot90(
            img,
            rot_times,
            [-2, -1]
        )

        # 随机水平翻转
        if random.random() > 0.5:
            img = torch.flip(img, [-1])

        # 随机垂直翻转
        if random.random() > 0.5:
            img = torch.flip(img, [-2])

        return img

    # --------------------------------------------------------
    # 获取数据
    # --------------------------------------------------------

    def __getitem__(self, idx):

        if self.is_train:

            # 当前图像块对应的图像名称及坐标
            image_name, top, left = self.patch_info[idx]

            # 加载可见光图像
            vis = self._load_image(
                os.path.join(self.vis_folder, image_name),
                True
            )

            # 加载红外图像
            ir = self._load_image(
                os.path.join(self.ir_folder, image_name),
                False
            )

            if vis.shape[-2:] != ir.shape[-2:]:

                raise RuntimeError(
                    f'图像尺寸不一致: {image_name}'
                )

            # 拼接红外与可见光
            combined = torch.cat(
                (vis, ir),
                dim=1
            )

            # 根据实际坐标同步裁剪
            patch = combined[
                :,
                :,
                top:top + self.img_size,
                left:left + self.img_size
            ]

            if patch.shape[-2:] != (
                self.img_size,
                self.img_size
            ):

                raise RuntimeError(
                    f'图像块尺寸错误: {image_name}'
                )

            # 同步数据增强
            if self.arg:
                patch = self._augment(patch)

            # 分离模态
            vis, ir = torch.split(
                patch,
                [3, 1],
                dim=1
            )

            vis = vis.squeeze(0)
            ir = ir.squeeze(0)

            # 转换为灰度图
            if not self.is_RGB:
                vis = rgb2y(vis)

            return {
                'VIS': vis,
                'IR': ir
            }

        # ----------------------------------------------------
        # 验证 / 测试模式
        # ----------------------------------------------------

        else:

            image_name = self.ir_list[idx]

            ir_path = os.path.join(
                self.ir_folder,
                image_name
            )

            vis_path = os.path.join(
                self.vis_folder,
                image_name
            )

            vis = self._load_image(
                vis_path,
                True
            )

            ir = self._load_image(
                ir_path,
                False
            )

            if vis.shape[-2:] != ir.shape[-2:]:

                raise RuntimeError(
                    f'验证图像尺寸不一致: {image_name}'
                )

            vis = vis.squeeze(0)
            ir = ir.squeeze(0)

            vis = rgb2y(vis)

            data = {
                'IR': ir,
                'VIS': vis,
                'IR_path': ir_path,
                'VIS_path': vis_path
            }

            if self.is_RGB:

                _, cbcr = bgr_to_ycrcb(vis_path)

                data['CBCR'] = cbcr

            return data

    # --------------------------------------------------------
    # 数据集长度
    # --------------------------------------------------------

    def __len__(self):

        if self.is_train:
            return len(self.patch_info)

        return len(self.ir_list)


# ============================================================
# 3. MSRS + LLVIP 混合数据集
# ============================================================

@DATASET_REGISTRY.register()
class WAVFUMixedDataset(Data.Dataset):

    def __init__(self, opt):

        super(WAVFUMixedDataset, self).__init__()

        if not opt.get('is_train', False):
            raise ValueError(
                'WAVFUMixedDataset 仅用于训练。'
            )

        mix_sources = opt.get('mix_sources', None)

        if mix_sources is None or len(mix_sources) != 2:

            raise ValueError(
                'mix_sources 必须包含 MSRS 和 LLVIP 两个数据集。'
            )

        self.datasets = []

        self.source_names = []

        self.source_indices = {}

        offset = 0

        # ----------------------------------------------------
        # 分别建立 MSRS 和 LLVIP 子数据集
        # ----------------------------------------------------

        for source in mix_sources:

            source_name = source['name']

            if source_name in self.source_indices:

                raise ValueError(
                    f'重复的数据集名称: {source_name}'
                )

            sub_opt = copy.deepcopy(opt)

            sub_opt.pop('mix_sources', None)
            sub_opt.pop('type', None)
            sub_opt.pop('name', None)

            # 写入当前子数据集的路径及独立配置
            sub_opt.update({
                k: v
                for k, v in source.items()
                if k != 'name'
            })

            sub_opt['is_train'] = True

            # 创建单个数据集
            dataset = WAVFUDataset(sub_opt)

            if len(dataset) == 0:

                raise RuntimeError(
                    f'{source_name} 没有有效训练数据。'
                )

            self.datasets.append(dataset)

            self.source_names.append(source_name)

            indices = list(
                range(
                    offset,
                    offset + len(dataset)
                )
            )

            self.source_indices[source_name] = indices

            offset += len(dataset)

            print(
                f'[Mixed Dataset] {source_name}: '
                f'{len(dataset)} patches'
            )

        # 合并索引空间
        self.concat_dataset = ConcatDataset(
            self.datasets
        )

        print('\n' + '=' * 65)
        print('[WAVFUMixedDataset] MSRS + LLVIP')
        print(
            f'MSRS patches: '
            f'{len(self.source_indices[self.source_names[0]])}'
        )
        print(
            f'LLVIP patches: '
            f'{len(self.source_indices[self.source_names[1]])}'
        )
        print(
            f'Total patches: {len(self.concat_dataset)}'
        )
        print('=' * 65)

    def __getitem__(self, idx):

        return self.concat_dataset[idx]

    def __len__(self):

        return len(self.concat_dataset)


# ============================================================
# 4. 均衡采样器
#
# 每个 Batch:
# MSRS  = 4
# LLVIP = 4
#
# 两个数据集每个 Batch 的采样数量始终相等。
# 较小的数据集在一个 Epoch 中允许重复采样。
# ============================================================

class WAVFUBalancedBatchSampler(Sampler):

    def __init__(
        self,
        dataset,
        batch_size=8,
        seed=42
    ):

        super(WAVFUBalancedBatchSampler, self).__init__()

        self.dataset = dataset

        self.batch_size = int(batch_size)

        self.seed = int(seed)

        self.epoch = 0

        if len(dataset.source_names) != 2:

            raise ValueError(
                '当前均衡采样器要求恰好两个数据集。'
            )

        if self.batch_size % 2 != 0:

            raise ValueError(
                'batch_size 必须能够被 2 整除。'
            )

        self.samples_per_source = self.batch_size // 2

        lengths = [
            len(dataset.source_indices[name])
            for name in dataset.source_names
        ]

        if min(lengths) <= 0:

            raise ValueError(
                '参与混合训练的数据集不能为空。'
            )

        # 以较大的数据集确定 Epoch 长度
        self.num_batches = math.ceil(
            max(lengths) / self.samples_per_source
        )

        print('\n[Balanced Sampler]')
        print(f'Batch size: {self.batch_size}')
        print(
            f'Each batch: '
            f'{self.samples_per_source} + '
            f'{self.samples_per_source}'
        )
        print(f'Batches per epoch: {self.num_batches}')

    def _make_epoch_indices(
        self,
        indices,
        target_count,
        generator
    ):

        result = []

        # 每轮随机排列；
        # 不足时重新随机排列并循环补充
        while len(result) < target_count:

            order = torch.randperm(
                len(indices),
                generator=generator
            ).tolist()

            result.extend(
                indices[i] for i in order
            )

        return result[:target_count]

    def __iter__(self):

        generator = torch.Generator()

        generator.manual_seed(
            self.seed + self.epoch
        )

        self.epoch += 1

        target_count = (
            self.num_batches * self.samples_per_source
        )

        # 为两个数据集分别生成均衡索引
        source_samples = {}

        for name in self.dataset.source_names:

            indices = self.dataset.source_indices[name]

            source_samples[name] = self._make_epoch_indices(
                indices,
                target_count,
                generator
            )

        # 随机排列 Batch 顺序
        batch_order = torch.randperm(
            self.num_batches,
            generator=generator
        ).tolist()

        for batch_idx in batch_order:

            batch = []

            start = (
                batch_idx * self.samples_per_source
            )

            end = (
                start + self.samples_per_source
            )

            # 每个数据集取相同数量的样本
            for name in self.dataset.source_names:

                batch.extend(
                    source_samples[name][start:end]
                )

            # 随机打乱当前 Batch 内部顺序
            order = torch.randperm(
                self.batch_size,
                generator=generator
            ).tolist()

            for index in order:

                yield batch[index]

    def __len__(self):

        return self.num_batches * self.batch_size

    def set_epoch(self, epoch):

        self.epoch = int(epoch)
