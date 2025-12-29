import os.path
import torch
import random
import numpy as np
import scipy.io as sio
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from .image_folder import make_dataset
from PIL import Image

import torchvision
import blobfile as bf

from glob import glob


# ============================================================
# 随机裁剪 & 翻转参数生成
# ============================================================
def get_params(size, resize_size, crop_size):
    """
    根据输入图像尺寸，生成一组【共享的】随机裁剪与翻转参数，
    用于 paired image（MRI / CT）保持空间对齐。
    """
    w, h = size  # 原始图像宽高

    # 记录短边和长边
    ss, ls = min(w, h), max(w, h)
    width_is_shorter = (w == ss)

    # 按比例将短边缩放到 resize_size
    ls = int(resize_size * ls / ss)
    ss = resize_size

    # 计算 resize 后的宽高
    new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    # 随机裁剪左上角坐标
    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    # 是否进行左右翻转
    flip = random.random() > 0.5

    return {
        'crop_pos': (x, y),
        'flip': flip
    }


# ============================================================
# 构造 torchvision transform（resize / flip / toTensor）
# ============================================================
def get_transform(
    params,
    resize_size,
    crop_size,
    method=Image.BICUBIC,
    flip=True,
    crop=True,
    totensor=True
):
    """
    构造一套 transform，用于 MRI / CT 共用，
    保证二者在数据增强后仍然严格对齐。
    """
    transform_list = []

    # resize 到固定尺寸（正方形）
    transform_list.append(
        transforms.Lambda(
            lambda img: __scale(img, crop_size, method)
        )
    )

    # 按 get_params 中的 flip 决定是否左右翻转
    if flip:
        transform_list.append(
            transforms.Lambda(
                lambda img: __flip(img, params['flip'])
            )
        )

    # 转为 Tensor（[C,H,W]，值域 [0,1]）
    if totensor:
        transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


# ============================================================
# 常规 tensor + normalize 组合（备用）
# ============================================================
def get_tensor(normalize=True, toTensor=True):
    transform_list = []

    if toTensor:
        transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        )

    return transforms.Compose(transform_list)


def normalize():
    """标准 [-1,1] 归一化"""
    return transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))


# ============================================================
# resize 操作（同时支持 PIL Image 和 Tensor）
# ============================================================
def __scale(img, target_width, method=Image.BICUBIC):
    """
    将图像 resize 为 (target_width, target_width)
    """
    if isinstance(img, torch.Tensor):
        # Tensor: 使用 torch 的 interpolate
        return torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(target_width, target_width),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
    else:
        # PIL Image
        return img.resize((target_width, target_width), method)


# ============================================================
# 左右翻转（Tensor / PIL 通用）
# ============================================================
def __flip(img, flip):
    if flip:
        if isinstance(img, torch.Tensor):
            return img.flip(-1)
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_flip(img, flip):
    """外部接口（实际调用 __flip）"""
    return __flip(img, flip)


# ============================================================
# EdgesDataset —— 拼接 MRI–CT paired dataset
# ============================================================
class EdgesDataset(torch.utils.data.Dataset):
    """
    用于【拼接形式的 paired 图像】的数据集

    假设：
        - 每张图片是左右拼接：
            [ MRI | CT ]
        - 左半边 = 输入（MRI）
        - 右半边 = 目标（CT）
    """

    def __init__(
        self,
        dataroot,
        train=True,
        img_size=256,
        random_crop=False,
        random_flip=True
    ):
        super().__init__()

        # 训练 / 验证路径
        if train:
            self.train_dir = os.path.join(dataroot, 'train')
            self.AB_paths = sorted(make_dataset(self.train_dir))
        else:
            self.test_dir = os.path.join(dataroot, 'val')
            self.AB_paths = make_dataset(self.test_dir)

        self.crop_size = img_size
        self.resize_size = img_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train = train

    def __getitem__(self, index):
        """
        返回：
            train:
                CT, MRI, index
            test:
                CT, MRI, index, path
        """
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # ==============================
        # 1️⃣ 拆分拼接图像
        # ==============================
        w, h = AB.size
        w2 = int(w / 2)

        A = AB.crop((0, 0, w2, h))  # 左半边：MRI
        B = AB.crop((w2, 0, w, h))  # 右半边：CT

        # ==============================
        # 2️⃣ 生成共享 transform 参数
        # ==============================
        params = get_params(A.size, self.resize_size, self.crop_size)
        transform_image = get_transform(
            params,
            self.resize_size,
            self.crop_size,
            crop=self.random_crop,
            flip=self.random_flip
        )

        # ==============================
        # 3️⃣ 同步数据增强
        # ==============================
        A = transform_image(A)  # [3, H, W]
        B = transform_image(B)  # [3, H, W]

        # ==============================
        # 4️⃣ 强制单通道（医学图像）
        # ==============================
        A = A.mean(dim=0, keepdim=True)  # MRI
        B = B.mean(dim=0, keepdim=True)  # CT

        # ==============================
        # 5️⃣ 返回（注意顺序）
        # ==============================
        if not self.train:
            return B, A, index, AB_path
        else:
            return B, A, index

    def __len__(self):
        return len(self.AB_paths)


# ============================================================
# DIODE Dataset（非 MRI–CT，用于法向图等）
# ============================================================
class DIODE(torch.utils.data.Dataset):
    """
    DIODE 数据集（RGB + normal map），
    与 MRI–CT 无关，属于另一套 paired 数据逻辑。
    """

    def __init__(
        self,
        dataroot,
        train=True,
        img_size=256,
        random_crop=False,
        random_flip=True,
        down_sample_img_size=0,
        cache_name='cache',
        disable_cache=False
    ):
        super().__init__()

        self.image_root = os.path.join(dataroot, 'train' if train else 'val')
        self.crop_size = img_size
        self.resize_size = img_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train = train

        # 过滤非图像文件
        self.filenames = [
            l for l in os.listdir(self.image_root)
            if not l.endswith('.pth')
            and not l.endswith('_depth.png')
            and not l.endswith('_normal.png')
        ]

        # 可选缓存
        self.cache_path = os.path.join(
            self.image_root, cache_name + f'_{img_size}.pth'
        )

        if os.path.exists(self.cache_path) and not disable_cache:
            self.cache = torch.load(self.cache_path)
            self.scale_factor = self.cache['scale_factor']
            print(f'Loaded cache from {self.cache_path}')
        else:
            self.cache = None

    def __getitem__(self, index):
        fn = self.filenames[index]

        img_path = os.path.join(self.image_root, fn)
        label_path = os.path.join(self.image_root, fn[:-4] + '_normal.png')

        # 读取 RGB 图像
        with bf.BlobFile(img_path, "rb") as f:
            pil_image = Image.open(f).convert("RGB")

        # 读取 normal map
        with bf.BlobFile(label_path, "rb") as f:
            pil_label = Image.open(f).convert("RGB")

        # 共享 transform 参数
        params = get_params(
            pil_image.size,
            self.resize_size,
            self.crop_size
        )

        transform_label = get_transform(
            params,
            self.resize_size,
            self.crop_size,
            method=Image.NEAREST,
            crop=False,
            flip=self.random_flip
        )

        transform_image = get_transform(
            params,
            self.resize_size,
            self.crop_size,
            crop=False,
            flip=self.random_flip
        )

        cond = transform_label(pil_label)
        img = transform_image(pil_image)

        if not self.train:
            return img, cond, index, fn
        else:
            return img, cond, index

    def __len__(self):
        if self.cache is not None:
            return len(self.cache['img'])
        else:
            return len(self.filenames)



class MRICTTestDataset(Dataset):
    """
    用于推理：
    只提供 MRI
    """

    def __init__(self, root, image_size=320):
        self.root = root
        self.image_size = image_size
        self.paths = sorted(os.listdir(root))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.paths[idx])

        img = Image.open(path).convert("L")
        w, h = img.size
        w2 = w // 2

        mri = img.crop((0, 0, w2, h))
        mri = to_tensor(mri)

        mri, pad_info = resize_and_pad(mri, self.image_size)

        return {
            "cond": mri,
            "pad_info": pad_info,
            "path": path
        }

