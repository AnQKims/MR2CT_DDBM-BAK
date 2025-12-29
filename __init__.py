# =========================
# 基础库与 PyTorch 相关导入
# =========================
import torch                          # PyTorch 核心库
import torchvision                    # torchvision（部分模型/工具，虽然这里几乎没用到）
import torchvision.transforms as transforms  # 图像变换工具

import os                             # 文件路径操作
from collections import defaultdict   # 默认字典（当前文件未使用，可能为历史遗留）
import numpy as np                    # 数值计算
from tqdm import tqdm                 # 进度条（当前文件未使用）
# import imageio                      # 未使用，注释掉

import math                           # 数学函数
import random                         # Python 随机数
import torch.distributed as dist      # PyTorch 分布式训练（DDP）

# PyTorch 数据加载相关
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset


# =========================================================
# 数据归一化函数（Diffusion / Score-based 模型非常关键）
# =========================================================
def get_data_scaler(config):
  """
  根据 config 决定是否对数据进行中心化（centered）

  假设：输入数据原本已经被归一化到 [0, 1]
  """
  if config.data.centered:
    # 若开启 centered：
    #   将数据从 [0, 1] 映射到 [-1, 1]
    #   这是 diffusion 模型中最常见的数据分布
    return lambda x: x * 2. - 1.
  else:
    # 若不开启 centered：
    #   数据保持在 [0, 1]
    return lambda x: x


def get_data_inverse_scaler(config):
  """
  get_data_scaler 的逆操作
  用于将模型输出还原回原始像素范围
  """
  if config.data.centered:
    # [-1, 1] → [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


# =========================================================
# Uniform Dequantization（将离散像素转为连续）
# =========================================================
class UniformDequant(object):
  def __call__(self, x):
    """
    对输入张量 x 加入 [0, 1/256) 的均匀噪声

    作用：
    - 原始图像像素是 8-bit 离散值
    - diffusion / likelihood 建模假设连续分布
    - 通过加噪声实现“去量化（dequantization）”
    """
    return x + torch.rand_like(x) / 256


# =========================================================
# RASampler：重复增强采样器（来自 DeiT）
# =========================================================
class RASampler(Sampler):
    """
    Repeated Augmentation Sampler（重复增强采样器）

    主要用途：
    - 分布式训练（多 GPU）
    - 同一张图像通过不同增强
    - 分配给不同 GPU
    - 提高数据利用率

    本项目中并非 diffusion 核心逻辑，属于通用组件
    """

    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, seed=0, repetitions=3):
        # 如果没有显式指定 GPU 数量，则从分布式环境中获取
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        # 如果没有指定当前进程 rank，则从分布式环境中获取
        if rank is None:
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas   # GPU 总数
        self.rank = rank                   # 当前 GPU 编号
        self.epoch = 0                     # 当前 epoch

        # 每个 GPU 理论上需要加载的样本数量
        self.num_samples = int(
            math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas)
        )

        # 全局样本数量（所有 GPU 总和）
        self.total_size = self.num_samples * self.num_replicas

        # 实际选用的样本数（做 256 对齐，经验设定）
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.num_replicas)
        )

        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        # 是否对样本顺序进行打乱
        if self.shuffle:
            # 使用 epoch + seed 保证可复现
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # 每个样本重复 repetitions 次（重复增强核心）
        indices = [ele for ele in indices for _ in range(self.repetitions)]

        # 补齐到 total_size，保证可整除 GPU 数量
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # 按 rank 切分，每个 GPU 只取自己那部分
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # 返回本 GPU 使用的 index
        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        # 每个 epoch 更新，用于 shuffle
        self.epoch = epoch


# =========================================================
# InfiniteBatchSampler：无限 batch 采样器（Diffusion 核心）
# =========================================================
class InfiniteBatchSampler(Sampler):
    """
    无限 batch 采样器：
    - 不以 epoch 为核心概念
    - 不断 yield batch index
    - 非常适合 diffusion 这类按 step 训练的模型
    """

    def __init__(self, dataset_len, batch_size,
                 seed=0, filling=False, shuffle=True, drop_last=False):
        self.dataset_len = dataset_len     # 数据集长度
        self.batch_size = batch_size       # batch 大小

        # 每个 epoch 需要多少次迭代
        self.iters_per_ep = (
            dataset_len // batch_size
            if drop_last
            else (dataset_len + batch_size - 1) // batch_size
        )

        # 一个 epoch 内最大可用 index 数
        self.max_p = self.iters_per_ep * batch_size

        self.filling = filling             # 是否补齐 batch
        self.shuffle = shuffle             # 是否 shuffle
        self.epoch = 0
        self.seed = seed

        # 预生成 index 序列
        self.indices = self.gener_indices()

    def gener_indices(self):
        # 根据是否 shuffle 生成 index 序列
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(self.dataset_len, generator=g).numpy()
        else:
            indices = torch.arange(self.dataset_len).numpy()

        # batch 不整除时是否补齐
        tails = self.batch_size - (self.dataset_len % self.batch_size)
        if tails != self.batch_size and self.filling:
            tails = indices[:tails]
            np.random.shuffle(indices)
            indices = np.concatenate((indices, tails))

        # 转为 tuple（比 numpy array 更快）
        return tuple(indices.tolist())

    def __iter__(self):
        # 无限循环，不断生成 batch
        self.epoch = 0
        while True:
            self.epoch += 1
            p = 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            # 一个 epoch 后重新生成 index
            if self.shuffle:
                self.indices = self.gener_indices()

    def __len__(self):
        return self.iters_per_ep


# =========================================================
# DistInfiniteBatchSampler：分布式无限 batch 采样器
# =========================================================
class DistInfiniteBatchSampler(InfiniteBatchSampler):
    """
    InfiniteBatchSampler 的分布式版本
    - 支持 DDP
    - 全局 batch → 自动拆分到各 GPU
    """

    def __init__(self, world_size, rank,
                 dataset_len, glb_batch_size,
                 seed=0, repeated_aug=0,
                 filling=False, shuffle=True):

        # 全局 batch 必须能被 GPU 数整除
        assert glb_batch_size % world_size == 0

        self.world_size = world_size        # GPU 数
        self.rank = rank                    # 当前 GPU 编号
        self.dataset_len = dataset_len

        self.glb_batch_size = glb_batch_size
        self.batch_size = glb_batch_size // world_size

        self.iters_per_ep = (dataset_len + glb_batch_size - 1) // glb_batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.repeated_aug = repeated_aug
        self.epoch = 0
        self.seed = seed

        self.indices = self.gener_indices()

    def gener_indices(self):
        # 全局最大 index 数
        global_max_p = self.iters_per_ep * self.glb_batch_size

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            global_indices = torch.randperm(self.dataset_len, generator=g)

            # 可选：重复增强
            if self.repeated_aug > 1:
                global_indices = (
                    global_indices[:(self.dataset_len + self.repeated_aug - 1) // self.repeated_aug]
                    .repeat_interleave(self.repeated_aug, dim=0)
                    [:global_max_p]
                )
        else:
            global_indices = torch.arange(self.dataset_len)

        # 补齐 index 数量
        filling = global_max_p - global_indices.shape[0]
        if filling > 0 and self.filling:
            global_indices = torch.cat((global_indices, global_indices[:filling]))

        # 转为 tuple
        global_indices = tuple(global_indices.numpy().tolist())

        # 将全局 index 按 GPU 切分
        seps = torch.linspace(0, len(global_indices), self.world_size + 1, dtype=torch.int)
        local_indices = global_indices[seps[self.rank]:seps[self.rank + 1]]

        self.max_p = len(local_indices)
        return local_indices


# =========================================================
# load_data：整个 datasets 模块的唯一对外接口
# =========================================================
def load_data(
    data_dir,
    dataset,
    batch_size,
    image_size,
    deterministic=False,
    include_test=False,
    seed=42,
    num_workers=2,
):
  """
  根据配置：
  - 构建 Dataset
  - 构建 Sampler
  - 返回 DataLoader
  """

  root = data_dir

  # -------------------------
  # 根据数据集名称选择 Dataset
  # -------------------------
  if dataset == 'edges2handbags':
    # 使用 aligned_dataset.py 中的 EdgesDataset
    from .aligned_dataset import EdgesDataset

    trainset = EdgesDataset(
        dataroot=root,
        train=True,
        img_size=image_size,
        random_crop=True,
        random_flip=True
    )

    valset = EdgesDataset(
        dataroot=root,
        train=True,
        img_size=image_size,
        random_crop=False,
        random_flip=False
    )

    if include_test:
      testset = EdgesDataset(
          dataroot=root,
          train=False,
          img_size=image_size,
          random_crop=False,
          random_flip=False
      )

  elif dataset == 'diode':
    from .aligned_dataset import DIODE

    trainset = DIODE(
        dataroot=root,
        train=True,
        img_size=image_size,
        random_crop=True,
        random_flip=True,
        disable_cache=True
    )

    valset = DIODE(
        dataroot=root,
        train=True,
        img_size=image_size,
        random_crop=False,
        random_flip=False,
        disable_cache=True
    )

    if include_test:
      testset = DIODE(
          dataroot=root,
          train=False,
          img_size=image_size,
          random_crop=False,
          random_flip=False
      )

  # -------------------------
  # 训练 DataLoader（无限 sampler）
  # -------------------------
  loader = DataLoader(
      dataset=trainset,
      num_workers=num_workers,
      pin_memory=True,
      batch_sampler=DistInfiniteBatchSampler(
          dataset_len=len(trainset),
          glb_batch_size=batch_size * dist.get_world_size(),
          seed=seed,
          shuffle=not deterministic,
          filling=True,
          rank=dist.get_rank(),
          world_size=dist.get_world_size(),
      )
  )

  # -------------------------
  # 验证 DataLoader（标准分布式采样）
  # -------------------------
  sampler = torch.utils.data.DistributedSampler(
      valset,
      num_replicas=dist.get_world_size(),
      rank=dist.get_rank(),
      shuffle=False,
      drop_last=False
  )

  val_loader = torch.utils.data.DataLoader(
      valset,
      batch_size=batch_size,
      sampler=sampler,
      num_workers=num_workers,
      drop_last=False
  )

  # -------------------------
  # 是否返回测试集
  # -------------------------
  if include_test:
    sampler = torch.utils.data.DistributedSampler(
        testset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
        drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False
    )
    return loader, val_loader, test_loader
  else:
    return loader, val_loader
