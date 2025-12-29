import math
import random

from PIL import Image
import blobfile as bf              # 支持本地 / 云端 / GCS / S3 等路径的统一文件接口
from mpi4py import MPI             # MPI 用于多卡/多进程数据切分
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    为给定数据集创建一个“无限生成器”，不断返回 (image, kwargs) 批次。

    返回的数据形式为：
        image: 形状为 (N, C, H, W) 的 float32 Tensor
        kwargs: 一个字典，用于存放条件信息（如类别 y）

    该接口是 DDBM / Diffusion 框架统一使用的数据入口。

    参数说明：
    ----------
    data_dir : str
        数据集所在目录（会递归读取所有图片）
    batch_size : int
        每个 batch 的样本数量
    image_size : int
        图像最终被裁剪/缩放到的尺寸（H = W = image_size）
    class_cond : bool
        是否启用类别条件（用于 class-conditional diffusion）
    deterministic : bool
        是否使用确定性顺序（一般用于评估/测试）
    random_crop : bool
        是否使用随机裁剪作为数据增强
    random_flip : bool
        是否随机左右翻转作为数据增强
    """

    # -----------------------------
    # 1️⃣ 检查数据目录是否合法
    # -----------------------------
    if not data_dir:
        raise ValueError("未指定数据目录 data_dir")

    # -----------------------------
    # 2️⃣ 递归列出所有图像文件
    # -----------------------------
    all_files = _list_image_files_recursively(data_dir)

    # -----------------------------
    # 3️⃣ 如果启用类别条件，则从文件名中解析类别
    # -----------------------------
    classes = None
    if class_cond:
        # 假设：文件名格式为 “类别名_xxx.png”
        class_names = [bf.basename(path).split("_")[0] for path in all_files]

        # 将类别名映射为整数标签
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    # -----------------------------
    # 4️⃣ 构建 Dataset 对象
    #     ⚠️ 注意：这里已经做了 MPI 数据切分
    # -----------------------------
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),      # 当前进程编号
        num_shards=MPI.COMM_WORLD.Get_size(), # 总进程数
        random_crop=random_crop,
        random_flip=random_flip,
    )

    # -----------------------------
    # 5️⃣ 构建 DataLoader
    # -----------------------------
    if deterministic:
        # 测试 / 验证阶段：不打乱顺序
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    else:
        # 训练阶段：打乱顺序
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )

    # -----------------------------
    # 6️⃣ 无限生成 batch（diffusion 项目常见写法）
    # -----------------------------
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """
    递归遍历目录，收集所有支持格式的图像文件路径
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]

        # 如果是图像文件，直接加入列表
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)

        # 如果是子目录，递归进入
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))

    return results


class ImageDataset(Dataset):
    """
    自定义 PyTorch Dataset
    负责：读取单张图像 → 预处理 → 返回 numpy array
    """

    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()

        # 目标图像分辨率（H=W=resolution）
        self.resolution = resolution

        # -----------------------------
        # MPI 切分数据（每个进程只读一部分）
        # -----------------------------
        self.local_images = image_paths[shard:][::num_shards]

        # 同步切分类别标签
        self.local_classes = (
            None if classes is None else classes[shard:][::num_shards]
        )

        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        # 当前进程可见的样本数量
        return len(self.local_images)

    def __getitem__(self, idx):
        # -----------------------------
        # 1️⃣ 读取图像文件
        # -----------------------------
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # -----------------------------
        # 2️⃣ 强制转为单通道灰度图
        #     ⚠️ 这是你 MRI/CT 项目中特别重要的一步
        # -----------------------------
        pil_image = pil_image.convert("L")

        # -----------------------------
        # 3️⃣ 裁剪（随机 or 中心）
        # -----------------------------
        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        # -----------------------------
        # 4️⃣ 随机左右翻转（数据增强）
        # -----------------------------
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        # -----------------------------
        # 5️⃣ 构造条件字典（如类别 y）
        # -----------------------------
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(
                self.local_classes[idx], dtype=np.int64
            )

        # -----------------------------
        # 6️⃣ 像素值归一化到 [-1, 1]
        # -----------------------------
        arr = arr.astype(np.float32) / 127.5 - 1.0

        # -----------------------------
        # 7️⃣ 确保是 (H, W, C)
        # -----------------------------
        if arr.ndim == 2:
            arr = arr[:, :, None]  # (H, W, 1)

        # -----------------------------
        # 8️⃣ 转为 PyTorch 习惯的 (C, H, W)
        # -----------------------------
        arr = np.transpose(arr, [2, 0, 1])

        return arr, out_dict


def center_crop_arr(pil_image, image_size):
    """
    中心裁剪：
    - 先多次 BOX 下采样（减少 aliasing）
    - 再双三次插值缩放
    - 最后中心裁剪
    """

    # 当图像尺寸远大于目标尺寸时，反复二分下采样
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.BOX,
        )

    # 缩放到目标尺寸比例
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC,
    )

    # 转为 numpy 数组
    arr = np.array(pil_image)

    # 中心裁剪
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2

    return arr[
        crop_y : crop_y + image_size,
        crop_x : crop_x + image_size,
    ]


def random_crop_arr(
    pil_image,
    image_size,
    min_crop_frac=0.8,
    max_crop_frac=1.0,
):
    """
    随机裁剪：
    - 随机决定裁剪比例
    - 再 resize + 随机位置裁剪
    """

    # 随机决定裁剪后较短边的尺寸
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(
        min_smaller_dim_size,
        max_smaller_dim_size + 1,
    )

    # 同样先做多次 BOX 下采样
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.BOX,
        )

    # resize 到目标随机尺度
    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC,
    )

    arr = np.array(pil_image)

    # 在图像中随机选择裁剪位置
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)

    return arr[
        crop_y : crop_y + image_size,
        crop_x : crop_x + image_size,
    ]
