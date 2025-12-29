# =========================
# 基础训练参数
# =========================

BS=2             

DATASET_NAME=e2h       # 使用的数据集类型
                       # e2h   -> Edges2Handbags（成对拼接数据）
                       # diode -> DIODE（条件图像数据）

PRED=ve                # diffusion 预测方式
                       # ve         : 预测噪声 ε（最稳定，强烈推荐）
                       # vp         : 预测 x0（医学数据不太稳）
                       # ve_simple  : 简化 ve
                       # vp_simple  : 简化 vp

NGPU=1                 # 使用 GPU 数量（mpiexec -n）


# =========================
# Diffusion 核心噪声参数
# =========================

SIGMA_MAX=80.0         # 最大噪声强度 σ_max
                       # 值越大，模型越“扩散”
                       # 医学图像通常不需要特别大

SIGMA_MIN=0.002        # 最小噪声强度 σ_min
                       # 控制生成细节质量

SIGMA_DATA=0.5         # 数据标准差（非常重要）
                       # ≈ 数据的真实强度分布尺度
                       # MRI/CT 归一化到 [0,1] 时，0.5 是合理默认值

COV_XY=0               # 是否引入 x,y 噪声相关性（一般不用）

IN_CHANNELS = 2 
OUT_CHANNELS = 1


# =========================
# UNet 网络结构参数
# =========================

NUM_CH=256             # UNet 基础通道数（512 分辨率推荐 256）
ATTN=32,16,8           # 在哪些分辨率层使用 Self-Attention
                       # 512 → 32 / 16 / 8 是非常合理的设置

SAMPLER=real-uniform   # 时间步采样策略
                       # real-uniform 是官方推荐

NUM_RES_BLOCKS=2       # 每个尺度的残差块数量
                       # 512 分辨率下 2 比 3 更稳

USE_16FP=True          # 是否使用 FP16（强烈建议开启）
ATTN_TYPE=vanilla       # attention 实现方式（flash 更快）


# =========================
# 数据集相关设置
# =========================

if [[ $DATASET_NAME == "e2h" ]]; then
    # ---------- MRI → CT 走 e2h 路线 ----------
    # DATA_DIR=/home/un/world/wd/DDBM-gray/DATA_DIR/320
    DATA_DIR=/home/un/world/wd/DDBM-gray/1TestResult/result

    # DATA_DIR 结构应为：
    # DATA_DIR/
    #   ├── train/
    #   │   ├── 000000.png
    #   │   ├── ...
    #   └── val/
    #       ├── 000000.png
    #       ├── ...

    DATASET=edges2handbags   # 告诉 DDBM 使用 paired dataset
    
    IMG_SIZE=320             # 实际图像尺寸（MRI / CT 是 512）

    NUM_CH=256
    NUM_RES_BLOCKS=2

    # 实验名（用于 checkpoint / log）
    EXP="mri2ct_ddbm_320_1225_LowSigma-CTsupervised2"

    # SAVE_ITER=20000          # 每 2w step 保存一次模型


elif [[ $DATASET_NAME == "diode" ]]; then
    # ---------- 如果你将来用 DIODE ----------
    DATA_DIR=YOUR_DATASET_PATH
    DATASET=diode
    IMG_SIZE=256

    SIGMA_MAX=20.0
    SIGMA_MIN=0.0005

    EXP="diode${IMG_SIZE}_${NUM_CH}d"
    # SAVE_ITER=20000
fi


# =========================
# Diffusion 预测模式设置
# =========================

if [[ $PRED == "ve" ]]; then
    # ===== 预测噪声 ε（最推荐）=====
    EXP+="_ve"
    COND=concat          # 条件方式：拼接（MRI + noisy CT）

elif [[ $PRED == "vp" ]]; then
    # ===== 预测 x0（不太推荐医学数据）=====
    EXP+="_vp"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001

elif [[ $PRED == "ve_simple" ]]; then
    EXP+="_ve_simple"
    COND=concat

elif [[ $PRED == "vp_simple" ]]; then
    EXP+="_vp_simple"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001

else
    echo "Not supported prediction mode"
    exit 1
fi



