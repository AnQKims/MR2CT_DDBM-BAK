
MODEL_PATH="/home/un/world/wd/DDBM-gray/workdir/mri2ct_ddbm_320_1225_LowSigma-CTsupervised2_ve/ema_0.9999_130000.pt"
CHURN_STEP_RATIO=0.33 # 官方推荐
GUIDANCE=2.0  # Bridge Guidance 强度

# | guidance | 行为                   |
# | -------- | --------------------- |
# | 0        | 完全无条件(MRI→MRI)    |
# | 1        | 标准 bridge(训练同分布)|
# | 2~4      | CT 风格更强(推荐)      |
# | >5       | 易过拟合 / 伪影        |

SPLIT=test   # train, test, val 数据集标签


source ./args.sh $DATASET_NAME $PRED 


N=40  # 推理步数
GEN_SAMPLER=heun # 采样器类型（Heun solver）
BS=2 # batch size
NGPU=1 # 使用 GPU 数量（通过 mpiexec 控制）

mpiexec -n $NGPU python scripts/image_sample.py --exp=$EXP \
--batch_size $BS --churn_step_ratio $CHURN_STEP_RATIO --steps $N --sampler $GEN_SAMPLER \
--model_path $MODEL_PATH --attention_resolutions $ATTN  --class_cond False --pred_mode $PRED \
${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
${COND:+ --condition_mode="${COND}"} --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
--dropout 0.1 --image_size $IMG_SIZE --num_channels $NUM_CH --num_head_channels 64 --num_res_blocks $NUM_RES_BLOCKS \
--resblock_updown True --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --use_scale_shift_norm True \
--weight_schedule bridge_karras --data_dir=$DATA_DIR \
 --dataset=$DATASET --rho 7 --upscale=False ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
 ${UNET:+ --unet_type="${UNET}"} ${SPLIT:+ --split="${SPLIT}"} ${GUIDANCE:+ --guidance="${GUIDANCE}"}
 

