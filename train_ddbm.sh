DATASET_NAME=$1
PRED=$2
CKPT=$3


source ./args.sh $DATASET_NAME $PRED

FREQ_SAVE_ITER=10000   
SAVE_ITER=10000  # 模型保存间隔/Step
TEST_ITER=50     # 可视化间隔/Step
NGPU=1

# """
# === FREQ_SAVE_ITER ===

# 高频「抢救用」checkpoint，防止训练过程中：机器死机、OOM、被 kill、集群抢占 / preemption

# 在TrainLoop.run_loop()用
# if took_step and self.step % self.save_interval_for_preemption == 0:
#     self.save(for_preemption=True)

# 在 workdir/ 中生成：
# freq_model_020000.pt
# freq_ema_0.9999_020000.pt
# freq_opt_020000.pt
# 最多保留 3 个（老的会被删），不会触发可视化，不会跑验证集DATASET_NAME=$1



# === SAVE_ITER ===

# 训练中的“正式节奏点”，它通常会触发：

# 保存模型、跑一次验证集、保存 train / val 可视化图片、记录一轮完整日志
# if self.step > 0 and self.step % self.save_interval == 0:
#     self.save()

#     test_batch, test_cond, _ = next(iter(self.test_data))
#     self.run_test_step(test_batch, test_cond)

#     self._visualize(batch, cond, self.step, split="train")
#     self._visualize(test_batch, test_cond, self.step, split="val")



# """


mpiexec -n $NGPU python scripts/ddbm_train.py --exp=$EXP \
 --attention_resolutions $ATTN --class_cond False --use_scale_shift_norm True \
  --dropout 0.1 --ema_rate 0.9999 --batch_size $BS \
   --image_size $IMG_SIZE --lr 0.0001 --num_channels $NUM_CH --num_head_channels 64 \
    --num_res_blocks $NUM_RES_BLOCKS --resblock_updown True ${COND:+ --condition_mode="${COND}"} ${MICRO:+ --microbatch="${MICRO}"} \
     --pred_mode=$PRED  --schedule_sampler $SAMPLER ${UNET:+ --unet_type="${UNET}"} \
    --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --weight_decay 0.0 --weight_schedule bridge_karras \
     ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
      --data_dir=$DATA_DIR --dataset=$DATASET ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
      --num_workers=8  --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
      --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER --debug=False \
      --test_interval=$TEST_ITER \
      ${CKPT:+ --resume_checkpoint="${CKPT}"} 
