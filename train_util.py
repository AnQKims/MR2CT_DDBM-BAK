import copy
import functools
import os
import time
from pathlib import Path
import torch
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
from PIL import Image
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from PIL import Image
from ddbm.random_util import get_generator
from .fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
import numpy as np
from ddbm.script_util import NUM_CLASSES

from ddbm.karras_diffusion import karras_sample

import glob 
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

import wandb

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        train_data,
        test_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        test_interval,
        save_interval,
        save_interval_for_preemption,
        resume_checkpoint,
        workdir,
        # vis_interval=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        total_training_steps=10000000,
        augment_pipe=None,
        **sample_kwargs,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = train_data
        self.test_data = test_data
        self.image_size = model.image_size
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.workdir = workdir
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.save_interval_for_preemption = save_interval_for_preemption
        self.resume_checkpoint = resume_checkpoint
        # self.vis_interval = vis_interval
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.total_training_steps = total_training_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()


        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step

        self.generator = get_generator(sample_kwargs['generator'], self.batch_size,42)
        self.sample_kwargs = sample_kwargs

        self.augment = augment_pipe
##↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ===== visualization directories =====
        self.image_dir = os.path.join(self.workdir, "image")
        self.image_dir_train = os.path.join(self.image_dir, "train")
        self.image_dir_val = os.path.join(self.image_dir, "val")

        if dist.get_rank() == 0:
            # logger.log(f"[Init] Visualization interval: {self.vis_interval}")
            os.makedirs(self.image_dir_train, exist_ok=True)
            os.makedirs(self.image_dir_val, exist_ok=True)
##↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑   
    

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                logger.log('Resume step: ', self.resume_step)
                
            self.model.load_state_dict(
                # dist_util.load_state_dict(
                #     resume_checkpoint, map_location=dist_util.dev()
                # ),
                th.load(resume_checkpoint, map_location=dist_util.dev()),
            )
        
            dist.barrier()

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            # state_dict = dist_util.load_state_dict(
            #     ema_checkpoint, map_location=dist_util.dev()
            # )
            state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

            dist.barrier()
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if main_checkpoint.split('/')[-1].startswith("latest"):
            prefix = 'latest_'
        else:
            prefix = ''
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"{prefix}opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            # state_dict = dist_util.load_state_dict(
            #     opt_checkpoint, map_location=dist_util.dev()
            # )
            state_dict = th.load(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)
            dist.barrier()

    # def preprocess(self, x):
    #     if x.shape[1] == 3:
    #         x =  x * 2 - 1
    def preprocess(self, x):
        # dataset 已经是 [-1,1]
        return x


##↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

    def run_loop(self):

        while True:
            for item in self.data:
                # ===============================
                # 1. 终止条件（保持原逻辑）
                # ===============================
                if not (not self.lr_anneal_steps or self.step < self.total_training_steps):
                    if (self.step - 1) % self.save_interval != 0:
                        self.save()
                    return

                # ===============================
                # 2. 解包 data（✅ 核心修改点）
                # ===============================
                assert isinstance(item, (list, tuple)), \
                    f"Expected list/tuple from dataloader, got {type(item)}"
                assert len(item) == 3, \
                    f"Expected 3 elements [CT, MRI, idx], got {len(item)}"

                # 🔥 关键：按 dataset 实际语义重新命名
                ct  = item[0]    # ✅ CT
                mri = item[1]    # ✅ MRI
                z   = item[2]    # index / noise（未直接用）

                # ===============================
                # 3. batch 定义（模型预测目标）
                # ===============================
                batch = self.preprocess(ct)   # ✅ GT = CT

                if self.augment is not None:
                    batch, _ = self.augment(batch)

                # ===============================
                # 4. cond 构造（模型条件）
                # ===============================
                cond = {
                    "xT": self.preprocess(mri),   # ✅ MRI 作为条件
                    "target": self.preprocess(ct) # ✅ CT 作为监督
                }

                # ===============================
                # 5. debug（只打印一次）
                # ===============================
                if self.step == 0:
                    print("=== Sanity Check ===")
                    print("MRI (xT) range:", cond["xT"].min().item(), cond["xT"].max().item())
                    print("CT  (batch) range:", batch.min().item(), batch.max().item())
                    print("MRI shape:", cond["xT"].shape)
                    print("CT  shape:", batch.shape)

                # ===============================
                # 6. 训练一步
                # ===============================
                took_step = self.run_step(batch, cond)

                # ===============================
                # 7. 可视化 + test（保持不变）
                # ===============================
                if took_step and self.step % self.test_interval == 0:
                    test_batch, test_cond, _ = next(iter(self.test_data))

                    # ⚠️ test_data 也遵循同样的语义
                    test_ct  = test_batch
                    test_mri = test_cond if isinstance(test_cond, th.Tensor) else test_cond["xT"]

                    test_batch = self.preprocess(test_ct)
                    test_cond = {"xT": self.preprocess(test_mri)}

                    self.run_test_step(test_batch, test_cond)

                    self._visualize(batch, cond, self.step, split="train")
                    self._visualize(test_batch, test_cond, self.step, split="val")

                    logs = logger.dumpkvs()
                    if dist.get_rank() == 0:
                        wandb.log(logs, step=self.step)

                # ===============================
                # 8. 保存模型（保持原逻辑）
                # ===============================
                if took_step and self.step % self.save_interval == 0:
                    self.save()
                    logs = logger.dumpkvs()
                    if dist.get_rank() == 0:
                        wandb.log(logs, step=self.step)

                if took_step and self.step % self.save_interval_for_preemption == 0:
                    self.save(for_preemption=True)


##↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑     


    
##↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

    def run_step(self, batch, cond):
        start_time = time.time()

        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)

        step_time = time.time() - start_time

        if took_step:
            self.step += 1
            self._update_ema()

            # ===== 记录 step 时间 =====
            logger.logkv("step_time_sec", step_time)

        self._anneal_lr()
        self.log_step()
        return took_step
    
    # def run_step(self, batch, cond):
    #     self.forward_backward(batch, cond)
    #     took_step = self.mp_trainer.optimize(self.opt)
    #     if took_step:
    #         self.step += 1
    #         self._update_ema()
    #     self._anneal_lr()
    #     self.log_step()
    #     return took_step
##↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑




    def run_test_step(self, batch, cond):
        with th.no_grad():
            self.forward_backward(batch, cond, train=False)

##↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    @th.no_grad()
    def _visualize(self, batch, cond, step, split="train"):
        """
        Visualization for conditional diffusion (MRI -> CT).
        Save image as: [MRI | Pred_CT | GT_CT]
        """

        logger.log(f"[visualize] ENTER step={step}, split={split}")

        if dist.get_rank() != 0:
            return

        device = dist_util.dev()
        self.model.eval()

        assert isinstance(cond, dict)
        assert "xT" in cond, f"cond keys = {cond.keys()}"

        # =========================================================
        # 1️⃣ 取 1 张样本
        # =========================================================
        batch = batch[:1].to(device)            # GT CT
        xT = cond["xT"][:1].to(device)          # MRI (condition)

        # =========================================================
        # 2️⃣ model_kwargs（只传模型真正需要的条件）
        # =========================================================
        model_kwargs = {"xT": xT}

        # =========================================================
        # 3️⃣ diffusion 采样
        # =========================================================
        # 条件扩散：x_0 = MRI
        x_0 = xT
        x_T = torch.randn_like(x_0)

        out = karras_sample(
            self.diffusion,
            self.model,
            x_T,
            x_0,
            steps=50,
            device=device,
            model_kwargs=model_kwargs,
            sigma_min=0.002,
            sigma_max=80,
            sampler="heun",
        )

        sample = out[0] if isinstance(out, tuple) else out  # Pred CT

        # =========================================================
        # 4️⃣ 医学图像专用可视化函数（关键）
        # =========================================================
        def to_img_med(x):
            """
            Medical image visualization:
            - 单通道
            - 不假设 [-1,1]
            - 自动 min-max 拉伸
            """
            x = x[0].detach().cpu().numpy()  # [C,H,W]
            x = np.squeeze(x)               # [H,W]

            vmin = x.min()
            vmax = x.max()
            if vmax > vmin:
                x = (x - vmin) / (vmax - vmin)
            else:
                x = np.zeros_like(x)

            return (x * 255).astype(np.uint8)

        # =========================================================
        # 5️⃣ 生成三张图
        # =========================================================
        img_cond   = to_img_med(xT)        # MRI
        img_sample = to_img_med(sample)    # Pred CT
        img_gt     = to_img_med(batch)     # GT CT

        # 横向拼接：[MRI | Pred | GT]
        img = np.concatenate([img_cond, img_sample, img_gt], axis=1)

        # =========================================================
        # 6️⃣ 保存
        # =========================================================
        if split == "train":
            save_dir = self.image_dir_train
        else:
            save_dir = self.image_dir_val

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"step_{step:06d}.png")

        Image.fromarray(img, mode="L").save(save_path)
        logger.log(f"[visualize] saved {save_path}")

        self.model.train()

##↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑



    def forward_backward(self, batch, cond, train=True):
        if train:
            self.mp_trainer.zero_grad()

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())

            # -------------------------------------------------
            # build model_kwargs (always needs xT)
            # -------------------------------------------------
            if isinstance(cond, dict):
                micro_cond = {
                    "xT": cond["xT"][i : i + self.microbatch].to(dist_util.dev())
                }
            else:
                micro_cond = {}

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(
                micro.shape[0], dist_util.dev()
            )

            # -------------------------------------------------
            # ★ 关键修改：只在 train 阶段取 target
            # -------------------------------------------------
            target = None
            if train and isinstance(cond, dict) and "target" in cond:
                target = cond["target"][i : i + self.microbatch].to(dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_bridge_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                target=target,   # train: Tensor | test: None
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            # -------------------------------------------------
            # loss-aware sampler（只在 train）
            # -------------------------------------------------
            if isinstance(self.schedule_sampler, LossAwareSampler) and train:
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            # -------------------------------------------------
            # logging（自动区分 train / test）
            # -------------------------------------------------
            log_loss_dict(
                self.diffusion,
                t,
                {k if train else "test_" + k: v * weights for k, v in losses.items()},
            )

            if train:
                self.mp_trainer.backward(loss)



    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)

    def save(self, for_preemption=False):
        def maybe_delete_earliest(filename):
            wc = filename.split(f'{(self.step):06d}')[0]+'*'
            freq_states = list(glob.glob(os.path.join(get_blob_logdir(), wc)))
            if len(freq_states) > 3:
                earliest = min(freq_states, key=lambda x: x.split('_')[-1].split('.')[0])
                os.remove(earliest)
                    

        # if dist.get_rank() == 0 and for_preemption:
        #     maybe_delete_earliest(get_blob_logdir())
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model_{(self.step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step):06d}.pt"
                if for_preemption:
                    filename = f"freq_{filename}"
                    maybe_delete_earliest(filename)
                
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            filename = f"opt_{(self.step):06d}.pt"
            if for_preemption:
                filename = f"freq_{filename}"
                maybe_delete_earliest(filename)
                
            with bf.BlobFile(
                bf.join(get_blob_logdir(), filename),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        dist.barrier()



def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/model_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    if main_checkpoint.split('/')[-1].startswith("latest"):
        prefix = 'latest_'
    else:
        prefix = ''
    filename = f"{prefix}ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
