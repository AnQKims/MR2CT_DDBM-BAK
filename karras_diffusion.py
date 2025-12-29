"""
Based on: https://github.com/crowsonkb/k-diffusion
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from piq import LPIPS
import torch

from torchvision.transforms.functional import gaussian_blur
from .nn import mean_flat, append_dims, append_zero

from functools import partial


def vp_logsnr(t, beta_d, beta_min):
    t = th.as_tensor(t)
    return - th.log((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1)
    
def vp_logs(t, beta_d, beta_min):
    t = th.as_tensor(t)
    return -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min

# 用于计算梯度loss函数
def gradient_l1_loss(x, y):
    # x, y: [B, C, H, W]
    dx_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx_y = y[:, :, :, 1:] - y[:, :, :, :-1]

    dy_x = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy_y = y[:, :, 1:, :] - y[:, :, :-1, :]

    return mean_flat(th.abs(dx_x - dx_y)) + mean_flat(th.abs(dy_x - dy_y))

def gradient(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy

def gradient_map(x):
    """
    计算一阶梯度幅值，用于 edge loss
    x: [B, C, H, W]
    """
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    grad = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
    return grad


class KarrasDenoiser:
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        beta_d=2,
        beta_min=0.1,
        cov_xy=0., # 0 for uncorrelated, sigma_data**2 / 2 for  C_skip=1/2 at sigma_max
        rho=7.0,
        image_size=64,
        weight_schedule="karras",
        pred_mode='both',
        loss_norm="lpips",
    ):
        self.sigma_data = sigma_data
        
        self.sigma_max = sigma_max 
        self.sigma_min = sigma_min 

        self.beta_d = beta_d
        self.beta_min = beta_min
        

        self.sigma_data_end = self.sigma_data
        self.cov_xy = cov_xy
            
        self.c = 1

        self.weight_schedule = weight_schedule
        self.pred_mode = pred_mode
        self.loss_norm = loss_norm
        if loss_norm == "lpips":
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")
        self.rho = rho
        self.num_timesteps = 40
        self.image_size = image_size


    def get_snr(self, sigmas):
        if self.pred_mode.startswith('vp'):
            return vp_logsnr(sigmas, self.beta_d, self.beta_min).exp()
        else:
            return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas


    def get_weightings(self, sigma):
        snrs = self.get_snr(sigma)
        
        if self.weight_schedule == "snr":
            weightings = snrs
        elif self.weight_schedule == "snr+1":
            weightings = snrs + 1
        elif self.weight_schedule == "karras":
            weightings = snrs + 1.0 / self.sigma_data**2
        elif self.weight_schedule.startswith("bridge_karras"):
            if self.pred_mode == 've':
                A = sigma**4 / self.sigma_max**4 * self.sigma_data_end**2 + (1 - sigma**2 / self.sigma_max**2)**2 * self.sigma_data**2 + 2*sigma**2 / self.sigma_max**2 * (1 - sigma**2 / self.sigma_max**2) * self.cov_xy + self.c**2 * sigma**2 * (1 - sigma**2 / self.sigma_max**2)
                weightings = A / ((sigma/self.sigma_max)**4 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * self.c**2 * sigma**2 * (1 - sigma**2/self.sigma_max**2) )
            
            elif self.pred_mode == 'vp':
                
                logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
                logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
                logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
                logs_T = vp_logs(1, self.beta_d, self.beta_min)

                a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
                b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                c_t = -th.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()

                A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2*a_t * b_t * self.cov_xy + self.c**2 * c_t
                weightings = A / (a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * self.c**2 * c_t )
                
            elif self.pred_mode == 'vp_simple' or  self.pred_mode == 've_simple':

                weightings = th.ones_like(snrs)
        elif self.weight_schedule == "truncated-snr":
            weightings = th.clamp(snrs, min=1.0)
        elif self.weight_schedule == "uniform":
            weightings = th.ones_like(snrs)
        else:
            raise NotImplementedError()

        return weightings


    def get_bridge_scalings(self, sigma):
        if self.pred_mode == 've':
            A = sigma**4 / self.sigma_max**4 * self.sigma_data_end**2 + (1 - sigma**2 / self.sigma_max**2)**2 * self.sigma_data**2 + 2*sigma**2 / self.sigma_max**2 * (1 - sigma**2 / self.sigma_max**2) * self.cov_xy + self.c **2 * sigma**2 * (1 - sigma**2 / self.sigma_max**2)
            c_in = 1 / (A) ** 0.5
            c_skip = ((1 - sigma**2 / self.sigma_max**2) * self.sigma_data**2 + sigma**2 / self.sigma_max**2 * self.cov_xy)/ A
            c_out =((sigma/self.sigma_max)**4 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 *  self.c **2 * sigma**2 * (1 - sigma**2/self.sigma_max**2) )**0.5 * c_in
            return c_skip, c_out, c_in
        
    
        elif self.pred_mode == 'vp':

            logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
            logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
            logs_T = vp_logs(1, self.beta_d, self.beta_min)

            a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
            b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            c_t = -th.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()

            A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2*a_t * b_t * self.cov_xy + self.c**2 * c_t
            
            
            c_in = 1 / (A) ** 0.5
            c_skip = (b_t * self.sigma_data**2 + a_t * self.cov_xy)/ A
            c_out =(a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 *  self.c **2 * c_t )**0.5 * c_in
            return c_skip, c_out, c_in
            
    
        elif self.pred_mode == 've_simple' or self.pred_mode == 'vp_simple':
            
            c_in = th.ones_like(sigma)
            c_out = th.ones_like(sigma) 
            c_skip = th.zeros_like(sigma)
            return c_skip, c_out, c_in
        

#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
   
#### ORI DDBM BRIDGE LOSS ###
    def training_bridge_losses(
            self,
            model,
            x_start,        # x_start：真实的 x₀，原始 MRI（无噪声）
            sigmas,         # sigmas：当前 batch 中每个样本的噪声强度 σ
            model_kwargs=None,  # 额外条件参数（必须包含 xT）
            noise=None,     # 可选噪声 ε，不给则内部随机生成
            vae=None,       # 预留接口（此 loss 中不使用）
            target=None,    # 预留接口（CT）
        ):
        """
        Original DDBM bridge loss（接口兼容版本）

        这是 DDBM 原论文 / 原实现中的 bridge loss，
        不使用 CT supervision，仅训练 MRI → MRI 的 bridge 去噪能力
        """

        # =========================================================
        # 1️⃣ 基本安全检查
        # =========================================================

        # DDBM 的 bridge 必须知道终点 xT（高噪 MRI）
        # xT 是 condition，不是模型预测目标
        assert model_kwargs is not None

        # 从 model_kwargs 中取出 xT（桥的另一端）
        xT = model_kwargs["xT"]

        # 如果没有显式传入噪声，就在这里生成标准高斯噪声 ε ~ N(0, I)
        # 形状和 x_start 完全一致
        if noise is None:
            noise = th.randn_like(x_start)

        # =========================================================
        # 2️⃣ sigma 裁剪（防数值不稳定）
        # =========================================================

        # 理论上 sigmas ∈ [0, σ_max]
        # 这里强制 clip，防止外部 sampler 给出异常 σ
        sigmas = th.minimum(
            sigmas,
            th.ones_like(sigmas) * self.sigma_max
        )

        # x_start 的维度数（通常是 4：B,C,H,W）
        dims = x_start.ndim

        # 用于存放所有 loss 项的字典
        terms = {}

        # =========================================================
        # 3️⃣ Bridge forward sampling（核心！）
        # =========================================================
        # 目标：
        #   在 x0（MRI）和 xT（高噪 MRI）之间
        #   构造一个中间态 x_t
        #
        #   x_t = μ_t(x0, xT) + σ_t * ε
        #
        def bridge_sample(x0, xT, t):

            # 将 t 扩展成 [B,1,1,1] 以便广播
            t = append_dims(t, dims)

            # =====================================================
            # VE（Variance Exploding）Bridge
            # =====================================================
            if self.pred_mode.startswith("ve"):

                # --------- 方差项 σ_t ---------
                # std_t = t * sqrt(1 - t^2 / σ_max^2)
                #
                # 含义：
                # - t → 0      → std_t → 0
                # - t → σ_max  → std_t → 0
                # - 中间最大
                #
                # 确保桥在两端是确定性的
                std_t = t * th.sqrt(1 - t**2 / self.sigma_max**2)

                # --------- 均值项 μ_t ---------
                # μ_t = α(t) * xT + (1-α(t)) * x0
                #
                # α(t) = t² / σ_max²
                #
                # t=0        → μ_t = x0
                # t=σ_max    → μ_t = xT
                mu_t = (
                    t**2 / self.sigma_max**2 * xT
                    + (1 - t**2 / self.sigma_max**2) * x0
                )

                # 最终 bridge 样本
                return mu_t + std_t * noise

            # =====================================================
            # VP（Variance Preserving）Bridge
            # =====================================================
            elif self.pred_mode.startswith("vp"):

                # VP 下使用 logSNR 参数化
                logsnr_t = vp_logsnr(t, self.beta_d, self.beta_min)
                logsnr_T = vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)

                logs_t = vp_logs(t, self.beta_d, self.beta_min)
                logs_T = vp_logs(self.sigma_max, self.beta_d, self.beta_min)

                # 系数 a_t：xT 的权重
                a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()

                # 系数 b_t：x0 的权重
                b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()

                # 噪声标准差
                std_t = (-th.expm1(logsnr_T - logsnr_t)).sqrt() * (
                    logs_t - logsnr_t / 2
                ).exp()

                return a_t * xT + b_t * x0 + std_t * noise

            else:
                raise NotImplementedError
            # =========================================================
            # 4️⃣ 生成中间态 x_t
            # =========================================================

        # 在 x0 和 xT 之间采样桥中间点
        x_t = bridge_sample(x_start, xT, sigmas)
        # =========================================================
        # 5️⃣ 调用去噪网络
        # =========================================================

        # denoised：模型对 x_t 的去噪结果（目标是 x_start）
        model_output, denoised = self.denoise(
            model,
            x_t,
            sigmas,
            **model_kwargs
        )
        # =========================================================
        # 6️⃣ 原始 DDBM Loss
        # =========================================================

        # 根据 σ 计算 loss 权重（论文中的 σ-dependent weighting）
        weights = self.get_weightings(sigmas)

        # 扩展维度以便广播到像素
        weights = append_dims(weights, dims)

        # ---------------------------------------------------------
        # xs_mse：无权重 MSE（仅用于日志观察）
        # ---------------------------------------------------------
        terms["xs_mse"] = mean_flat(
            (denoised - x_start) ** 2
        )

        # ---------------------------------------------------------
        # mse：真正用于训练的加权 MSE
        # ---------------------------------------------------------
        terms["mse"] = mean_flat(
            weights * (denoised - x_start) ** 2
        )

        # ---------------------------------------------------------
        # 总 loss
        # ---------------------------------------------------------

        # 如果有 variational bound（vb），就一起加
        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms

# 1224 ---------------------------------------------------------------

    # def training_bridge_losses(
    #         self,
    #         model,
    #         x_start,        # x_start：真实的 x₀，原始 MRI（无噪声）
    #         sigmas,         # sigmas：当前 batch 中每个样本的噪声强度 σ
    #         model_kwargs=None,  # 额外条件参数（必须包含 xT）
    #         noise=None,     # 可选噪声 ε，不给则内部随机生成
    #         vae=None,       # 预留接口（此 loss 中不使用）
    #         target=None,    # 预留接口（CT）
    #     ):
    #     """
    #     DDBM bridge loss + 方案三（σ→0 的弱 CT 结构约束）

    #     ✔ 主体仍然是 Original DDBM bridge loss
    #     ✔ CT 不参与 bridge、不作为预测目标
    #     ✔ CT 只在 σ → 0 时，作为极弱的结构正则
    #     """

    #     # =========================================================
    #     # 1️⃣ 基本安全检查
    #     # =========================================================

    #     # DDBM 的 bridge 必须知道终点 xT（高噪 MRI）
    #     # xT 是 condition，不是模型预测目标
    #     assert model_kwargs is not None

    #     # 从 model_kwargs 中取出 xT（桥的另一端）
    #     xT = model_kwargs["xT"]

    #     # 如果没有显式传入噪声，就在这里生成标准高斯噪声 ε ~ N(0, I)
    #     # 形状和 x_start 完全一致
    #     if noise is None:
    #         noise = th.randn_like(x_start)

    #     # =========================================================
    #     # 2️⃣ sigma 裁剪（防数值不稳定）
    #     # =========================================================

    #     # 理论上 sigmas ∈ [0, σ_max]
    #     # 这里强制 clip，防止外部 sampler 给出异常 σ
    #     sigmas = th.minimum(
    #         sigmas,
    #         th.ones_like(sigmas) * self.sigma_max
    #     )

    #     # x_start 的维度数（通常是 4：B,C,H,W）
    #     dims = x_start.ndim

    #     # 用于存放所有 loss 项的字典
    #     terms = {}

    #     # =========================================================
    #     # 3️⃣ Bridge forward sampling（核心！）
    #     # =========================================================
    #     # 目标：
    #     #   在 x0（MRI）和 xT（高噪 MRI）之间
    #     #   构造一个中间态 x_t
    #     #
    #     #   x_t = μ_t(x0, xT) + σ_t * ε
    #     #
    #     def bridge_sample(x0, xT, t):

    #         # 将 t 扩展成 [B,1,1,1] 以便广播
    #         t = append_dims(t, dims)

    #         # =====================================================
    #         # VE（Variance Exploding）Bridge
    #         # =====================================================
    #         if self.pred_mode.startswith("ve"):

    #             # --------- 方差项 σ_t ---------
    #             # std_t = t * sqrt(1 - t^2 / σ_max^2)
    #             #
    #             # 含义：
    #             # - t → 0      → std_t → 0
    #             # - t → σ_max  → std_t → 0
    #             # - 中间最大
    #             #
    #             # 确保桥在两端是确定性的
    #             std_t = t * th.sqrt(1 - t**2 / self.sigma_max**2)

    #             # --------- 均值项 μ_t ---------
    #             # μ_t = α(t) * xT + (1-α(t)) * x0
    #             #
    #             # α(t) = t² / σ_max²
    #             #
    #             # t=0        → μ_t = x0
    #             # t=σ_max    → μ_t = xT
    #             mu_t = (
    #                 t**2 / self.sigma_max**2 * xT
    #                 + (1 - t**2 / self.sigma_max**2) * x0
    #             )

    #             # 最终 bridge 样本
    #             return mu_t + std_t * noise

    #         # =====================================================
    #         # VP（Variance Preserving）Bridge
    #         # =====================================================
    #         elif self.pred_mode.startswith("vp"):

    #             # VP 下使用 logSNR 参数化
    #             logsnr_t = vp_logsnr(t, self.beta_d, self.beta_min)
    #             logsnr_T = vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)

    #             logs_t = vp_logs(t, self.beta_d, self.beta_min)
    #             logs_T = vp_logs(self.sigma_max, self.beta_d, self.beta_min)

    #             # 系数 a_t：xT 的权重
    #             a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()

    #             # 系数 b_t：x0 的权重
    #             b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()

    #             # 噪声标准差
    #             std_t = (-th.expm1(logsnr_T - logsnr_t)).sqrt() * (
    #                 logs_t - logsnr_t / 2
    #             ).exp()

    #             return a_t * xT + b_t * x0 + std_t * noise

    #         else:
    #             raise NotImplementedError
    #         # =========================================================
    #         # 4️⃣ 生成中间态 x_t
    #         # =========================================================

    #     # 在 x0 和 xT 之间采样桥中间点
    #     x_t = bridge_sample(x_start, xT, sigmas)
    #     # =========================================================
    #     # 5️⃣ 调用去噪网络
    #     # =========================================================

    #     # denoised：模型对 x_t 的去噪结果（目标是 x_start）
    #     model_output, denoised = self.denoise(
    #         model,
    #         x_t,
    #         sigmas,
    #         **model_kwargs
    #     )
    #     # =========================================================
    #     # 6️⃣ 原始 DDBM Loss
    #     # =========================================================

    #     # 根据 σ 计算 loss 权重（论文中的 σ-dependent weighting）
    #     weights = self.get_weightings(sigmas)

    #     # 扩展维度以便广播到像素
    #     weights = append_dims(weights, dims)

    #     # ---------------------------------------------------------
    #     # xs_mse：无权重 MSE（仅用于日志观察）
    #     # ---------------------------------------------------------
    #     terms["xs_mse"] = mean_flat(
    #         (denoised - x_start) ** 2
    #     )

    #     # ---------------------------------------------------------
    #     # mse：真正用于训练的加权 MSE
    #     # ---------------------------------------------------------
    #     terms["mse"] = mean_flat(
    #         weights * (denoised - x_start) ** 2
    #     )

    #     # =========================================================
    #     # 7️⃣ 添加方案：σ → 0 的 CT 结构约束
    #     # =========================================================
        
    #     # 1224 CT 结构弱约束 --------------------------------------
    #     ct_loss = 0.0

    #     if target is not None:
    #         # 仅在 σ 很小时启用 CT
    #         sigma_ct = 0.05 * self.sigma_max  # 可调：0.02 ~ 0.1
    #         ct_mask = (sigmas < sigma_ct).float()
    #         ct_mask = append_dims(ct_mask, dims)

    #         # ---- 结构约束：梯度（edge）----
    #         def gradient(x):
    #             dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    #             dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    #             return dx, dy

    #         dx_p, dy_p = gradient(denoised)
    #         dx_t, dy_t = gradient(target)

    #         edge_l1 = (
    #             (dx_p - dx_t).abs().mean()
    #             + (dy_p - dy_t).abs().mean()
    #         )

    #         ct_loss = (edge_l1 * ct_mask).mean()
    #         terms["ct_edge"] = ct_loss
    #     # ---------------------------------------------------------

    #     # =========================================================
    #     #  总 loss（DDBM 主导，CT 极弱修正）
    #     # =========================================================

    #     lambda_ct = 1e-3  # 非常小！建议从 1e-4 / 1e-3 开始

    #     if "vb" in terms:
    #         terms["loss"] = terms["mse"] + terms["vb"] + lambda_ct * ct_loss
    #     else:
    #         terms["loss"] = terms["mse"] + lambda_ct * ct_loss

    #     return terms





#  1225  --------------------------------------------------------------
    # def training_bridge_losses(
    #     self,
    #     model,
    #     x_start,            # MRI x0
    #     sigmas,             # [B]
    #     model_kwargs=None,  # must contain xT
    #     noise=None,
    #     vae=None,
    #     target=None,        # CT
    # ):
    #     """
    #     - 原始 DDBM bridge loss（不动）
    #     - σ→0 连续 CT 结构 + 强度 + 能量弱约束
    #     """

    #     assert model_kwargs is not None
    #     xT = model_kwargs["xT"]

    #     if noise is None:
    #         noise = th.randn_like(x_start)

    #     # ===============================
    #     # 1. sigma safety
    #     # ===============================
    #     sigmas = th.minimum(
    #         sigmas,
    #         th.ones_like(sigmas) * self.sigma_max
    #     )
    #     dims = x_start.ndim
    #     terms = {}

    #     # ===============================
    #     # 2. Bridge sampling（不动）
    #     # ===============================
    #     def bridge_sample(x0, xT, t):
    #         t = append_dims(t, dims)

    #         if self.pred_mode.startswith("ve"):
    #             std_t = t * th.sqrt(1 - t**2 / self.sigma_max**2)
    #             mu_t = (
    #                 t**2 / self.sigma_max**2 * xT
    #                 + (1 - t**2 / self.sigma_max**2) * x0
    #             )
    #             return mu_t + std_t * noise

    #         elif self.pred_mode.startswith("vp"):
    #             logsnr_t = vp_logsnr(t, self.beta_d, self.beta_min)
    #             logsnr_T = vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)
    #             logs_t = vp_logs(t, self.beta_d, self.beta_min)
    #             logs_T = vp_logs(self.sigma_max, self.beta_d, self.beta_min)

    #             a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
    #             b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
    #             std_t = (-th.expm1(logsnr_T - logsnr_t)).sqrt() * (
    #                 logs_t - logsnr_t / 2
    #             ).exp()
    #             return a_t * xT + b_t * x0 + std_t * noise

    #         else:
    #             raise NotImplementedError

    #     x_t = bridge_sample(x_start, xT, sigmas)

    #     # ===============================
    #     # 3. Denoise（标准 DDBM）
    #     # ===============================
    #     model_output, denoised = self.denoise(
    #         model, x_t, sigmas, **model_kwargs
    #     )

    #     # ===============================
    #     # 4. 原始 bridge MSE（核心）
    #     # ===============================
    #     weights = append_dims(self.get_weightings(sigmas), dims)

    #     terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
    #     terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

    #     # ===============================
    #     # 5. 1225：σ→0 连续 CT 约束
    #     # ===============================
    #     ct_loss = 0.0
    #     energy_loss = 0.0

    #     if target is not None:
    #         # σ 连续权重（替代 hard mask）
    #         sigma_ct = 0.05 * self.sigma_max
    #         ct_weight = (1.0 - sigmas / sigma_ct).clamp(0.0, 1.0)
    #         ct_weight = append_dims(ct_weight, dims)

    #         # ---- region / thickness ----
    #         ct_blur_pred = gaussian_blur(denoised, kernel_size=5, sigma=1.0)
    #         ct_blur_gt   = gaussian_blur(target,   kernel_size=5, sigma=1.0)
    #         ct_region = mean_flat((ct_blur_pred - ct_blur_gt).abs())

    #         # ---- edge ----
    #         grad_p = gradient_map(denoised)
    #         grad_t = gradient_map(target)
    #         ct_edge = mean_flat((grad_p - grad_t).abs())

    #         # ---- intensity ----
    #         ct_int = mean_flat((denoised - target) ** 2)

    #         ct_loss = (
    #             0.5 * ct_region +
    #             0.3 * ct_edge +
    #             0.2 * ct_int
    #         )
    #         ct_loss = mean_flat(ct_weight * ct_loss)

    #         # ---- MRI–CT energy alignment ----
    #         mri_energy = mean_flat(x_start.abs())
    #         ct_energy  = mean_flat(denoised.abs())
    #         energy_loss = mean_flat(ct_weight * (ct_energy - mri_energy) ** 2)

    #         terms.update({
    #             "ct_region": ct_region,
    #             "ct_edge": ct_edge,
    #             "ct_intensity": ct_int,
    #             "ct_weight": ct_weight.mean(),
    #             "energy_loss": energy_loss,
    #         })

    #     # ===============================
    #     # 6. 总 loss
    #     # ===============================
    #     lambda_ct = 1e-3

    #     if "vb" in terms:
    #         terms["loss"] = (
    #             terms["mse"] +
    #             terms["vb"] +
    #             lambda_ct * ct_loss +
    #             1e-3 * energy_loss
    #         )
    #     else:
    #         terms["loss"] = (
    #             terms["mse"] +
    #             lambda_ct * ct_loss +
    #             1e-3 * energy_loss
    #         )

    #     return terms

# -----------------------------------------------------------------------

#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑



    


    def denoise(self, model, x_t, sigmas ,**model_kwargs):

        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_bridge_scalings(sigmas)
        ]
               
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised

def karras_sample(
    diffusion,
    model,
    x_T,
    x_0,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    churn_step_ratio=0.,
    guidance=1,
):
    assert sampler in ["heun", ], 'only heun sampler is supported currently'
    
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max-1e-4, rho, device=device)


    sample_fn = {
        "heun": partial(sample_heun, beta_d=diffusion.beta_d, beta_min=diffusion.beta_min),
    }[sampler]

    sampler_args = dict(
            pred_mode=diffusion.pred_mode, churn_step_ratio=churn_step_ratio, sigma_max=sigma_max
        )
    def denoiser(x_t, sigma, x_T=None):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
                
        return denoised
    
    x_0, path, nfe = sample_fn(
        denoiser,
        x_T,
        sigmas,
        progress=progress,
        callback=callback,
        guidance=guidance,
        **sampler_args,
    )
    print('nfe:', nfe)

    return x_0.clamp(-1, 1), [x.clamp(-1, 1) for x in path], nfe


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_bridge_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, eps=1e-4, device="cpu"):
    
    sigma_t_crit = sigma_max / np.sqrt(2)
    min_start_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_t_crit ** (1 / rho)
    sigmas_second_half = (max_inv_rho + th.linspace(0, 1, n//2 ) * (min_start_inv_rho - max_inv_rho)) ** rho
    sigmas_first_half = sigma_max - ((sigma_max - sigma_t_crit)  ** (1 / rho) + th.linspace(0, 1, n - n//2 +1 ) * (eps  ** (1 / rho)  - (sigma_max - sigma_t_crit)  ** (1 / rho))) ** rho
    sigmas = th.cat([sigmas_first_half.flip(0)[:-1], sigmas_second_half])
    sigmas_bridge = sigmas**2 *(1-sigmas**2/sigma_max**2)
    return append_zero(sigmas).to(device)#, append_zero(sigmas_bridge).to(device)


def to_d(x, sigma, denoised, x_T, sigma_max,   w=1, stochastic=False):
    """Converts a denoiser output to a Karras ODE derivative."""
    grad_pxtlx0 = (denoised - x) / append_dims(sigma**2, x.ndim)
    grad_pxTlxt = (x_T - x) / (append_dims(th.ones_like(sigma)*sigma_max**2, x.ndim) - append_dims(sigma**2, x.ndim))
    gt2 = 2*sigma
    d = - (0.5 if not stochastic else 1) * gt2 * (grad_pxtlx0 - w * grad_pxTlxt * (0 if stochastic else 1))
    if stochastic:
        return d, gt2
    else:
        return d


def get_d_vp(x, denoised, x_T, std_t,logsnr_t, logsnr_T, logs_t, logs_T, s_t_deriv, sigma_t, sigma_t_deriv, w, stochastic=False):
    
    a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
    b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
    
    mu_t = a_t * x_T + b_t * denoised 
    
    grad_logq = - (x - mu_t)/std_t**2 / (-th.expm1(logsnr_T - logsnr_t))
    # grad_logpxtlx0 = - (x - logs_t.exp()*denoised)/std_t**2 
    grad_logpxTlxt = -(x - th.exp(logs_t-logs_T)*x_T) /std_t**2  / th.expm1(logsnr_t - logsnr_T)

    f = s_t_deriv * (-logs_t).exp() * x
    gt2 = 2 * (logs_t).exp()**2 * sigma_t * sigma_t_deriv 
    # breakpoint()

    d = f -  gt2 * ((0.5 if not stochastic else 1)* grad_logq - w * grad_logpxTlxt)
    # d = f - (0.5 if not stochastic else 1) * gt2 * (grad_logpxtlx0 - w * grad_logpxTlxt* (0 if stochastic else 1))
    if stochastic:
        return d, gt2
    else:
        return d

@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    pred_mode='both',
    progress=False,
    callback=None,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    churn_step_ratio=0.,
    guidance=1,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    x_T = x
    path = [x]
    
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    nfe = 0
    assert churn_step_ratio < 1

    if pred_mode.startswith('vp'):
        vp_snr_sqrt_reciprocal = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_snr_sqrt_reciprocal_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (vp_snr_sqrt_reciprocal(t) + 1 / vp_snr_sqrt_reciprocal(t))
        s = lambda t: (1 + vp_snr_sqrt_reciprocal(t) ** 2).rsqrt()
        s_deriv = lambda t: -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)

        logs = lambda t: -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min
        
        std =  lambda t: vp_snr_sqrt_reciprocal(t) * s(t)
        
        logsnr = lambda t :  - 2 * th.log(vp_snr_sqrt_reciprocal(t))

        logsnr_T = logsnr(th.as_tensor(sigma_max))
        logs_T = logs(th.as_tensor(sigma_max))
    
    for j, i in enumerate(indices):
        
        if churn_step_ratio > 0:
            # 1 step euler
            sigma_hat = (sigmas[i+1] - sigmas[i]) * churn_step_ratio + sigmas[i]
            
            denoised = denoiser(x, sigmas[i] * s_in, x_T)
            if pred_mode == 've':
                d_1, gt2 = to_d(x, sigmas[i] , denoised, x_T, sigma_max,  w=guidance, stochastic=True)
            elif pred_mode.startswith('vp'):
                d_1, gt2 = get_d_vp(x, denoised, x_T, std(sigmas[i]),logsnr(sigmas[i]), logsnr_T, logs(sigmas[i] ), logs_T, s_deriv(sigmas[i] ), vp_snr_sqrt_reciprocal(sigmas[i] ), vp_snr_sqrt_reciprocal_deriv(sigmas[i] ), guidance, stochastic=True)
            
            dt = (sigma_hat - sigmas[i]) 
            x = x + d_1 * dt + th.randn_like(x) *((dt).abs() ** 0.5)*gt2.sqrt()
            
            nfe += 1
            
            path.append(x.detach().cpu())
        else:
            sigma_hat =  sigmas[i]
        
        # heun step
        denoised = denoiser(x, sigma_hat * s_in, x_T)
        if pred_mode == 've':
            # d =  (x - denoised ) / append_dims(sigma_hat, x.ndim)
            d = to_d(x, sigma_hat, denoised, x_T, sigma_max, w=guidance)
        elif pred_mode.startswith('vp'):
            d = get_d_vp(x, denoised, x_T, std(sigma_hat),logsnr(sigma_hat), logsnr_T, logs(sigma_hat), logs_T, s_deriv(sigma_hat), vp_snr_sqrt_reciprocal(sigma_hat), vp_snr_sqrt_reciprocal_deriv(sigma_hat), guidance)
            
        nfe += 1
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            
            x = x + d * dt 
            
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, x_T)
            if pred_mode == 've':
                # d_2 =  (x_2 - denoised_2) / append_dims(sigmas[i + 1], x.ndim)
                d_2 = to_d(x_2,  sigmas[i + 1], denoised_2, x_T, sigma_max, w=guidance)
            elif pred_mode.startswith('vp'):
                d_2 = get_d_vp(x_2, denoised_2, x_T, std(sigmas[i + 1]),logsnr(sigmas[i + 1]), logsnr_T, logs(sigmas[i + 1]), logs_T, s_deriv(sigmas[i + 1]),
                                vp_snr_sqrt_reciprocal(sigmas[i + 1]), vp_snr_sqrt_reciprocal_deriv(sigmas[i + 1]), guidance)
            
            d_prime = (d + d_2) / 2

            # noise = th.zeros_like(x) if 'flow' in pred_mode or pred_mode == 'uncond' else generator.randn_like(x)
            x = x + d_prime * dt #+ noise * (sigmas[i + 1]**2 - sigma_hat**2).abs() ** 0.5
            nfe += 1
        # loss = (denoised.detach().cpu() - x0).pow(2).mean().item()
        # losses.append(loss)

        path.append(x.detach().cpu())
        
    return x, path, nfe

@th.no_grad()
def forward_sample(
    x0,
    y0,
    sigma_max,
    ):

    ts = th.linspace(0, sigma_max, 120)
    x = x0
    # for t, t_next in zip(ts[:-1], ts[1:]):
    #     grad_pxTlxt = (y0 - x) / (append_dims(th.ones_like(ts)*sigma_max**2, x.ndim) - append_dims(t**2, x.ndim))
    #     dt = (t_next - t) 
    #     gt2 = 2*t
    #     x = x + grad_pxTlxt * dt + th.randn_like(x) *((dt).abs() ** 0.5)*gt2.sqrt()
    path = [x]
    for t in ts:
        std_t = th.sqrt(t)* th.sqrt(1 - t / sigma_max)
        mu_t= t / sigma_max * y0 + (1 - t / sigma_max) * x0
        xt = (mu_t +  std_t * th.randn_like(x0) )
        path.append(xt)

    path.append(y0)

    return path


