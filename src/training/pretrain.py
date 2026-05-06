"""
Training utilities for video prediction models.

Provides training loop, evaluation, and checkpoint management
for TwoStreamModel and VideoMAEModel.

Multi-GPU 지원:
- 단일 노드: torch.nn.DataParallel (간편)
- 다중 노드 / 다중 GPU 풀스케일: DistributedDataParallel (DDP)
  스크립트가 SLURM_PROCID 또는 RANK 환경변수를 보면 자동으로 DDP 모드 진입.
"""

import json
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from src.models.two_stream import patch_normalize


# ── 분산 학습 (DDP) 헬퍼 ──────────────────────────────────────────────────────

def is_distributed_env():
    """SLURM 또는 torchrun 환경 변수가 있으면 DDP 모드."""
    return ("SLURM_PROCID" in os.environ) or ("RANK" in os.environ and "WORLD_SIZE" in os.environ)


def init_distributed():
    """분산 학습 초기화. (rank, device_idx, world_size) 반환.

    SLURM 환경에서는 SLURM_* 변수에서 직접 추출 (torchrun 불필요).
    그 외에는 torchrun이 설정한 RANK/LOCAL_RANK/WORLD_SIZE 사용.

    device_idx는 torch.cuda.set_device()에 넘길 실제 CUDA 디바이스 번호.
    --gpus-per-task=1 같은 Slurm 옵션은 각 task의 CUDA_VISIBLE_DEVICES를
    제한해서 1개의 GPU만 보이게 만듦 → 이 경우 SLURM_LOCALID가 1이어도
    실제 CUDA device는 0번. torch.cuda.device_count()로 감지해서 처리.
    """
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        slurm_local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        # MASTER_ADDR: SLURM_JOB_NODELIST의 첫 노드 호스트명
        if "MASTER_ADDR" not in os.environ:
            nodelist = os.environ["SLURM_JOB_NODELIST"]
            try:
                first_node = subprocess.check_output(
                    ["scontrol", "show", "hostnames", nodelist],
                    text=True,
                ).strip().splitlines()[0]
                os.environ["MASTER_ADDR"] = first_node
            except Exception as e:
                raise RuntimeError(f"Failed to resolve MASTER_ADDR from SLURM_JOB_NODELIST: {e}")
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
    else:  # torchrun
        rank = int(os.environ["RANK"])
        slurm_local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # CUDA 디바이스 번호 결정:
    # - 각 task가 전체 GPU를 다 볼 수 있으면 (e.g., --gres=gpu:N) → SLURM_LOCALID 그대로
    # - 각 task가 1 GPU로 제한되면 (e.g., --gpus-per-task=1) → 항상 0
    n_visible = torch.cuda.device_count()
    device_idx = slurm_local_rank if n_visible > 1 else 0

    torch.cuda.set_device(device_idx)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return rank, device_idx, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_master():
    """현재 프로세스가 rank 0인지. dist 미초기화 시 True (단일 프로세스)."""
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)


def ssim_loss(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """Structural Similarity loss. Returns 1 - SSIM (lower is better).

    Gaussian window로 local statistics를 계산하여
    luminance, contrast, structure 유사도를 평가.
    FP32 강제: AMP 환경에서 BF16 precision 문제 방지.
    """
    # AMP autocast 비활성화 + FP32 강제 (BF16에서 sigma_sq 음수 → NaN 방지)
    with torch.cuda.amp.autocast(enabled=False):
        pred = pred.float()
        target = target.float()

        # Gaussian window
        coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        window = (g.unsqueeze(0) * g.unsqueeze(1))  # [K, K]
        window = window / window.sum()
        window = window.unsqueeze(0).unsqueeze(0).expand(pred.shape[1], -1, -1, -1)  # [C, 1, K, K]

        pad = window_size // 2
        mu1 = F.conv2d(pred, window, padding=pad, groups=pred.shape[1])
        mu2 = F.conv2d(target, window, padding=pad, groups=target.shape[1])

        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        # clamp: E[X²] - E[X]² 계산에서 부동소수점 오차로 음수 가능
        sigma1_sq = F.conv2d(pred ** 2, window, padding=pad, groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=pad, groups=target.shape[1]) - mu2_sq
        sigma1_sq = sigma1_sq.clamp(min=0)
        sigma2_sq = sigma2_sq.clamp(min=0)
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=pred.shape[1]) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


def train_epoch(model, dataloader, optimizer, device, epoch, dataset=None,
                scaler=None, use_ssim=False, use_bf16=True,
                v12_momentum=None):
    """
    Train for one epoch with multi-gap weighted loss.

    Args:
        model: Video prediction model
        dataloader: DataLoader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch number
        dataset: Dataset with get_loss_weight() method (optional)
        scaler: GradScaler for AMP (None = BF16 mode, BF16에서는 불필요)
        use_bf16: BF16 autocast 사용 여부. H100에서 ~2배 가속.

    Returns:
        avg_loss: Average unweighted loss for the epoch
    """
    model.train()
    total_loss = 0
    total_weighted_loss = 0
    total_loss_current = 0
    total_loss_future = 0
    num_batches = 0
    gap_counts = {}

    for batch_idx, batch in enumerate(dataloader):
        # Unpack batch — v13은 (img_t, img_tk, img_t_global, img_tk_global, gap) 5-tuple
        img_t_global = None
        img_tk_global = None
        if len(batch) == 5:
            img_t, img_tk, img_t_global, img_tk_global, gaps = batch
            gaps = gaps.numpy()
            img_t_global = img_t_global.to(device)
            img_tk_global = img_tk_global.to(device)
        elif len(batch) == 4:
            # backward compat (이전 v13: img_tk_global만)
            img_t, img_tk, img_tk_global, gaps = batch
            gaps = gaps.numpy()
            img_tk_global = img_tk_global.to(device)
        elif len(batch) == 3:
            img_t, img_tk, gaps = batch
            gaps = gaps.numpy()
        else:
            img_t, img_tk = batch
            gaps = np.ones(img_t.shape[0])

        img_t = img_t.to(device)
        img_tk = img_tk.to(device)

        optimizer.zero_grad()

        # AMP autocast: BF16은 GradScaler 없이 autocast만으로 동작
        # (이전 코드: scaler is not None으로 게이팅 → BF16에서 autocast 꺼지는 버그)
        amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_bf16 else torch.cuda.amp.autocast(enabled=False)

        # Compute loss based on model type
        actual_model = model.module if hasattr(model, 'module') else model
        model_name = type(actual_model).__name__

        with amp_ctx:
            if model_name == 'VideoMAEModel':
                loss, img_pred = actual_model.compute_loss(img_t, img_tk)
                weighted_loss = loss
                unweighted_loss = loss
            elif model_name == 'TwoStreamV13Model':
                # v13: dual-frame recon + motion-routed patches + DINO multi-crop CLS
                # forward: (image_current, image_future, image_current_global, image_future_global)
                actual_model = model.module if hasattr(model, 'module') else model
                out = model(img_t, img_tk, img_t_global, img_tk_global)
                loss = out['loss']
                loss_t = out['loss_t']
                loss_tk = out['loss_tk']
                img_pred = out['pred_tk']
                weighted_loss = loss
                unweighted_loss = loss
                total_loss_current += loss_t.item()
                total_loss_future += loss_tk.item()

                with torch.no_grad():
                    cls_m = out['cls_m']
                    cls_p = out['cls_p']
                    # DINO student CLS (multi-crop, mask 0.4) — collapse 모니터링 대상
                    student_dino_cls_t = out['student_dino_cls_t'].float()
                    student_dino_cls_tk = out['student_dino_cls_tk'].float()
                    target_cls_tk_global = out['target_cls_tk_global'].float()
                    pred_cls_tk = student_dino_cls_t  # 진단 alias (예전 metric 호환)
                    std_m = cls_m.std(dim=0).mean()
                    std_p = cls_p.std(dim=0).mean()
                    std_pred_cls = pred_cls_tk.std(dim=0).mean()
                    std_target_cls = target_cls_tk_global.std(dim=0).mean()
                    cm_norm = F.normalize(cls_m.float(), dim=-1)
                    cp_norm = F.normalize(cls_p.float(), dim=-1)
                    pcls_norm = F.normalize(pred_cls_tk, dim=-1)
                    B = cm_norm.shape[0]
                    if B > 1:
                        eye_mask = ~torch.eye(B, device=cm_norm.device, dtype=torch.bool)
                        cos_intra_m = (cm_norm @ cm_norm.T)[eye_mask].mean()
                        cos_intra_p = (cp_norm @ cp_norm.T)[eye_mask].mean()
                        cos_intra_pred_cls = (pcls_norm @ pcls_norm.T)[eye_mask].mean()
                    else:
                        cos_intra_m = torch.tensor(0.0)
                        cos_intra_p = torch.tensor(0.0)
                        cos_intra_pred_cls = torch.tensor(0.0)
                    norm_pred_cls = pred_cls_tk.norm(dim=-1).mean()
                    norm_target_cls = target_cls_tk_global.norm(dim=-1).mean()
                    norm_center = actual_model.dino_center.norm().item()

                if not hasattr(train_epoch, '_v13_metrics_buf'):
                    train_epoch._v13_metrics_buf = {}
                v13buf = train_epoch._v13_metrics_buf
                v13buf['loss_t'] = v13buf.get('loss_t', 0.0) + loss_t.item()
                v13buf['loss_tk'] = v13buf.get('loss_tk', 0.0) + loss_tk.item()
                v13buf['loss_pred_patch'] = v13buf.get('loss_pred_patch', 0.0) + out['loss_pred_patch'].item()
                v13buf['loss_pred_cls'] = v13buf.get('loss_pred_cls', 0.0) + out['loss_pred_cls'].item()
                v13buf['feat_std_m'] = v13buf.get('feat_std_m', 0.0) + std_m.item()
                v13buf['feat_std_p'] = v13buf.get('feat_std_p', 0.0) + std_p.item()
                v13buf['cos_intra_m'] = v13buf.get('cos_intra_m', 0.0) + cos_intra_m.item()
                v13buf['cos_intra_p'] = v13buf.get('cos_intra_p', 0.0) + cos_intra_p.item()
                v13buf['cos_intra_pred_cls'] = v13buf.get('cos_intra_pred_cls', 0.0) + cos_intra_pred_cls.item()
                v13buf['std_pred_cls'] = v13buf.get('std_pred_cls', 0.0) + std_pred_cls.item()
                v13buf['std_target_cls'] = v13buf.get('std_target_cls', 0.0) + std_target_cls.item()
                v13buf['norm_pred_cls'] = v13buf.get('norm_pred_cls', 0.0) + norm_pred_cls.item()
                v13buf['norm_target_cls'] = v13buf.get('norm_target_cls', 0.0) + norm_target_cls.item()
                v13buf['norm_center'] = v13buf.get('norm_center', 0.0) + norm_center
                v13buf['_count'] = v13buf.get('_count', 0) + 1

                # NOTE: DINO center / EMA teacher update는 backward + optimizer.step 후
                # (아래 backward 블록 참조)
            elif model_name == 'TwoStreamV14Model':
                # v14: Stream-wise paradigm specialization (P=MAE+V-JEPA, M=DINO)
                # forward: (image_current, image_future, image_current_global, image_future_global)
                actual_model = model.module if hasattr(model, 'module') else model
                out = model(img_t, img_tk, img_t_global, img_tk_global)
                loss = out['loss']
                loss_t = out['loss_t']
                loss_tk = out['loss_tk']
                loss_pred = out['loss_pred']
                loss_dino = out['loss_dino']
                img_pred = out['pred_tk']
                weighted_loss = loss
                unweighted_loss = loss
                total_loss_current += loss_t.item()
                total_loss_future += loss_tk.item()

                with torch.no_grad():
                    cls_m = out['cls_m']
                    cls_p = out['cls_p']
                    student_dino_cls = out['student_dino_cls'].float()
                    teacher_dino_cls = out['teacher_dino_cls'].float()
                    predicted_tk_repr = out['predicted_tk_repr'].float()
                    target_tk_repr = out['target_tk_repr'].float()

                    std_m = cls_m.std(dim=0).mean()
                    std_p = cls_p.std(dim=0).mean()
                    std_student_dino = student_dino_cls.std(dim=0).mean()
                    std_teacher_dino = teacher_dino_cls.std(dim=0).mean()

                    cm_norm = F.normalize(cls_m.float(), dim=-1)
                    cp_norm = F.normalize(cls_p.float(), dim=-1)
                    sdino_norm = F.normalize(student_dino_cls, dim=-1)
                    B = cm_norm.shape[0]
                    if B > 1:
                        eye_mask = ~torch.eye(B, device=cm_norm.device, dtype=torch.bool)
                        cos_intra_m = (cm_norm @ cm_norm.T)[eye_mask].mean()
                        cos_intra_p = (cp_norm @ cp_norm.T)[eye_mask].mean()
                        cos_intra_dino = (sdino_norm @ sdino_norm.T)[eye_mask].mean()
                    else:
                        cos_intra_m = torch.tensor(0.0)
                        cos_intra_p = torch.tensor(0.0)
                        cos_intra_dino = torch.tensor(0.0)

                    # V-JEPA predictor identity 진단:
                    #   predicted_tk_repr와 target_tk_repr가 cos≈1이면 trivial 통과 의심
                    #   (둘 다 P encoder space의 [B, 1+N, D] → patch mean으로 줄여 비교)
                    pred_pool = predicted_tk_repr[:, 1:].mean(dim=1)
                    tgt_pool = target_tk_repr[:, 1:].mean(dim=1)
                    pred_n = F.normalize(pred_pool, dim=-1)
                    tgt_n = F.normalize(tgt_pool, dim=-1)
                    cos_pred_target = (pred_n * tgt_n).sum(dim=-1).mean()

                    norm_pred = predicted_tk_repr.norm(dim=-1).mean()
                    norm_target = target_tk_repr.norm(dim=-1).mean()
                    norm_center = actual_model.dino_center.norm().item()

                if not hasattr(train_epoch, '_v14_metrics_buf'):
                    train_epoch._v14_metrics_buf = {}
                buf = train_epoch._v14_metrics_buf
                buf['loss_t'] = buf.get('loss_t', 0.0) + loss_t.item()
                buf['loss_tk'] = buf.get('loss_tk', 0.0) + loss_tk.item()
                buf['loss_pred'] = buf.get('loss_pred', 0.0) + loss_pred.item()
                buf['loss_dino'] = buf.get('loss_dino', 0.0) + loss_dino.item()
                buf['feat_std_m'] = buf.get('feat_std_m', 0.0) + std_m.item()
                buf['feat_std_p'] = buf.get('feat_std_p', 0.0) + std_p.item()
                buf['cos_intra_m'] = buf.get('cos_intra_m', 0.0) + cos_intra_m.item()
                buf['cos_intra_p'] = buf.get('cos_intra_p', 0.0) + cos_intra_p.item()
                buf['cos_intra_dino'] = buf.get('cos_intra_dino', 0.0) + cos_intra_dino.item()
                buf['std_student_dino'] = buf.get('std_student_dino', 0.0) + std_student_dino.item()
                buf['std_teacher_dino'] = buf.get('std_teacher_dino', 0.0) + std_teacher_dino.item()
                buf['cos_pred_target'] = buf.get('cos_pred_target', 0.0) + cos_pred_target.item()
                buf['norm_pred'] = buf.get('norm_pred', 0.0) + norm_pred.item()
                buf['norm_target'] = buf.get('norm_target', 0.0) + norm_target.item()
                buf['norm_center'] = buf.get('norm_center', 0.0) + norm_center
                buf['_count'] = buf.get('_count', 0) + 1

                # NOTE: EMA teacher (P+M+head) + DINO center update는 backward + optimizer.step 후.
            elif model_name in ('TwoStreamV11Model', 'TwoStreamV12Model'):
                # v11: dual-target reconstruction (L_t + L_tk, masked positions only)
                # v12: v11 + semantic residual + VICReg + EMA teacher
                # rotation_aug는 model.forward() 내부에서 처리
                out = model(img_t, img_tk)
                loss = out['loss']
                loss_t = out['loss_t']
                loss_tk = out['loss_tk']
                img_pred = out['pred_tk']
                weighted_loss = loss
                unweighted_loss = loss
                total_loss_current += loss_t.item()
                total_loss_future += loss_tk.item()

                # v11 진단 metrics (feat_std, cos_intra: collapse monitoring)
                with torch.no_grad():
                    cls_m = out['cls_m']
                    cls_p = out['cls_p']
                    std_m = cls_m.std(dim=0).mean()
                    std_p = cls_p.std(dim=0).mean()
                    cm_norm = F.normalize(cls_m.float(), dim=-1)
                    cp_norm = F.normalize(cls_p.float(), dim=-1)
                    B = cm_norm.shape[0]
                    if B > 1:
                        sim_m = cm_norm @ cm_norm.T
                        sim_p = cp_norm @ cp_norm.T
                        eye_mask = ~torch.eye(B, device=sim_m.device, dtype=torch.bool)
                        cos_intra_m = sim_m[eye_mask].mean()
                        cos_intra_p = sim_p[eye_mask].mean()
                    else:
                        cos_intra_m = torch.tensor(0.0)
                        cos_intra_p = torch.tensor(0.0)
                if not hasattr(train_epoch, '_v11_metrics_buf'):
                    train_epoch._v11_metrics_buf = {}
                buf = train_epoch._v11_metrics_buf
                buf['loss_t'] = buf.get('loss_t', 0.0) + loss_t.item()
                buf['loss_tk'] = buf.get('loss_tk', 0.0) + loss_tk.item()
                buf['feat_std_m'] = buf.get('feat_std_m', 0.0) + std_m.item()
                buf['feat_std_p'] = buf.get('feat_std_p', 0.0) + std_p.item()
                buf['cos_intra_m'] = buf.get('cos_intra_m', 0.0) + cos_intra_m.item()
                buf['cos_intra_p'] = buf.get('cos_intra_p', 0.0) + cos_intra_p.item()
                buf['_count'] = buf.get('_count', 0) + 1

                # v12 추가 진단 metrics
                if model_name == 'TwoStreamV12Model':
                    with torch.no_grad():
                        sem_m = out['semantic_m'].float()
                        sem_pt = out['semantic_p_t'].float()
                        pred_ptk = out['predicted_p_tk'].float()
                        teacher_ptk = out['teacher_p_tk'].float()

                        # Norms
                        norm_sem_m = sem_m.norm(dim=-1).mean()
                        norm_pred = pred_ptk.norm(dim=-1).mean()
                        norm_teacher = teacher_ptk.norm(dim=-1).mean()

                        # Per-dim std (collapse signal)
                        std_sem_m = sem_m.std(dim=0).mean()
                        std_sem_pt = sem_pt.std(dim=0).mean()
                        std_pred = pred_ptk.std(dim=0).mean()

                        # Time-invariance collapse: cos(sem_p_t, predicted_p_tk)
                        # 1.0 근처면 P가 시간-불변으로 collapse
                        sem_pt_n = F.normalize(sem_pt, dim=-1)
                        pred_n = F.normalize(pred_ptk, dim=-1)
                        cos_pt_pred = (sem_pt_n * pred_n).sum(dim=-1).mean()

                        # Residual equation 만족도: cos(sem_m, teacher_p_tk - sem_p_t)
                        # positive면 학습 진행 (M이 P 변화 방향을 예측)
                        # NOTE: teacher_p_tk와 sem_p_t는 다른 stream/space라서 absolute
                        # cosine이 아니라 trend로 해석
                        delta_p = teacher_ptk - sem_pt
                        delta_p_n = F.normalize(delta_p, dim=-1)
                        sem_m_n = F.normalize(sem_m, dim=-1)
                        cos_m_delta = (sem_m_n * delta_p_n).sum(dim=-1).mean()
                    if not hasattr(train_epoch, '_v12_metrics_buf'):
                        train_epoch._v12_metrics_buf = {}
                    v12buf = train_epoch._v12_metrics_buf
                    v12buf['loss_recon'] = v12buf.get('loss_recon', 0.0) + out['loss_recon'].item()
                    v12buf['loss_residual'] = v12buf.get('loss_residual', 0.0) + out['loss_residual'].item()
                    v12buf['loss_var'] = v12buf.get('loss_var', 0.0) + out['loss_var'].item()
                    v12buf['loss_cov'] = v12buf.get('loss_cov', 0.0) + out['loss_cov'].item()
                    v12buf['norm_sem_m'] = v12buf.get('norm_sem_m', 0.0) + norm_sem_m.item()
                    v12buf['norm_pred'] = v12buf.get('norm_pred', 0.0) + norm_pred.item()
                    v12buf['norm_teacher'] = v12buf.get('norm_teacher', 0.0) + norm_teacher.item()
                    v12buf['std_sem_m'] = v12buf.get('std_sem_m', 0.0) + std_sem_m.item()
                    v12buf['std_sem_pt'] = v12buf.get('std_sem_pt', 0.0) + std_sem_pt.item()
                    v12buf['std_pred'] = v12buf.get('std_pred', 0.0) + std_pred.item()
                    v12buf['cos_pt_pred'] = v12buf.get('cos_pt_pred', 0.0) + cos_pt_pred.item()
                    v12buf['cos_m_delta'] = v12buf.get('cos_m_delta', 0.0) + cos_m_delta.item()
                    v12buf['_count'] = v12buf.get('_count', 0) + 1
            elif model_name == 'TwoStreamModel':
                # rotation_aug: training loop가 compute_loss를 우회하므로 여기서 명시 적용
                if actual_model.rotation_aug and actual_model.training:
                    img_t, img_tk = actual_model._apply_rotation_aug(img_t, img_tk)

                out1, out2, info = model(img_t, img_tk)

                pred_m, pred_p = out1, out2
                # v9: P target 선택 — future(v4), current(MAE), residual
                p_target_mode = getattr(actual_model, 'p_target', 'future')
                if p_target_mode == 'current':
                    target_p = img_t
                elif p_target_mode == 'residual':
                    # Patch-wise norm — 0-output trivial minimum 방지 (MAE 표준)
                    target_p = patch_normalize(img_tk - img_t)
                else:
                    target_p = img_tk
                mse_m = F.mse_loss(pred_m, img_tk, reduction='none').mean(dim=(1, 2, 3))
                mse_p = F.mse_loss(pred_p, target_p, reduction='none').mean(dim=(1, 2, 3))
                if use_ssim:
                    loss_m = mse_m + 0.1 * ssim_loss(pred_m.float(), img_tk.float())
                    # residual target은 diff image라 SSIM 의미 없음
                    if p_target_mode == 'residual':
                        loss_p = mse_p
                    else:
                        loss_p = mse_p + 0.1 * ssim_loss(pred_p.float(), target_p.float())
                else:
                    loss_m = mse_m
                    loss_p = mse_p
                # v9: loss_weight_p로 residual magnitude 보정 (default 1.0 = v4 동일)
                loss_p_raw = loss_p  # weight 곱 전 (monitoring / balance 결정용)
                loss_p = loss_p * actual_model.loss_weight_p
                img_pred = pred_p

                # v9 전용 metrics (p_target != 'future'일 때만 수집, v4 default는 skip)
                if p_target_mode != 'future' and isinstance(info, dict):
                    with torch.no_grad():
                        cls_m = info.get('cls_m')
                        cls_p = info.get('cls_p')
                        # Per-dim std across batch (collapse시 0 근처)
                        std_m = cls_m.std(dim=0).mean()
                        std_p = cls_p.std(dim=0).mean()
                        # Intra-batch cosine (trivial이면 1.0 근처)
                        cp_norm = F.normalize(cls_p.float(), dim=-1)
                        B = cp_norm.shape[0]
                        sim = cp_norm @ cp_norm.T
                        eye_mask = ~torch.eye(B, device=sim.device, dtype=torch.bool)
                        cos_intra_p = sim[eye_mask].mean()
                        # M stream도 동일 진단
                        cm_norm = F.normalize(cls_m.float(), dim=-1)
                        sim_m = cm_norm @ cm_norm.T
                        cos_intra_m = sim_m[eye_mask].mean()
                        resid_mag = (img_tk - img_t).abs().mean()
                    if not hasattr(train_epoch, '_v9_metrics_buf'):
                        train_epoch._v9_metrics_buf = {}
                    buf = train_epoch._v9_metrics_buf
                    buf['loss_m_raw'] = buf.get('loss_m_raw', 0.0) + loss_m.mean().item()
                    buf['loss_p_raw'] = buf.get('loss_p_raw', 0.0) + loss_p_raw.mean().item()
                    buf['loss_p_weighted'] = buf.get('loss_p_weighted', 0.0) + loss_p.mean().item()
                    buf['feat_std_m'] = buf.get('feat_std_m', 0.0) + std_m.item()
                    buf['feat_std_p'] = buf.get('feat_std_p', 0.0) + std_p.item()
                    buf['cos_intra_m'] = buf.get('cos_intra_m', 0.0) + cos_intra_m.item()
                    buf['cos_intra_p'] = buf.get('cos_intra_p', 0.0) + cos_intra_p.item()
                    buf['resid_mag'] = buf.get('resid_mag', 0.0) + resid_mag.item()
                    buf['_count'] = buf.get('_count', 0) + 1

                per_sample_loss = loss_m + loss_p

                total_loss_current += loss_m.mean().item()
                total_loss_future += loss_p.mean().item()

                if dataset is not None and hasattr(dataset, 'get_loss_weight'):
                    weights = torch.tensor(
                        [dataset.get_loss_weight(int(g)) for g in gaps],
                        device=device, dtype=per_sample_loss.dtype
                    )
                else:
                    weights = torch.ones_like(per_sample_loss)

                weighted_loss = (per_sample_loss * weights).mean()
                unweighted_loss = per_sample_loss.mean()
            else:
                img_pred, _ = model(img_t, img_tk)
                per_sample_loss = F.mse_loss(img_pred, img_tk, reduction='none')
                per_sample_loss = per_sample_loss.mean(dim=(1, 2, 3))

                if dataset is not None and hasattr(dataset, 'get_loss_weight'):
                    weights = torch.tensor(
                        [dataset.get_loss_weight(int(g)) for g in gaps],
                        device=device, dtype=per_sample_loss.dtype
                    )
                else:
                    weights = torch.ones_like(per_sample_loss)

                weighted_loss = (per_sample_loss * weights).mean()
                unweighted_loss = per_sample_loss.mean()

        # Backward (BF16 autocast 사용 시 scaler 불필요 — dynamic range가 FP32와 동일)
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # v12 / v13: EMA teacher update (each optimizer step)
        if model_name == 'TwoStreamV12Model' and v12_momentum is not None:
            actual_model.update_teacher(v12_momentum)
        elif model_name == 'TwoStreamV13Model' and v12_momentum is not None:
            actual_model.update_teacher(v12_momentum)
            # DINO center: teacher prototype logits의 running mean (uniform collapse 방어).
            with torch.no_grad():
                teacher_proto = out['teacher_proto_logits']
                if teacher_proto.abs().sum() > 0:  # global view가 실제 사용된 경우만
                    actual_model.update_dino_center(teacher_proto)
        elif model_name == 'TwoStreamV14Model' and v12_momentum is not None:
            # v14: 3 teachers EMA update (TeacherP + TeacherM + TeacherDINOHead)
            actual_model.update_teacher(v12_momentum)
            # DINO center: M stream teacher prototype logits running mean.
            with torch.no_grad():
                teacher_proto = out['teacher_proto_logits']
                if teacher_proto.abs().sum() > 0:
                    actual_model.update_dino_center(teacher_proto)

        total_loss += unweighted_loss.item()
        total_weighted_loss += weighted_loss.item()
        num_batches += 1

        # Track gap distribution
        for g in gaps:
            gap_counts[int(g)] = gap_counts.get(int(g), 0) + 1

        if batch_idx % 10 == 0 and _is_master():
            print(f"  Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {unweighted_loss.item():.4f}, "
                  f"Weighted: {weighted_loss.item():.4f}")

    avg_loss = total_loss / num_batches
    avg_weighted = total_weighted_loss / num_batches

    # Print gap distribution (rank 0)
    if _is_master():
        total_samples = sum(gap_counts.values())
        gap_dist = {k: f"{v/total_samples*100:.1f}%" for k, v in sorted(gap_counts.items())}
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Weighted = {avg_weighted:.4f}")
        if total_loss_future > 0:
            avg_future = total_loss_future / num_batches
            avg_current = total_loss_current / num_batches
            print(f"  Loss breakdown: future={avg_future:.4f}, current={avg_current:.4f}")
        print(f"  Gap distribution: {gap_dist}")

    result = {'loss': avg_loss, 'weighted_loss': avg_weighted}
    if total_loss_future > 0:
        result['loss_future'] = total_loss_future / num_batches
        result['loss_current'] = total_loss_current / num_batches
    return result


def evaluate(model, eval_dataset, device, batch_size=8, num_samples=500, use_ssim=False):
    """
    Evaluate model on validation/test dataset.

    Args:
        model: Video prediction model (or DataParallel wrapped)
        eval_dataset: Dataset with __getitem__ returning (img_t, img_tk, gap)
        device: Device
        batch_size: Batch size for evaluation
        num_samples: Number of samples to evaluate (for speed)

    Returns:
        dict: {'loss': float, 'weighted_loss': float, 'gap_distribution': dict}
    """
    model.eval()

    # Limit sample size
    eval_size = min(num_samples, len(eval_dataset))
    indices = np.random.choice(len(eval_dataset), eval_size, replace=False)

    total_loss = 0
    total_weighted_loss = 0
    total_loss_current = 0
    total_loss_future = 0
    num_batches = 0
    gap_counts = {}

    # Manual batch processing (for fast evaluation)
    for i in range(0, eval_size, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_data = [eval_dataset[idx] for idx in batch_indices]

        img_t = torch.stack([d[0] for d in batch_data]).to(device)
        img_tk = torch.stack([d[1] for d in batch_data]).to(device)
        gaps = np.array([d[2] for d in batch_data])

        # Forward (handle VideoMAE vs prediction models)
        actual_model = model.module if hasattr(model, 'module') else model
        model_name = type(actual_model).__name__

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if model_name == 'VideoMAEModel':
                loss, img_pred = actual_model.compute_loss(img_t, img_tk)
                weighted_loss = loss
                unweighted_loss = loss
            elif model_name == 'TwoStreamV13Model':
                # eval에서는 image_future_global=None (DINO loss 0). reconstruction + patch latent만 평가.
                out = model(img_t, img_tk, None)
                loss = out['loss']
                img_pred = out['pred_tk']
                weighted_loss = loss
                unweighted_loss = loss
                total_loss_current += out['loss_t'].item()
                total_loss_future += out['loss_tk'].item()
            elif model_name == 'TwoStreamV14Model':
                # eval에서는 global=None → V-JEPA / DINO loss 모두 0. MAE reconstruction만 평가.
                out = model(img_t, img_tk, None, None)
                loss = out['loss']
                img_pred = out['pred_tk']
                weighted_loss = loss
                unweighted_loss = loss
                total_loss_current += out['loss_t'].item()
                total_loss_future += out['loss_tk'].item()
            elif model_name in ('TwoStreamV11Model', 'TwoStreamV12Model'):
                out = model(img_t, img_tk)
                loss = out['loss']
                img_pred = out['pred_tk']
                weighted_loss = loss
                unweighted_loss = loss
                total_loss_current += out['loss_t'].item()
                total_loss_future += out['loss_tk'].item()
            elif model_name == 'TwoStreamModel':
                # eval 모드이므로 rotation_aug는 자동 skip (self.training=False)
                out1, out2, _ = model(img_t, img_tk)

                pred_m, pred_p = out1, out2
                # v9: P target 선택 — future(v4), current(MAE), residual
                p_target_mode = getattr(actual_model, 'p_target', 'future')
                if p_target_mode == 'current':
                    target_p = img_t
                elif p_target_mode == 'residual':
                    # Patch-wise norm — train loop와 동일 (MAE 표준)
                    target_p = patch_normalize(img_tk - img_t)
                else:
                    target_p = img_tk
                mse_m = F.mse_loss(pred_m, img_tk, reduction='none').mean(dim=(1, 2, 3))
                mse_p = F.mse_loss(pred_p, target_p, reduction='none').mean(dim=(1, 2, 3))
                if use_ssim:
                    loss_m = mse_m + 0.1 * ssim_loss(pred_m.float(), img_tk.float())
                    # residual target은 diff image라 SSIM 의미 없음
                    if p_target_mode == 'residual':
                        loss_p = mse_p
                    else:
                        loss_p = mse_p + 0.1 * ssim_loss(pred_p.float(), target_p.float())
                else:
                    loss_m = mse_m
                    loss_p = mse_p
                # v9: loss_weight_p로 residual magnitude 보정 (default 1.0 = v4 동일)
                loss_p = loss_p * actual_model.loss_weight_p
                img_pred = pred_p

                per_sample_loss = loss_m + loss_p

                total_loss_current += loss_m.mean().item()
                total_loss_future += loss_p.mean().item()

                if hasattr(eval_dataset, 'get_loss_weight'):
                    weights = torch.tensor(
                        [eval_dataset.get_loss_weight(int(g)) for g in gaps],
                        device=device, dtype=per_sample_loss.dtype
                    )
                else:
                    weights = torch.ones_like(per_sample_loss)

                weighted_loss = (per_sample_loss * weights).mean()
                unweighted_loss = per_sample_loss.mean()
            else:
                # Single-stream: future prediction
                img_pred, _ = model(img_t, img_tk)
                per_sample_loss = F.mse_loss(img_pred, img_tk, reduction='none')
                per_sample_loss = per_sample_loss.mean(dim=(1, 2, 3))

                # Gap-dependent weights
                if hasattr(eval_dataset, 'get_loss_weight'):
                    weights = torch.tensor(
                        [eval_dataset.get_loss_weight(int(g)) for g in gaps],
                        device=device, dtype=per_sample_loss.dtype
                    )
                else:
                    weights = torch.ones_like(per_sample_loss)

                weighted_loss = (per_sample_loss * weights).mean()
                unweighted_loss = per_sample_loss.mean()

        total_loss += unweighted_loss.item()
        total_weighted_loss += weighted_loss.item()
        num_batches += 1

        for g in gaps:
            gap_counts[int(g)] = gap_counts.get(int(g), 0) + 1

    avg_loss = total_loss / num_batches
    avg_weighted = total_weighted_loss / num_batches

    model.train()

    result = {
        'loss': avg_loss,
        'weighted_loss': avg_weighted,
        'gap_distribution': gap_counts,
    }
    if total_loss_future > 0:
        result['loss_future'] = total_loss_future / num_batches
        result['loss_current'] = total_loss_current / num_batches
    return result


def save_epoch_samples(model, eval_dataset, device, epoch, run_dir, num_samples=4):
    """Epoch 끝에 예측 샘플 시각화를 저장.

    run_dir/samples/epoch_XX.png 에 저장.
    실패해도 학습을 중단하지 않음 (전체를 try/except로 보호).
    """
    was_training = model.training
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        model.eval()
        actual_model = model.module if hasattr(model, 'module') else model
        model_name = type(actual_model).__name__

        samples_dir = run_dir / 'samples'
        samples_dir.mkdir(exist_ok=True)

        # 랜덤 샘플 추출
        indices = random.sample(range(len(eval_dataset)), min(num_samples, len(eval_dataset)))

        col_titles = ['Frame t', 'Frame t+k (target)', 'Predicted']
        rows = []
        with torch.no_grad():
            for idx in indices:
                try:
                    img_t, img_tk, gap = eval_dataset[idx]
                except Exception:
                    continue
                x = img_t.unsqueeze(0).to(device)
                y = img_tk.unsqueeze(0).to(device)

                if model_name == 'TwoStreamModel':
                    out1, out2, _ = actual_model(x, y)
                    out1_np = out1.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
                    out2_np = out2.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
                    imgs = [img_t, img_tk, out1_np, out2_np]
                    col_titles = ['Frame t', 'Frame t+k (target)', 'Pred M', 'Pred P']
                elif model_name in ('TwoStreamV11Model', 'TwoStreamV12Model', 'TwoStreamV13Model', 'TwoStreamV14Model'):
                    # 시각화는 mask 없이 (full reconstruction inference). 학습 forward는
                    # MAE-style random mask 적용 → 시각화엔 부적절. mask_ratio=0이면
                    # encoder가 모든 patch 처리 + mask_token inject 없이 통과 → 학습 분포
                    # 밖이지만 model의 reconstruction 능력을 직관적으로 보여줌.
                    saved_mask_p = actual_model.mask_ratio_p
                    saved_mask_m = actual_model.mask_ratio_m
                    actual_model.mask_ratio_p = 0.0
                    actual_model.mask_ratio_m = 0.0
                    try:
                        if model_name == 'TwoStreamV14Model':
                            out = actual_model(x, y, None, None)
                        elif model_name == 'TwoStreamV13Model':
                            out = actual_model(x, y, None)
                        else:
                            out = actual_model(x, y)
                    finally:
                        actual_model.mask_ratio_p = saved_mask_p
                        actual_model.mask_ratio_m = saved_mask_m
                    pred_t_np = out['pred_t'].squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
                    pred_tk_np = out['pred_tk'].squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
                    if model_name == 'TwoStreamV13Model':
                        # v13 encoder-level routing: motion-routed encoder-level latent을
                        # interpreter_1 + recon_head (학습된 decoder path) 통과 → pixel.
                        # In-distribution decoder라 의미 있는 motion-routed reconstruction 예상.
                        predicted_full = torch.cat(
                            [out['predicted_cls_tk'].unsqueeze(1), out['predicted_patches_tk']],
                            dim=1,
                        )  # [B, 1+N, D]
                        predicted_decoded = actual_model._run_interpreter(
                            predicted_full, actual_model.interpreter_1,
                            actual_model.interpreter_1_norm,
                        )
                        patch_pred_motion = actual_model.recon_head(predicted_decoded[:, 1:])
                        pred_tk_motion = (
                            actual_model._unpatchify(patch_pred_motion)
                            .squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
                        )
                        imgs = [img_t, img_tk, pred_t_np, pred_tk_np, pred_tk_motion]
                        col_titles = [
                            'Frame t', 'Frame t+k',
                            'Pred t (single)', 'Pred t+k (single)',
                            'Pred t+k (motion-routed → decoder)',
                        ]
                    else:
                        imgs = [img_t, img_tk, pred_t_np, pred_tk_np]
                        col_titles = ['Frame t', 'Frame t+k', 'Pred t (Ph1)', 'Pred t+k (Ph3)']
                elif model_name == 'VideoMAEModel':
                    # 픽셀 복원이 아닌 feature/masked 예측 → 이미지 시각화 생략
                    continue
                else:
                    pred, _ = actual_model(x, y)
                    pred = pred.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
                    imgs = [img_t, img_tk, pred]

                # tensor → numpy for display
                processed = []
                for img in imgs:
                    if isinstance(img, torch.Tensor):
                        img = img.permute(1, 2, 0).numpy().clip(0, 1)
                    processed.append(img)
                rows.append((processed, gap))

        if not rows:
            return

        ncols = len(rows[0][0])
        nrows = len(rows)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        if nrows == 1:
            axes = [axes]

        for row_idx, (imgs, gap) in enumerate(rows):
            for col_idx, img in enumerate(imgs):
                axes[row_idx][col_idx].imshow(img)
                axes[row_idx][col_idx].axis('off')
                if row_idx == 0:
                    axes[row_idx][col_idx].set_title(col_titles[col_idx], fontsize=10)
            axes[row_idx][0].set_ylabel(f'gap={gap}', fontsize=9, rotation=0, labelpad=40, va='center')

        fig.suptitle(f'Epoch {epoch}', fontsize=13, y=1.01)
        plt.tight_layout()
        path = samples_dir / f'epoch_{epoch:03d}.png'
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved samples: {path.name}")

    except Exception as e:
        print(f"  [WARNING] Sample visualization failed (epoch {epoch}): {e}")
    finally:
        if was_training:
            model.train()


def train(
    model,
    train_dataset,
    num_epochs=10,
    batch_size=8,
    lr=1e-4,
    device="cuda",
    checkpoint_dir=None,
    save_interval=None,
    eval_dataset=None,
    eval_interval=1,
    resume_from=None,
    multi_gpu=True,
    use_ssim=False,
    num_workers=16,
    v12_ema_momentum_init=0.996,
    v12_ema_momentum_final=0.9999,
):
    """
    Main training loop with periodic evaluation and checkpointing.

    분산 학습 모드 (SLURM_PROCID 또는 RANK env가 설정된 경우):
    - DistributedDataParallel 사용 (DataParallel 우회)
    - DistributedSampler가 데이터 분할
    - rank 0만 print/checkpoint/eval/TB 수행

    Args:
        model: Video prediction model
        train_dataset: Training dataset
        num_epochs: Number of training epochs
        batch_size: Per-GPU batch size (DDP 모드: 각 GPU의 배치, DP 모드: 단일 GPU 배치)
        lr: Learning rate (DDP 모드에서는 외부에서 미리 scaling해서 전달할 것)
        device: 'cuda' 또는 'cpu'. DDP 모드에서는 무시되고 자동으로 cuda:local_rank 사용
        checkpoint_dir: Base directory for checkpoints (auto-creates timestamped subfolder)
        save_interval: Save checkpoint every N epochs (None = only save best)
        eval_dataset: Evaluation dataset (optional)
        eval_interval: Evaluate every N epochs
        resume_from: Path to checkpoint to resume training from
        multi_gpu: DP 모드에서만 의미. DDP 모드에서는 무시됨
        use_ssim: Add SSIM loss to MSE (for TwoStream)
        num_workers: DataLoader worker 수

    Returns:
        model: Trained model
        history: Training history dict
    """
    # ── 분산 학습 vs 단일/DP 모드 결정 ─────────────────────────────────────────
    distributed = is_distributed_env()
    rank = 0
    local_rank = 0
    world_size = 1

    if distributed:
        rank, local_rank, world_size = init_distributed()
        device = torch.device(f"cuda:{local_rank}")
        is_master = (rank == 0)
        if is_master:
            print(f"DDP enabled: rank={rank}, local_rank={local_rank}, world_size={world_size}")
            print(f"  Backend: nccl, MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
                  f"MASTER_PORT={os.environ.get('MASTER_PORT')}")
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        effective_batch_size = batch_size  # per-GPU; DistributedSampler가 데이터 분할
        global_batch_size = batch_size * world_size
        if is_master:
            print(f"  Per-GPU batch size: {batch_size}")
            print(f"  Global batch size: {global_batch_size}")
    else:
        is_master = True
        # Multi-GPU DP setup (기존 방식)
        use_multi_gpu = multi_gpu and torch.cuda.device_count() > 1
        if use_multi_gpu:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = model.to(device)
            model = torch.nn.DataParallel(model)
            effective_batch_size = batch_size * torch.cuda.device_count()
            print(f"  Per-GPU batch size: {batch_size}")
            print(f"  Effective batch size: {effective_batch_size}")
        else:
            model = model.to(device)
            effective_batch_size = batch_size

    def log(msg):
        if is_master:
            print(msg)

    # Setup checkpoint directory with timestamp (rank 0만 생성)
    run_dir = None
    if checkpoint_dir and is_master:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(checkpoint_dir) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        log(f"Checkpoint directory: {run_dir}")

    # DataLoader: DDP는 DistributedSampler 사용, 그 외는 shuffle=True
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=True,
        )
    else:
        train_sampler = None
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    # Fused AdamW — optimizer step을 단일 CUDA 커널로 합침 (5~10% 가속, PyTorch 2.0+)
    #
    # Weight decay param group 정책:
    # - VideoMAEModel: 공식 VideoMAE 프로토콜 준수 → LN/bias/mask_token 제외
    #   (공식 optim_factory.py의 no_weight_decay() 동작 재현)
    # - TwoStreamModel: 기존 uniform weight_decay 유지 (이미 학습된
    #   체크포인트와 호환성 보존, optimizer state_dict 구조 변경 회피)
    # 논문 메소드 섹션에 "VideoMAE-ours만 공식 optimizer 프로토콜 적용, 우리 모델은
    # simpler uniform weight decay 사용"으로 투명하게 기재.
    _actual = model.module if hasattr(model, 'module') else model
    _model_name = type(_actual).__name__

    if _model_name == 'VideoMAEModel':
        # bias, LayerNorm γ/β, cls_token, pos_embed, mask_token → wd=0
        decay_params, no_decay_params = [], []
        for name, p in _actual.named_parameters():
            if not p.requires_grad:
                continue
            # 1D 파라미터(bias, LN γ/β) 또는 cls_token/pos_embed/mask_token → no decay
            no_decay_kw = ('cls_token', 'pos_embed', 'mask_token')
            if p.ndim <= 1 or any(kw in name for kw in no_decay_kw):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        log(f"{_model_name} optimizer param_groups: decay={len(decay_params)}, "
            f"no_decay={len(no_decay_params)}")
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': 0.01},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=lr, fused=True)
    else:
        # requires_grad=True 파라미터만 옵티마이저에 등록 (v12 teacher params 제외)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01, fused=True)

    # LR schedule: linear warmup (10% of epochs) + cosine decay
    # Warmup은 EMA 기반 모델(V-JEPA)의 초기 안정성에 필수.
    # Two-Stream(pixel target)에도 무해하므로 공통 적용.
    warmup_epochs = max(1, num_epochs // 10)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6 / lr, total_iters=warmup_epochs,
    )
    # Guard: num_epochs <= warmup_epochs (sanity test 등) → T_max=0 division 방지
    cosine_T_max = max(1, num_epochs - warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_T_max,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # AMP BF16 — H100 Tensor Core 기준 FP32 대비 ~2배 throughput
    # BF16은 FP32와 동일한 exponent range → GradScaler 불필요 (FP16만 필요)
    use_bf16 = True
    log(f"AMP: BF16 autocast enabled, Fused AdamW enabled")

    # Resume from checkpoint if specified
    start_epoch = 1
    best_eval_loss = float('inf')
    history = {
        'train_loss': [],
        'eval_loss': [],
        'epoch_time': [],        # Seconds per epoch
        'samples_per_sec': [],   # Training throughput
        'timestamps': [],        # ISO timestamp at epoch end
    }

    if resume_from:
        log(f"Resuming from {resume_from}")
        # 모든 rank가 동일 체크포인트를 메모리에서 로드 (DDP는 자동 broadcast 안 함)
        map_location = {'cuda:0': str(device)} if distributed else device
        checkpoint = torch.load(resume_from, map_location=map_location)
        # DDP 또는 DataParallel: model.module에 로드
        model_to_load = model.module if hasattr(model, 'module') else model
        try:
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            log(f"  WARNING: Checkpoint incompatible — {e}")
            log(f"  Starting from scratch (ignoring old checkpoint)")
            resume_from = None
        if resume_from:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'history' in checkpoint:
                history = checkpoint['history']
                history.setdefault('epoch_time', [])
                history.setdefault('samples_per_sec', [])
                history.setdefault('timestamps', [])
            if 'best_eval_loss' in checkpoint:
                best_eval_loss = checkpoint['best_eval_loss']
            log(f"  Resumed from epoch {checkpoint['epoch']}, LR: {scheduler.get_last_lr()[0]:.2e}")

    log(f"\nTraining for {num_epochs} epochs (starting from epoch {start_epoch})")
    log(f"  Train dataset: {len(train_dataset)} samples")
    if eval_dataset:
        log(f"  Eval dataset: {len(eval_dataset)} samples (every {eval_interval} epochs)")
    log(f"  Batch size (per GPU): {batch_size}")
    if distributed:
        log(f"  World size: {world_size}, global batch: {batch_size * world_size}")
    log(f"  Learning rate: {lr}")
    if save_interval:
        log(f"  Save interval: every {save_interval} epochs")
    log("")

    # TensorBoard (rank 0만)
    writer = None
    if run_dir and HAS_TENSORBOARD and is_master:
        writer = SummaryWriter(log_dir=str(run_dir / 'tb'))
        log(f"TensorBoard: {run_dir / 'tb'}")

    # Save config (rank 0만)
    if run_dir and is_master:
        config = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'global_batch_size': batch_size * world_size,
            'world_size': world_size,
            'distributed': distributed,
            'lr': lr,
            'eval_interval': eval_interval,
            'save_interval': save_interval,
            'train_dataset_size': len(train_dataset),
            'eval_dataset_size': len(eval_dataset) if eval_dataset else 0,
            'start_time': datetime.now().isoformat(),
        }
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()

        # DistributedSampler shuffle 재초기화 (필수 — 빠뜨리면 매 epoch 같은 순서)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # v12: EMA momentum schedule (linear warmup over training)
        # current epoch progress ratio in [0, 1]
        if num_epochs > 1:
            progress = (epoch - 1) / (num_epochs - 1)
        else:
            progress = 0.0
        v12_momentum = v12_ema_momentum_init + (v12_ema_momentum_final - v12_ema_momentum_init) * progress

        # Train
        train_result = train_epoch(
            model, dataloader, optimizer, device, epoch,
            dataset=train_dataset, use_ssim=use_ssim, use_bf16=use_bf16,
            v12_momentum=v12_momentum,
        )
        avg_loss = train_result['loss']
        scheduler.step()
        history['train_loss'].append(avg_loss)

        # Evaluate (rank 0만 수행, 다른 rank는 barrier 대기)
        eval_loss = None
        if eval_dataset and epoch % eval_interval == 0:
            if is_master:
                try:
                    eval_result = evaluate(model, eval_dataset, device, batch_size=batch_size, use_ssim=use_ssim)
                    eval_loss = eval_result['loss']
                    history['eval_loss'].append(eval_loss)
                    log(f"  [Eval] Loss: {eval_loss:.4f}, Weighted: {eval_result['weighted_loss']:.4f}")
                except Exception as e:
                    log(f"  [WARNING] Evaluation failed (epoch {epoch}): {e}")
            if distributed:
                dist.barrier()

        # Track time metrics
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        samples_per_sec = len(train_dataset) / epoch_duration
        current_timestamp = datetime.now().isoformat()

        history['epoch_time'].append(epoch_duration)
        history['samples_per_sec'].append(samples_per_sec)
        history['timestamps'].append(current_timestamp)

        # Print time metrics (rank 0)
        current_lr = scheduler.get_last_lr()[0]
        log(f"  [Time] Epoch: {epoch_duration:.1f}s, Throughput: {samples_per_sec:.1f} samples/sec, LR: {current_lr:.2e}")

        # Estimate remaining time
        if epoch < num_epochs:
            avg_epoch_time = sum(history['epoch_time']) / len(history['epoch_time'])
            remaining_epochs = num_epochs - epoch
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_hours = eta_seconds / 3600
            log(f"  [ETA] {remaining_epochs} epochs remaining, ~{eta_hours:.1f}h ({eta_seconds/60:.0f}min)")

        # TensorBoard logging (rank 0만; writer는 rank 0에서만 생성됨)
        if writer:
            writer.add_scalar('loss/train', avg_loss, epoch)
            if 'loss_future' in train_result:
                writer.add_scalar('loss/train_future', train_result['loss_future'], epoch)
                writer.add_scalar('loss/train_current', train_result['loss_current'], epoch)
            if eval_loss is not None:
                writer.add_scalar('loss/eval', eval_loss, epoch)
            writer.add_scalar('lr', current_lr, epoch)
            writer.add_scalar('perf/samples_per_sec', samples_per_sec, epoch)
            writer.add_scalar('perf/epoch_time_sec', epoch_duration, epoch)
            # v9 전용 monitoring metrics (residual P target, collapse detector)
            if hasattr(train_epoch, '_v9_metrics_buf'):
                buf = train_epoch._v9_metrics_buf
                n = buf.get('_count', 0)
                if n > 0:
                    lm_raw = buf['loss_m_raw'] / n
                    lp_raw = buf['loss_p_raw'] / n
                    lp_w = buf['loss_p_weighted'] / n
                    writer.add_scalar('v9/loss_m_raw', lm_raw, epoch)
                    writer.add_scalar('v9/loss_p_raw', lp_raw, epoch)
                    writer.add_scalar('v9/loss_p_weighted', lp_w, epoch)
                    writer.add_scalar('v9/loss_ratio_p_over_m', lp_raw / max(lm_raw, 1e-8), epoch)
                    writer.add_scalar('v9/feat_std_m', buf['feat_std_m'] / n, epoch)
                    writer.add_scalar('v9/feat_std_p', buf['feat_std_p'] / n, epoch)
                    writer.add_scalar('v9/cos_intra_m', buf['cos_intra_m'] / n, epoch)
                    writer.add_scalar('v9/cos_intra_p', buf['cos_intra_p'] / n, epoch)
                    writer.add_scalar('v9/resid_mag', buf['resid_mag'] / n, epoch)
                    log(
                        f"  [v9] L_m={lm_raw:.5f} L_p_raw={lp_raw:.5f} "
                        f"(ratio p/m={lp_raw/max(lm_raw,1e-8):.3f}) "
                        f"weighted L_p={lp_w:.5f} | "
                        f"std_m={buf['feat_std_m']/n:.3f} std_p={buf['feat_std_p']/n:.3f} | "
                        f"cos_intra_m={buf['cos_intra_m']/n:.3f} cos_intra_p={buf['cos_intra_p']/n:.3f} | "
                        f"resid_mag={buf['resid_mag']/n:.4f}"
                    )
                train_epoch._v9_metrics_buf = {}
            # v11 metrics (dual-target + collapse monitoring)
            if hasattr(train_epoch, '_v11_metrics_buf'):
                buf = train_epoch._v11_metrics_buf
                n = buf.get('_count', 0)
                if n > 0:
                    lt = buf['loss_t'] / n
                    ltk = buf['loss_tk'] / n
                    writer.add_scalar('v11/loss_t', lt, epoch)
                    writer.add_scalar('v11/loss_tk', ltk, epoch)
                    writer.add_scalar('v11/loss_ratio_tk_over_t', ltk / max(lt, 1e-8), epoch)
                    writer.add_scalar('v11/feat_std_m', buf['feat_std_m'] / n, epoch)
                    writer.add_scalar('v11/feat_std_p', buf['feat_std_p'] / n, epoch)
                    writer.add_scalar('v11/cos_intra_m', buf['cos_intra_m'] / n, epoch)
                    writer.add_scalar('v11/cos_intra_p', buf['cos_intra_p'] / n, epoch)
                    log(
                        f"  [v11] L_t={lt:.5f} L_tk={ltk:.5f} "
                        f"(ratio tk/t={ltk/max(lt,1e-8):.3f}) | "
                        f"std_m={buf['feat_std_m']/n:.3f} std_p={buf['feat_std_p']/n:.3f} | "
                        f"cos_intra_m={buf['cos_intra_m']/n:.3f} cos_intra_p={buf['cos_intra_p']/n:.3f}"
                    )
                train_epoch._v11_metrics_buf = {}

            # v12 metrics (semantic residual + EMA teacher diagnostics)
            if hasattr(train_epoch, '_v12_metrics_buf'):
                v12buf = train_epoch._v12_metrics_buf
                n = v12buf.get('_count', 0)
                if n > 0:
                    l_recon = v12buf['loss_recon'] / n
                    l_res = v12buf['loss_residual'] / n
                    l_var = v12buf['loss_var'] / n
                    l_cov = v12buf['loss_cov'] / n
                    writer.add_scalar('v12/loss_recon', l_recon, epoch)
                    writer.add_scalar('v12/loss_residual', l_res, epoch)
                    writer.add_scalar('v12/loss_var', l_var, epoch)
                    writer.add_scalar('v12/loss_cov', l_cov, epoch)
                    writer.add_scalar('v12/ema_momentum', v12_momentum, epoch)
                    writer.add_scalar('v12/norm_sem_m', v12buf['norm_sem_m'] / n, epoch)
                    writer.add_scalar('v12/norm_pred', v12buf['norm_pred'] / n, epoch)
                    writer.add_scalar('v12/norm_teacher', v12buf['norm_teacher'] / n, epoch)
                    writer.add_scalar('v12/std_sem_m', v12buf['std_sem_m'] / n, epoch)
                    writer.add_scalar('v12/std_sem_pt', v12buf['std_sem_pt'] / n, epoch)
                    writer.add_scalar('v12/std_pred', v12buf['std_pred'] / n, epoch)
                    writer.add_scalar('v12/cos_pt_pred', v12buf['cos_pt_pred'] / n, epoch)
                    writer.add_scalar('v12/cos_m_delta', v12buf['cos_m_delta'] / n, epoch)
                    log(
                        f"  [v12] L_res={l_res:.5f} L_var={l_var:.4f} L_cov={l_cov:.4f} | "
                        f"ema_m={v12_momentum:.4f} | "
                        f"||sem_m||={v12buf['norm_sem_m']/n:.2f} ||pred||={v12buf['norm_pred']/n:.2f} "
                        f"||teacher||={v12buf['norm_teacher']/n:.2f} | "
                        f"std_pred={v12buf['std_pred']/n:.3f} | "
                        f"cos(p_t,pred)={v12buf['cos_pt_pred']/n:.3f} "
                        f"cos(m,Δp)={v12buf['cos_m_delta']/n:.3f}"
                    )
                train_epoch._v12_metrics_buf = {}

            # v13 metrics (dual-frame recon + motion-routed latent + DINO CLS)
            if hasattr(train_epoch, '_v13_metrics_buf'):
                v13buf = train_epoch._v13_metrics_buf
                n = v13buf.get('_count', 0)
                if n > 0:
                    lt = v13buf['loss_t'] / n
                    ltk = v13buf['loss_tk'] / n
                    lpp = v13buf['loss_pred_patch'] / n
                    lpc = v13buf['loss_pred_cls'] / n
                    writer.add_scalar('v13/loss_t', lt, epoch)
                    writer.add_scalar('v13/loss_tk', ltk, epoch)
                    writer.add_scalar('v13/loss_pred_patch', lpp, epoch)
                    writer.add_scalar('v13/loss_pred_cls', lpc, epoch)
                    writer.add_scalar('v13/feat_std_m', v13buf['feat_std_m'] / n, epoch)
                    writer.add_scalar('v13/feat_std_p', v13buf['feat_std_p'] / n, epoch)
                    writer.add_scalar('v13/cos_intra_m', v13buf['cos_intra_m'] / n, epoch)
                    writer.add_scalar('v13/cos_intra_p', v13buf['cos_intra_p'] / n, epoch)
                    writer.add_scalar('v13/cos_intra_pred_cls', v13buf['cos_intra_pred_cls'] / n, epoch)
                    writer.add_scalar('v13/std_pred_cls', v13buf['std_pred_cls'] / n, epoch)
                    writer.add_scalar('v13/std_target_cls', v13buf['std_target_cls'] / n, epoch)
                    writer.add_scalar('v13/norm_pred_cls', v13buf['norm_pred_cls'] / n, epoch)
                    writer.add_scalar('v13/norm_target_cls', v13buf['norm_target_cls'] / n, epoch)
                    writer.add_scalar('v13/norm_dino_center', v13buf['norm_center'] / n, epoch)
                    writer.add_scalar('v13/ema_momentum', v12_momentum, epoch)
                    log(
                        f"  [v13] L_t={lt:.5f} L_tk={ltk:.5f} L_pp={lpp:.5f} L_pc={lpc:.5f} | "
                        f"ema_m={v12_momentum:.4f} | "
                        f"std_m={v13buf['feat_std_m']/n:.3f} std_p={v13buf['feat_std_p']/n:.3f} "
                        f"std_pred_cls={v13buf['std_pred_cls']/n:.3f} | "
                        f"cos_intra_p={v13buf['cos_intra_p']/n:.3f} "
                        f"cos_intra_pred_cls={v13buf['cos_intra_pred_cls']/n:.3f} | "
                        f"||pred||={v13buf['norm_pred_cls']/n:.2f} ||tgt||={v13buf['norm_target_cls']/n:.2f} "
                        f"||center||={v13buf['norm_center']/n:.2f}"
                    )
                train_epoch._v13_metrics_buf = {}

            # v14 metrics (stream-wise paradigm specialization)
            if hasattr(train_epoch, '_v14_metrics_buf'):
                v14buf = train_epoch._v14_metrics_buf
                n = v14buf.get('_count', 0)
                if n > 0:
                    lt = v14buf['loss_t'] / n
                    ltk = v14buf['loss_tk'] / n
                    lpr = v14buf['loss_pred'] / n
                    ldn = v14buf['loss_dino'] / n
                    writer.add_scalar('v14/loss_t', lt, epoch)
                    writer.add_scalar('v14/loss_tk', ltk, epoch)
                    writer.add_scalar('v14/loss_pred', lpr, epoch)
                    writer.add_scalar('v14/loss_dino', ldn, epoch)
                    writer.add_scalar('v14/feat_std_m', v14buf['feat_std_m'] / n, epoch)
                    writer.add_scalar('v14/feat_std_p', v14buf['feat_std_p'] / n, epoch)
                    writer.add_scalar('v14/cos_intra_m', v14buf['cos_intra_m'] / n, epoch)
                    writer.add_scalar('v14/cos_intra_p', v14buf['cos_intra_p'] / n, epoch)
                    writer.add_scalar('v14/cos_intra_dino', v14buf['cos_intra_dino'] / n, epoch)
                    writer.add_scalar('v14/std_student_dino', v14buf['std_student_dino'] / n, epoch)
                    writer.add_scalar('v14/std_teacher_dino', v14buf['std_teacher_dino'] / n, epoch)
                    writer.add_scalar('v14/cos_pred_target', v14buf['cos_pred_target'] / n, epoch)
                    writer.add_scalar('v14/norm_pred', v14buf['norm_pred'] / n, epoch)
                    writer.add_scalar('v14/norm_target', v14buf['norm_target'] / n, epoch)
                    writer.add_scalar('v14/norm_dino_center', v14buf['norm_center'] / n, epoch)
                    writer.add_scalar('v14/ema_momentum', v12_momentum, epoch)
                    log(
                        f"  [v14] L_t={lt:.5f} L_tk={ltk:.5f} L_pred={lpr:.5f} L_dino={ldn:.5f} | "
                        f"ema_m={v12_momentum:.4f} | "
                        f"std_m={v14buf['feat_std_m']/n:.3f} std_p={v14buf['feat_std_p']/n:.3f} "
                        f"std_sdino={v14buf['std_student_dino']/n:.3f} | "
                        f"cos_intra_p={v14buf['cos_intra_p']/n:.3f} "
                        f"cos_intra_dino={v14buf['cos_intra_dino']/n:.3f} | "
                        f"cos(pred,tgt)={v14buf['cos_pred_target']/n:.3f} | "
                        f"||center||={v14buf['norm_center']/n:.2f}"
                    )
                train_epoch._v14_metrics_buf = {}
            writer.flush()

        # Save prediction samples every epoch (rank 0)
        if run_dir and eval_dataset and is_master:
            save_epoch_samples(model, eval_dataset, device, epoch, run_dir)

        # Save checkpoint (rank 0만)
        if run_dir and is_master:
            # DDP/DP 모두 model.module에 실모델
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
                'eval_loss': eval_loss,
                'best_eval_loss': best_eval_loss,
                'history': history,
            }

            # Save periodic checkpoint
            if save_interval and epoch % save_interval == 0:
                ckpt_path = run_dir / f'checkpoint_epoch{epoch:04d}.pt'
                torch.save(checkpoint_data, ckpt_path)
                log(f"  Saved checkpoint: {ckpt_path.name}")

            # Save best model
            current_loss = eval_loss if eval_loss is not None else avg_loss
            if current_loss < best_eval_loss:
                best_eval_loss = current_loss
                checkpoint_data['best_eval_loss'] = best_eval_loss
                best_path = run_dir / 'best_model.pt'
                torch.save(checkpoint_data, best_path)
                log(f"  Saved best model (loss: {current_loss:.4f})")

            # Always save latest (for resume)
            latest_path = run_dir / 'latest.pt'
            torch.save(checkpoint_data, latest_path)

        # 모든 rank가 다음 epoch 진입 동기화
        if distributed:
            dist.barrier()

    # Save final history (rank 0)
    if run_dir and is_master:
        with open(run_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    if writer:
        writer.close()

    if distributed:
        cleanup_distributed()

    return model, history
