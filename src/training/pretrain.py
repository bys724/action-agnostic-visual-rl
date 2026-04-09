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


# ── 분산 학습 (DDP) 헬퍼 ──────────────────────────────────────────────────────

def is_distributed_env():
    """SLURM 또는 torchrun 환경 변수가 있으면 DDP 모드."""
    return ("SLURM_PROCID" in os.environ) or ("RANK" in os.environ and "WORLD_SIZE" in os.environ)


def init_distributed():
    """분산 학습 초기화. (rank, local_rank, world_size) 반환.

    SLURM 환경에서는 SLURM_* 변수에서 직접 추출 (torchrun 불필요).
    그 외에는 torchrun이 설정한 RANK/LOCAL_RANK/WORLD_SIZE 사용.
    """
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
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
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return rank, local_rank, world_size


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


def train_epoch(model, dataloader, optimizer, device, epoch, dataset=None, scaler=None, use_ssim=False):
    """
    Train for one epoch with multi-gap weighted loss.

    Args:
        model: Video prediction model
        dataloader: DataLoader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch number
        dataset: Dataset with get_loss_weight() method (optional)
        scaler: GradScaler for AMP (None = FP32)

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
        # Unpack batch (img_t, img_tk, gap) or (img_t, img_tk)
        if len(batch) == 3:
            img_t, img_tk, gaps = batch
            gaps = gaps.numpy()
        else:
            img_t, img_tk = batch
            gaps = np.ones(img_t.shape[0])

        img_t = img_t.to(device)
        img_tk = img_tk.to(device)

        optimizer.zero_grad()

        # AMP autocast context
        use_amp = scaler is not None
        amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp else torch.cuda.amp.autocast(enabled=False)

        # Compute loss based on model type
        actual_model = model.module if hasattr(model, 'module') else model
        model_name = type(actual_model).__name__

        with amp_ctx:
            if model_name == 'VideoMAEModel':
                loss, img_pred = actual_model.compute_loss(img_t, img_tk)
                weighted_loss = loss
                unweighted_loss = loss
            elif model_name == 'TwoStreamModel':
                pred_m, pred_p, _ = model(img_t, img_tk)
                mse_m = F.mse_loss(pred_m, img_tk, reduction='none').mean(dim=(1, 2, 3))
                mse_p = F.mse_loss(pred_p, img_tk, reduction='none').mean(dim=(1, 2, 3))
                if use_ssim:
                    loss_m = mse_m + 0.1 * ssim_loss(pred_m.float(), img_tk.float())
                    loss_p = mse_p + 0.1 * ssim_loss(pred_p.float(), img_tk.float())
                else:
                    loss_m = mse_m
                    loss_p = mse_p
                per_sample_loss = loss_m + loss_p
                img_pred = pred_p

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

        # Backward with AMP scaling
        if use_amp:
            scaler.scale(weighted_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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
            elif model_name == 'TwoStreamModel':
                pred_m, pred_p, _ = model(img_t, img_tk)
                mse_m = F.mse_loss(pred_m, img_tk, reduction='none').mean(dim=(1, 2, 3))
                mse_p = F.mse_loss(pred_p, img_tk, reduction='none').mean(dim=(1, 2, 3))
                if use_ssim:
                    loss_m = mse_m + 0.1 * ssim_loss(pred_m.float(), img_tk.float())
                    loss_p = mse_p + 0.1 * ssim_loss(pred_p.float(), img_tk.float())
                else:
                    loss_m = mse_m
                    loss_p = mse_p
                per_sample_loss = loss_m + loss_p
                img_pred = pred_p

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
                    pred_m, pred_p, _ = actual_model(x, y)
                    pred_m = pred_m.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
                    pred_p = pred_p.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
                    imgs = [img_t, img_tk, pred_m, pred_p]
                    col_titles = ['Frame t', 'Frame t+k (target)', 'Pred M', 'Pred P']
                elif model_name == 'VideoMAEModel':
                    # VideoMAE는 masked patch만 예측 → full image 시각화 생략
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # AMP (BF16) — H100 Tensor Core 최적화
    # BF16은 FP32와 동일한 dynamic range → GradScaler 불필요
    # GradScaler + BF16 조합은 mixed-precision loss (SSIM 등)에서 NaN 유발 가능
    scaler = None
    print(f"AMP enabled (BF16, no GradScaler)")

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
            if 'scaler_state_dict' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
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

        # Train
        train_result = train_epoch(model, dataloader, optimizer, device, epoch, dataset=train_dataset, scaler=scaler, use_ssim=use_ssim)
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
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
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
