#!/usr/bin/env python
"""
Training script for Video Prediction Models.

Supports two models:
    1. two-stream: Two-Stream Interleaved ViT (ours)
    2. videomae: Masked autoencoder (comparison baseline)

Usage:
    python scripts/pretrain.py --model two-stream --epochs 30
    python scripts/pretrain.py --model videomae --epochs 30
"""

import argparse
import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.models import TwoStreamModel, TwoStreamV11Model, VideoMAEModel
from src.datasets import EgoDexDataset
from src.training.pretrain import train


def main():
    parser = argparse.ArgumentParser(description='Training for Video Prediction Models')

    # Model selection
    parser.add_argument('--model', type=str, default='two-stream',
                        choices=['two-stream', 'two-stream-v11', 'videomae'],
                        help='Model type (default: two-stream). '
                             'two-stream-v11 = motion-guided routing + dual-target.')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32, H100 can handle more)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')

    # Dataset parameters
    parser.add_argument('--train-data', type=str, default='egodex',
                        choices=['egodex'],
                        help='Training dataset (default: egodex)')
    parser.add_argument('--egodex-root', type=str,
                        default='/workspace/data/egodex',
                        help='EgoDex data root')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Max videos for EgoDex (None = all)')
    parser.add_argument('--egodex-splits', type=str, default='part1',
                        help='EgoDex splits to use, comma-separated (default: part1, e.g. part1,part2,part3)')

    # Checkpoint parameters
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Checkpoint directory (default: /workspace/data/checkpoints/<model>)')
    parser.add_argument('--save-interval', type=int, default=None,
                        help='Save checkpoint every N epochs (default: epochs//12)')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='Evaluate every N epochs (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')

    # Multi-gap sampling parameters
    parser.add_argument('--max-gap', type=int, default=30,
                        help='Max frame gap (default: 30, ~1sec at 30fps)')
    parser.add_argument('--sample-decay', type=float, default=0.0,
                        help='Sample probability decay (0=uniform, >0=exponential, <0=linear)')
    parser.add_argument('--sample-dist', type=str, default='auto',
                        choices=['auto', 'uniform', 'linear', 'exp', 'triangular'],
                        help='Gap sampling distribution (default: auto, determined by sample-decay)')
    parser.add_argument('--sample-center', type=int, default=None,
                        help='Center gap for triangular distribution (default: max_gap//2)')
    parser.add_argument('--loss-decay', type=float, default=0.0,
                        help='Loss weight decay (0=uniform, >0=exponential)')

    # Architecture (Two-Stream)
    parser.add_argument('--depth', type=int, default=12,
                        help='Transformer depth per stream (default: 12)')
    parser.add_argument('--num-stages', type=int, default=3,
                        help='Number of CLS exchange stages (default: 3)')

    # Loss
    parser.add_argument('--ssim', action='store_true',
                        help='Add SSIM loss (MSE + 0.1 * SSIM)')
    parser.add_argument('--mask-ratio', type=float, default=0.0,
                        help='MAE-style mask ratio for M stream (0=disabled, 0.3 recommended)')
    parser.add_argument('--mask-ratio-p', type=float, default=None,
                        help='P stream mask ratio (default: same as --mask-ratio, higher recommended)')
    parser.add_argument('--use-ape', action='store_true',
                        help='[Two-Stream] Use APE (learnable pos embed) instead of 2D RoPE. '
                             '진단용: RoPE vs APE 편향 비교 실험')
    parser.add_argument('--rotation-aug', action='store_true',
                        help='[Two-Stream] 학습 시 입력 프레임 회전 augmentation. '
                             'Position prior overfit 방지 (90%% 동일회전 + 10%% 독립회전)')
    parser.add_argument('--drop-path-rate', type=float, default=0.0,
                        help='DropPath (stochastic depth) rate (default 0.0, 미래 확장 대비). '
                             '현재는 파라미터만 저장. ViT-Base 관례 0.1.')

    # v9: P decoder target 선택 — v4 base 구조 유지, decoder target만 교체
    parser.add_argument('--v9-p-target', type=str, default='future',
                        choices=['future', 'current', 'residual'],
                        help='[v9] P decoder target. '
                             'future=frame_{t+k} (v4 기본). '
                             'current=frame_t (standard MAE, semantic 강제). '
                             'residual=frame_{t+k}-frame_t (변화량, P encoder collapse 위험 관찰됨).')
    parser.add_argument('--v9-loss-weight-p', type=float, default=1.0,
                        help='[v9] P loss weight. residual은 magnitude 작아 5~10, current/future는 1.0 권장.')

    # v11: M encoder depth (P와 독립. P는 --depth, M은 motion sensor로 작게)
    parser.add_argument('--v11-m-depth', type=int, default=6,
                        help='[v11] M encoder depth (기본 6, sensor 역할이라 작게). '
                             'P encoder depth은 --depth (기본 12) 사용.')

    # Multi-GPU
    parser.add_argument('--no-multi-gpu', action='store_true',
                        help='Disable multi-GPU training (use single GPU)')

    # Acceleration
    parser.add_argument('--compile', action='store_true',
                        help='torch.compile() 적용 (10-30%% 가속, 첫 실행 시 컴파일 오버헤드)')

    # AWS / Post-training
    parser.add_argument('--s3-bucket', type=str, default=None,
                        help='S3 bucket to sync checkpoints after training (e.g. bys724-research-2026)')
    parser.add_argument('--s3-prefix', type=str, default='checkpoints',
                        help='S3 prefix for checkpoint upload (default: checkpoints)')
    parser.add_argument('--shutdown', action='store_true',
                        help='Shutdown EC2 instance after training completes')

    args = parser.parse_args()

    # 분산 학습 환경 감지 (SLURM_PROCID 또는 RANK 존재 시)
    import os as _os
    distributed = ("SLURM_PROCID" in _os.environ) or ("RANK" in _os.environ and "WORLD_SIZE" in _os.environ)
    if distributed:
        rank = int(_os.environ.get("SLURM_PROCID", _os.environ.get("RANK", "0")))
        world_size = int(_os.environ.get("SLURM_NTASKS", _os.environ.get("WORLD_SIZE", "1")))
        is_master = (rank == 0)
    else:
        rank = 0
        world_size = 1
        is_master = True

    # Auto-detect latest checkpoint (resume)
    if args.resume is None:
        import glob
        checkpoint_dir = args.checkpoint_dir or f'/workspace/data/checkpoints/{args.model.replace("-", "_")}'
        candidates = glob.glob(f"{checkpoint_dir}/*/latest.pt")
        if candidates:
            args.resume = sorted(candidates)[-1]
            if is_master:
                print(f"[Resume] Auto-detected checkpoint: {args.resume}")

    # Print GPU info (rank 0만)
    import torch
    if is_master:
        print("\n" + "="*60)
        print("GPU Information")
        print("="*60)
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        if distributed:
            print(f"  Distributed: rank={rank}, world_size={world_size} (DDP)")
        else:
            print(f"  Multi-GPU: {'Disabled' if args.no_multi_gpu else f'Enabled ({num_gpus} GPUs DataParallel)'}")

    # Auto-set checkpoint directory based on model name
    if args.checkpoint_dir is None:
        model_name = args.model.replace('-', '_')
        args.checkpoint_dir = f'/workspace/data/checkpoints/{model_name}'
        if is_master:
            print(f"Auto checkpoint_dir: {args.checkpoint_dir}")

    # Auto-calculate save interval for ~12 checkpoints
    if args.save_interval is None:
        args.save_interval = max(1, args.epochs // 12)
        if is_master:
            print(f"Auto save_interval: {args.save_interval} (for ~12 checkpoints)")

    # Create model
    if is_master:
        print("\n" + "="*60)
        print(f"Creating model: {args.model}")
        print("="*60)

    if args.model == 'two-stream':
        model = TwoStreamModel(depth=args.depth, num_stages=args.num_stages,
                               mask_ratio=args.mask_ratio, mask_ratio_p=args.mask_ratio_p,
                               use_ape=args.use_ape,
                               rotation_aug=args.rotation_aug,
                               drop_path_rate=args.drop_path_rate,
                               p_target=args.v9_p_target,
                               loss_weight_p=args.v9_loss_weight_p)
    elif args.model == 'two-stream-v11':
        # v11: Motion-Guided Attention Routing + Dual-Target Reconstruction
        # - v9/v10 P target/weight args는 v11에선 미사용 (hardcoded: L_t + L_tk)
        # - mask_ratio (M stream), mask_ratio_p (P stream) 사용. 기본 0.3 / 0.75
        # - P encoder depth = --depth, M encoder depth = --v11-m-depth (비대칭)
        # - CLS exchange 없음 (stream-independent)
        v11_mask_m = args.mask_ratio if args.mask_ratio > 0 else 0.3
        v11_mask_p = args.mask_ratio_p if args.mask_ratio_p is not None else 0.75
        model = TwoStreamV11Model(
            p_depth=args.depth,
            m_depth=args.v11_m_depth,
            mask_ratio_m=v11_mask_m,
            mask_ratio_p=v11_mask_p,
            rotation_aug=args.rotation_aug,
        )
    elif args.model == 'videomae':
        # 2-frame 적응: 공식 0.75는 16-frame temporal redundancy 전제.
        # 2-frame에서는 masking 완화 필요.
        vm_mask = args.mask_ratio if args.mask_ratio > 0 else 0.5
        model = VideoMAEModel(mask_ratio=vm_mask)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if is_master:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # torch.compile (DDP wrap 전에 적용해야 효과적)
    if args.compile:
        model = torch.compile(model)
        if is_master:
            print(f"torch.compile() enabled — 첫 step 컴파일 후 10-30% 가속 예상")

    # Create training dataset
    if is_master:
        print("\n" + "="*60)
        print(f"Loading training dataset: {args.train_data}")
        print("="*60)

    splits = [s.strip() for s in args.egodex_splits.split(',')]
    split_datasets = []
    for split in splits:
        ds = EgoDexDataset(
            data_root=args.egodex_root,
            split=split,
            max_gap=args.max_gap,
            sample_decay=args.sample_decay,
            loss_decay=args.loss_decay,
            max_videos=args.max_videos,
            sample_dist=args.sample_dist,
            sample_center=args.sample_center,
        )
        split_datasets.append(ds)
    if len(split_datasets) == 1:
        train_dataset = split_datasets[0]
    else:
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(split_datasets)
        if is_master:
            print(f"  Combined {len(splits)} splits: {splits} → {len(train_dataset)} samples")

    # Create evaluation dataset (always use EgoDex test)
    if is_master:
        print("\n" + "="*60)
        print("Loading evaluation dataset: EgoDex test")
        print("="*60)

    eval_dataset = EgoDexDataset(
        data_root=args.egodex_root,
        split='test',
        max_gap=args.max_gap,
        sample_decay=args.sample_decay,
        loss_decay=args.loss_decay,
    )

    # Training configuration summary (rank 0)
    if is_master:
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        print(f"  Model: {args.model}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size (per GPU): {args.batch_size}")
        if distributed:
            print(f"  Global batch size: {args.batch_size * world_size}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Save interval: {args.save_interval} epochs (~{args.epochs // args.save_interval} checkpoints)")
        print(f"  Eval interval: {args.eval_interval} epochs")
        print(f"  Max gap: {args.max_gap}")
        print(f"  Sample decay: {args.sample_decay}")
        print(f"  Loss decay: {args.loss_decay}")
        if args.resume:
            print(f"  Resume from: {args.resume}")
        print()
        print("="*60)
        print("Starting training...")
        print("="*60)

    model, history = train(
        model=model,
        train_dataset=train_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device='cuda',
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
        eval_dataset=eval_dataset,
        eval_interval=args.eval_interval,
        resume_from=args.resume,
        multi_gpu=not args.no_multi_gpu,
        use_ssim=args.ssim,
    )

    if is_master:
        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        if history['eval_loss']:
            print(f"Final eval loss: {history['eval_loss'][-1]:.6f}")

    # S3 upload
    if args.s3_bucket and args.checkpoint_dir:
        import subprocess
        model_name = args.model.replace('-', '_')
        s3_dest = f"s3://{args.s3_bucket}/{args.s3_prefix}/{model_name}/"
        print(f"\nUploading checkpoints to {s3_dest} ...")
        result = subprocess.run(
            ["aws", "s3", "sync", args.checkpoint_dir, s3_dest],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("S3 upload complete.")
        else:
            print(f"S3 upload failed:\n{result.stderr}")

    # Auto shutdown
    if args.shutdown:
        import subprocess
        print("\nShutting down instance in 60 seconds (Ctrl+C to cancel)...")
        subprocess.run(["sudo", "shutdown", "-h", "+1"])


if __name__ == '__main__':
    main()
