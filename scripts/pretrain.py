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

from src.models import TwoStreamModel, VJEPAModel, VideoMAEModel
from src.datasets import EgoDexDataset, BridgeDataset
from src.training.pretrain import train


def main():
    parser = argparse.ArgumentParser(description='Training for Video Prediction Models')

    # Model selection
    parser.add_argument('--model', type=str, default='two-stream',
                        choices=['two-stream', 'v-jepa', 'videomae'],
                        help='Model type (default: two-stream)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32, H100 can handle more)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')

    # Dataset parameters
    parser.add_argument('--train-data', type=str, default='bridge',
                        choices=['bridge', 'egodex'],
                        help='Training dataset (default: bridge)')
    parser.add_argument('--bridge-root', type=str,
                        default='/workspace/data/datasets/bridge_v2',
                        help='Bridge V2 data root')
    parser.add_argument('--egodex-root', type=str,
                        default='/workspace/data/egodex',
                        help='EgoDex data root')
    parser.add_argument('--max-trajectories', type=int, default=None,
                        help='Max trajectories for Bridge (None = all)')
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
    parser.add_argument('--v7-big', action='store_true',
                        help='[Two-Stream v7-big] P decoder 제거, M decoder 2개 (bg/motion), '
                             'CLS_P 2개로 분리. |ΔL| 기반 Gaussian weighted loss.')
    parser.add_argument('--sigma', type=float, default=0.03,
                        help='[v7-big] Gaussian weighting σ — w_bg=exp(-(|ΔL|/σ)²). '
                             'Default 0.03 (EgoDex p75 경계).')

    # v8: SimSiam-like temporal SSL (EMA on P-stream, M stream shared & detached from L_P)
    parser.add_argument('--v8-mode', action='store_true',
                        help='[Two-Stream v8] L_M (pixel) + λ·L_P (representation prediction). '
                             'Teacher는 student P의 EMA, input (t+k, t+k) → zero M. '
                             'Student CLS exchange 직전 cls_m.detach() → L_P는 M stream에 안 흐름. '
                             '자세한 설계: docs/architecture/v8_siamsimmae.md')
    parser.add_argument('--v8-lambda-max', type=float, default=0.5,
                        help='[v8] L_P weight max after warmup (default 0.5). '
                             '조정 여지: collapse 시 낮춤 (0.1~0.2), 정체 시 상향 (1.0)')
    parser.add_argument('--v8-lambda-warmup-epochs', type=int, default=5,
                        help='[v8] λ cosine warmup epochs (default 5). '
                             '조정 여지: L_M 안정화 느리면 연장 (10ep)')
    parser.add_argument('--v8-ema-tau-base', type=float, default=0.996,
                        help='[v8] EMA momentum τ base (0.996 BYOL/I-JEPA 표준). '
                             'Cosine schedule로 1.0까지 증가. 빠른 teacher 원하면 0.99')
    parser.add_argument('--v8-pred-head-ratio', type=float, default=2.0,
                        help='[v8] Prediction head hidden ratio (default 2.0 → hidden=2D)')
    parser.add_argument('--v8-alpha-var', type=float, default=0.0,
                        help='[v8] VICReg-lite variance regularization weight (default 0.0, 비활성). '
                             'Collapse 방어용: 0.1 권장. 강하면 0.2, 미미한 효과면 0.05.')
    parser.add_argument('--v8-var-target', type=float, default=1.0,
                        help='[v8] Variance target γ for VICReg-lite (default 1.0). '
                             'L_var = mean(relu(γ - std_per_dim)).')
    parser.add_argument('--v8-cls-m-grad-ratio', type=float, default=0.0,
                        help='[v8] CLS exchange 직전 cls_m의 L_P gradient 유입 비율 (default 0.0 = 완전 detach). '
                             '0.3 권장: M stream이 L_M에 주로 반응하되 L_P의 약한 pull 허용 (L_M trivial 문제 완화).')
    parser.add_argument('--drop-path-rate', type=float, default=0.0,
                        help='DropPath (stochastic depth) rate (default 0.0, 미래 확장 대비). '
                             '현재는 파라미터만 저장. ViT-Base 관례 0.1.')

    # v9: P decoder residual target (frame_{t+k} - frame_t) — v4 base 구조에서 loss만 수정
    parser.add_argument('--v9-residual-p', action='store_true',
                        help='[Two-Stream v9] P decoder target을 residual (frame_{t+k} - frame_t)로 변경. '
                             'P가 frame_t identity shortcut을 학습 못 하도록 강제 → motion cue 활용 의무화. '
                             'v4/v6 구조 그대로 + loss 수정만 (EMA teacher 없음).')
    parser.add_argument('--v9-loss-weight-p', type=float, default=1.0,
                        help='[v9] P loss weight (residual MSE magnitude 보정용). '
                             'Residual MSE는 full frame MSE 대비 작으므로 5~10 권장. '
                             'Default 1.0 = v4 동일 (residual-p 미사용 시).')

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
                               v7_big_mode=args.v7_big,
                               sigma=args.sigma,
                               v8_mode=args.v8_mode,
                               pred_head_ratio=args.v8_pred_head_ratio,
                               cls_m_grad_ratio=args.v8_cls_m_grad_ratio,
                               drop_path_rate=args.drop_path_rate,
                               residual_p_target=args.v9_residual_p,
                               loss_weight_p=args.v9_loss_weight_p)
    elif args.model == 'v-jepa':
        model = VJEPAModel(depth=args.depth)
    elif args.model == 'videomae':
        # 2-frame 적응: 공식 0.75는 16-frame temporal redundancy 전제.
        # V-JEPA 실패 사례와 동일한 이유로 2-frame에서는 masking 완화 필요.
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

    if args.train_data == 'bridge':
        train_dataset = BridgeDataset(
            data_root=args.bridge_root,
            max_gap=args.max_gap,
            sample_decay=args.sample_decay,
            loss_decay=args.loss_decay,
            max_trajectories=args.max_trajectories,
        )
    else:  # egodex
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
        # v8 하이퍼파라미터 (v8_mode=True일 때만 사용됨)
        v8_lambda_max=args.v8_lambda_max,
        v8_lambda_warmup_epochs=args.v8_lambda_warmup_epochs,
        v8_ema_tau_base=args.v8_ema_tau_base,
        v8_alpha_var=args.v8_alpha_var,
        v8_var_target=args.v8_var_target,
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
