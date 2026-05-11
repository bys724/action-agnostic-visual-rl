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

from src.models import (
    TwoStreamModel,
    TwoStreamV11Model,
    TwoStreamV12Model,
    TwoStreamV13Model,
    TwoStreamV14Model,
    TwoStreamV15Model,
    VideoMAEModel,
)
from src.datasets import EgoDexDataset
from src.training.pretrain import train


def main():
    parser = argparse.ArgumentParser(description='Training for Video Prediction Models')

    # Model selection
    parser.add_argument('--model', type=str, default='two-stream',
                        choices=['two-stream', 'two-stream-v11', 'two-stream-v12', 'two-stream-v13', 'two-stream-v14', 'two-stream-v15', 'videomae'],
                        help='Model type (default: two-stream). '
                             'two-stream-v11 = motion-guided routing + dual-target. '
                             'two-stream-v12 = v11 + CLS semantic residual + EMA teacher. '
                             'two-stream-v13 = dual-frame recon + motion-routed latent + DINO global CLS. '
                             'two-stream-v14 = stream-wise paradigm specialization (P=MAE+V-JEPA, M=DINO). '
                             'two-stream-v15 = v15 final: predictor-only V-JEPA + V-JEPA-M (Option B) + L_compose + 3-frame triple training.')

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

    # v11 ablation: motion-routing 메커니즘 선택
    parser.add_argument('--v11-routing-mode', type=str, default='v_from_p',
                        choices=['v_from_p', 'v_from_m'],
                        help='[v11] Phase 2 routing 방식. '
                             'v_from_p (기본, paper novelty): Q,K←M, V←P. '
                             'v_from_m (ablation): 표준 cross-attn (Q←P, K,V←M).')

    # v12: Semantic Residual + EMA Teacher (post-CoRL follow-up)
    parser.add_argument('--v12-residual-weight', type=float, default=0.05,
                        help='[v12] λ_residual (default 0.05, v8 1차 0.2 scale 실패 교훈). '
                             'ep4 sanity 후 조정.')
    parser.add_argument('--v12-vicreg-var-weight', type=float, default=1.0,
                        help='[v12] α (variance hinge weight, V-JEPA 1 standard).')
    parser.add_argument('--v12-vicreg-cov-weight', type=float, default=1.0,
                        help='[v12] β (off-diagonal covariance weight).')
    parser.add_argument('--v12-ema-momentum-init', type=float, default=0.996,
                        help='[v12] EMA momentum start (V-JEPA 2 schedule).')
    parser.add_argument('--v12-ema-momentum-final', type=float, default=0.9999,
                        help='[v12] EMA momentum final (linear warmup).')
    parser.add_argument('--v12-predictor-heads', type=int, default=12,
                        help='[v12] Cross-attention predictor heads.')

    # v13: Dual-Frame Reconstruction + Motion-Routed Latent + Full DINO Global CLS
    parser.add_argument('--v13-patch-pred-weight', type=float, default=1.5,
                        help='[v13] λ_patch — per-patch SmoothL1 weight (V-JEPA-style).')
    parser.add_argument('--v13-cls-pred-weight', type=float, default=0.3,
                        help='[v13] λ_cls — DINO distillation weight. P encoder direct distill 후 '
                             '신호 강화 가능 (이전 motion-routed CLS 0.01 → DINOv2 표준 0.3).')
    parser.add_argument('--v13-dino-center-momentum', type=float, default=0.9,
                        help='[v13] EMA momentum for DINO center buffer (DINOv2 default 0.9).')
    parser.add_argument('--v13-mask-ratio-p-dino', type=float, default=0.4,
                        help='[v13] DINO path mask ratio (recon mask 0.75와 분리, DINOv2 30~50% 표준).')
    parser.add_argument('--v13-num-prototypes', type=int, default=1024,
                        help='[v13] DINO prototype count K (cluster diversity). '
                             'EgoDex 다양성 수준에 K=1024 적정 (DINO ImageNet 표준 65536 대비 작게).')
    parser.add_argument('--v13-dino-teacher-temp', type=float, default=0.04,
                        help='[v13] Teacher temperature τ_t (low → sharp distribution).')
    parser.add_argument('--v13-dino-student-temp', type=float, default=0.1,
                        help='[v13] Student temperature τ_s (higher than τ_t).')
    parser.add_argument('--v13-ema-momentum-init', type=float, default=0.996,
                        help='[v13] EMA momentum start for teacher.')
    parser.add_argument('--v13-ema-momentum-final', type=float, default=0.9999,
                        help='[v13] EMA momentum final (linear warmup).')

    # v14: Stream-wise Paradigm Specialization (P=MAE+V-JEPA, M=DINO)
    parser.add_argument('--v14-lambda-pred', type=float, default=1.0,
                        help='[v14] λ_pred — V-JEPA loss weight (P stream). '
                             'Warmup이 활성화되면 이 값이 schedule target.')
    parser.add_argument('--v14-lambda-pred-warmup-start', type=float, default=None,
                        help='[v14] λ_pred 시작 값. None이면 warmup 비활성 (정적 λ_pred). '
                             '권장: 0.01 — V-JEPA target이 미숙한 학습 초기에는 미세하게만 반영.')
    parser.add_argument('--v14-lambda-pred-warmup-epochs', type=int, default=10,
                        help='[v14] λ_pred linear warmup 길이 (epoch). 이 값 도달 시 target. '
                             '50ep 학습 기준 default 10 (V-JEPA 2/DINOv2 표준 첫 1/5 구간).')
    parser.add_argument('--v14-lambda-dino', type=float, default=1.0,
                        help='[v14] λ_dino — DINO loss weight (M stream).')
    parser.add_argument('--v14-dino-n-crop', type=int, default=1,
                        help='[v14] DINO student multi-crop count. '
                             'sanity=1 (baseline), 본 학습=2 (Option B). '
                             '추가 crop은 raw pair에서 GPU random crop으로 생성, M_encoder만 추가 forward.')
    parser.add_argument('--v14-num-prototypes', type=int, default=1024,
                        help='[v14] DINO prototype K (default 1024, 데이터셋 보수적).')
    parser.add_argument('--v14-dino-teacher-temp', type=float, default=0.04,
                        help='[v14] Teacher temperature τ_T.')
    parser.add_argument('--v14-dino-student-temp', type=float, default=0.1,
                        help='[v14] Student temperature τ_S.')
    parser.add_argument('--v14-dino-center-momentum', type=float, default=0.9,
                        help='[v14] EMA momentum for DINO center (DINOv2 default 0.9).')
    parser.add_argument('--v14-ema-momentum-init', type=float, default=0.996,
                        help='[v14] EMA momentum start for teachers (P + M + DINOHead).')
    parser.add_argument('--v14-ema-momentum-final', type=float, default=0.9999,
                        help='[v14] EMA momentum final (linear warmup).')

    # v15: Layered specialization with predictor-only V-JEPA + V-JEPA-M
    parser.add_argument('--v15-lambda-pred', type=float, default=1.0,
                        help='[v15] λ_pred — V-JEPA P loss weight. Warmup target.')
    parser.add_argument('--v15-lambda-pred-warmup-start', type=float, default=None,
                        help='[v15] λ_pred warmup 시작 값. None이면 정적 λ_pred. 권장 0.01.')
    parser.add_argument('--v15-lambda-pred-warmup-epochs', type=int, default=10,
                        help='[v15] λ_pred linear warmup 길이 (epoch). 기본 10.')
    parser.add_argument('--v15-lambda-m-jepa', type=float, default=1.0,
                        help='[v15] λ_m_jepa — V-JEPA M loss weight (masked patches). Warmup target.')
    parser.add_argument('--v15-lambda-m-jepa-warmup-start', type=float, default=None,
                        help='[v15] λ_m_jepa warmup 시작 값. None이면 정적. 권장 0.01.')
    parser.add_argument('--v15-lambda-m-jepa-warmup-epochs', type=int, default=10,
                        help='[v15] λ_m_jepa linear warmup 길이 (epoch). 기본 10.')
    parser.add_argument('--v15-lambda-compose', type=float, default=1.0,
                        help='[v15 final] λ_compose — Compositional auxiliary loss weight (replaces DINO).')
    parser.add_argument('--v15-lambda-compose-warmup-start', type=float, default=None,
                        help='[v15 final] λ_compose warmup 시작 값. None이면 정적. 권장 0.01.')
    parser.add_argument('--v15-lambda-compose-warmup-epochs', type=int, default=10,
                        help='[v15 final] λ_compose linear warmup 길이 (epoch). 기본 10.')
    parser.add_argument('--v15-composition-mode', type=str, default='linear_residual',
                        choices=['linear_residual', 'linear', 'mlp'],
                        help='[v15 final] composition_head 모드. sanity=linear_residual (param 0), 본=mlp.')
    parser.add_argument('--v15-composition-hidden-dim', type=int, default=None,
                        help='[v15 final] composition_head mlp mode hidden dim (None=embed_dim).')
    parser.add_argument('--v15-mask-ratio-m-jepa', type=float, default=0.5,
                        help='[v15 final] V-JEPA-M에서 M stream mask ratio (default 0.5).')
    parser.add_argument('--v15-ema-momentum-init', type=float, default=0.999,
                        help='[v15 final] EMA momentum start (TeacherP + TeacherM_encoder).')
    parser.add_argument('--v15-ema-momentum-final', type=float, default=0.9999,
                        help='[v15 final] EMA momentum final (linear warmup).')

    # Multi-GPU
    parser.add_argument('--no-multi-gpu', action='store_true',
                        help='Disable multi-GPU training (use single GPU)')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='DataLoader num_workers (default 16). 디버깅 시 0으로.')

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
            routing_mode=args.v11_routing_mode,
        )
    elif args.model == 'two-stream-v13':
        # v13: dual-frame reconstruction + motion-routed latent + DINO global CLS
        # - frame_t / frame_{t+k} 모두 student P encoder 통과 (각자 mask 독립)
        # - motion-routing은 frame_t의 p_state에서 시작 → predicted_p_tk
        # - teacher (EMA) 두 input: cropped frame_{t+k} (per-patch target) +
        #   raw 256x256 frame_{t+k} (DINO-style global CLS target)
        # - L_total = L_t + L_tk + λ_patch · L_pred_patch + λ_cls · L_pred_cls
        v13_mask_m = args.mask_ratio if args.mask_ratio > 0 else 0.3
        v13_mask_p = args.mask_ratio_p if args.mask_ratio_p is not None else 0.75
        model = TwoStreamV13Model(
            p_depth=args.depth,
            m_depth=args.v11_m_depth,
            mask_ratio_m=v13_mask_m,
            mask_ratio_p=v13_mask_p,
            rotation_aug=args.rotation_aug,
            routing_mode=args.v11_routing_mode,
            patch_pred_weight=args.v13_patch_pred_weight,
            cls_pred_weight=args.v13_cls_pred_weight,
            dino_center_momentum=args.v13_dino_center_momentum,
            num_prototypes=args.v13_num_prototypes,
            dino_teacher_temp=args.v13_dino_teacher_temp,
            dino_student_temp=args.v13_dino_student_temp,
            mask_ratio_p_dino=args.v13_mask_ratio_p_dino,
        )
    elif args.model == 'two-stream-v14':
        # v14: Stream-wise paradigm specialization
        # - P stream: MAE (L_t + L_tk_recon) + V-JEPA (L_pred), reconstruction-anchored
        # - M stream: DINO (L_dino) only, distillation-only
        # - 3 EMA teachers: TeacherP (V-JEPA target) + TeacherM + TeacherDINOHead
        # - Multi-crop N: sanity=1, 본=2 (DINO student에만, M_encoder 추가 forward)
        # - mask_ratio_m은 v14에서 무시 (M unmasked hardcoded)
        v14_mask_p = args.mask_ratio_p if args.mask_ratio_p is not None else 0.75
        model = TwoStreamV14Model(
            p_depth=args.depth,
            m_depth=args.v11_m_depth,
            mask_ratio_p=v14_mask_p,
            rotation_aug=args.rotation_aug,
            routing_mode=args.v11_routing_mode,
            lambda_pred=args.v14_lambda_pred,
            lambda_dino=args.v14_lambda_dino,
            dino_n_crop=args.v14_dino_n_crop,
            num_prototypes=args.v14_num_prototypes,
            dino_teacher_temp=args.v14_dino_teacher_temp,
            dino_student_temp=args.v14_dino_student_temp,
            dino_center_momentum=args.v14_dino_center_momentum,
        )
    elif args.model == 'two-stream-v15':
        # v15 final: docs/v15_compositional_aux_design.md 기준
        # - DINO 제거 → L_compose 추가 (compositional structure on M_encoder)
        # - V-JEPA P: V source + target 모두 TeacherP (predictor only)
        # - V-JEPA M (Option B): M_encoder masked + M_decoder vs TeacherM_encoder unmasked
        # - 3-frame triple (옵션 B): P MAE × 3 frame + V-JEPA P × 3 segment + V-JEPA M × 1 (long) + L_compose
        # - p_motion_decoder = (routing + interpreter) × N (interleaved)
        # - mask_token_m 활성화 (V-JEPA-M)
        v15_mask_p = args.mask_ratio_p if args.mask_ratio_p is not None else 0.75
        model = TwoStreamV15Model(
            p_depth=args.depth,
            m_depth=args.v11_m_depth,
            mask_ratio_p=v15_mask_p,
            rotation_aug=args.rotation_aug,
            routing_mode=args.v11_routing_mode,
            lambda_pred=args.v15_lambda_pred,
            lambda_m_jepa=args.v15_lambda_m_jepa,
            lambda_compose=args.v15_lambda_compose,
            mask_ratio_m_jepa=args.v15_mask_ratio_m_jepa,
            composition_mode=args.v15_composition_mode,
            composition_hidden_dim=args.v15_composition_hidden_dim,
        )
    elif args.model == 'two-stream-v12':
        # v12: v11 + CLS-level semantic residual + EMA teacher (post-CoRL follow-up)
        # - v11 모든 reconstruction path 유지 (L_t + L_tk)
        # - 추가: M_head, P_head, CrossAttnPredictor (Q←P, K/V←M), TeacherP (EMA)
        # - Loss: L_total = L_recon + λ·L_residual + α·L_var + β·L_cov
        # - EMA momentum은 training loop에서 스케줄링됨 (--v12-ema-momentum-init/-final)
        v12_mask_m = args.mask_ratio if args.mask_ratio > 0 else 0.3
        v12_mask_p = args.mask_ratio_p if args.mask_ratio_p is not None else 0.75
        model = TwoStreamV12Model(
            p_depth=args.depth,
            m_depth=args.v11_m_depth,
            mask_ratio_m=v12_mask_m,
            mask_ratio_p=v12_mask_p,
            rotation_aug=args.rotation_aug,
            routing_mode=args.v11_routing_mode,
            residual_weight=args.v12_residual_weight,
            vicreg_var_weight=args.v12_vicreg_var_weight,
            vicreg_cov_weight=args.v12_vicreg_cov_weight,
            predictor_num_heads=args.v12_predictor_heads,
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

    # v13/v14: train dataset이 raw 256 view도 함께 반환해야 함 (DINO teacher용)
    needs_global = args.model in ('two-stream-v13', 'two-stream-v14')
    needs_triple = args.model == 'two-stream-v15'

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
            return_global=needs_global,
            return_triple=needs_triple,
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
        return_triple=needs_triple,
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

    # v12/v13 EMA momentum schedule (linear warmup over training)
    v12_kwargs = {}
    if args.model == 'two-stream-v12':
        v12_kwargs = {
            'v12_ema_momentum_init': args.v12_ema_momentum_init,
            'v12_ema_momentum_final': args.v12_ema_momentum_final,
        }
    elif args.model == 'two-stream-v13':
        v12_kwargs = {
            'v12_ema_momentum_init': args.v13_ema_momentum_init,
            'v12_ema_momentum_final': args.v13_ema_momentum_final,
        }
    elif args.model == 'two-stream-v14':
        v12_kwargs = {
            'v12_ema_momentum_init': args.v14_ema_momentum_init,
            'v12_ema_momentum_final': args.v14_ema_momentum_final,
            'v14_lambda_pred_warmup_start': args.v14_lambda_pred_warmup_start,
            'v14_lambda_pred_warmup_epochs': args.v14_lambda_pred_warmup_epochs,
            'v14_lambda_pred_target': args.v14_lambda_pred,
        }
    elif args.model == 'two-stream-v15':
        v12_kwargs = {
            'v12_ema_momentum_init': args.v15_ema_momentum_init,
            'v12_ema_momentum_final': args.v15_ema_momentum_final,
            # v15 final: λ_pred + λ_m_jepa + λ_compose warmup (DINO 제거)
            'v14_lambda_pred_warmup_start': args.v15_lambda_pred_warmup_start,
            'v14_lambda_pred_warmup_epochs': args.v15_lambda_pred_warmup_epochs,
            'v14_lambda_pred_target': args.v15_lambda_pred,
            'v15_lambda_m_jepa_warmup_start': args.v15_lambda_m_jepa_warmup_start,
            'v15_lambda_m_jepa_warmup_epochs': args.v15_lambda_m_jepa_warmup_epochs,
            'v15_lambda_m_jepa_target': args.v15_lambda_m_jepa,
            # v15 final: λ_compose 신규 (replaces λ_dino)
            'v15_lambda_compose_warmup_start': args.v15_lambda_compose_warmup_start,
            'v15_lambda_compose_warmup_epochs': args.v15_lambda_compose_warmup_epochs,
            'v15_lambda_compose_target': args.v15_lambda_compose,
        }

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
        num_workers=args.num_workers,
        **v12_kwargs,
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
