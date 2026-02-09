#!/usr/bin/env python
"""
Training script for Video Prediction Models.

Supports three models:
    1. two-stream: Two-Stream Interleaved ViT (ours)
    2. single-stream: Single ViT encoder
    3. videomae: Masked autoencoder

Usage (in Docker container):
    # Two-Stream (default)
    python scripts/train_long.py --model two-stream --epochs 100

    # Single-Stream baseline
    python scripts/train_long.py --model single-stream --epochs 100

    # VideoMAE baseline
    python scripts/train_long.py --model videomae --epochs 100
"""

import argparse
import sys
sys.path.insert(0, '/workspace')

from src.models.two_stream import (
    TwoStreamVideoPredictor,
    BridgeDataset,
    EgoDexDataset,
    train,
)
from src.models.baselines import (
    VideoMAEPredictor,
    SingleStreamVideoPredictor,
)


def main():
    parser = argparse.ArgumentParser(description='Training for Video Prediction Models')

    # Model selection
    parser.add_argument('--model', type=str, default='two-stream',
                        choices=['two-stream', 'single-stream', 'videomae'],
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
    parser.add_argument('--max-gap', type=int, default=10,
                        help='Max frame gap (default: 10, for multi-scale temporal learning)')
    parser.add_argument('--sample-decay', type=float, default=0.3,
                        help='Sample probability decay (default: 0.3)')
    parser.add_argument('--loss-decay', type=float, default=0.7,
                        help='Loss weight decay (default: 0.7)')

    # Multi-GPU
    parser.add_argument('--no-multi-gpu', action='store_true',
                        help='Disable multi-GPU training (use single GPU)')

    args = parser.parse_args()

    # Print GPU info
    import torch
    print("\n" + "="*60)
    print("GPU Information")
    print("="*60)
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"  Multi-GPU: {'Disabled' if args.no_multi_gpu else f'Enabled ({num_gpus} GPUs)'}")

    # Auto-set checkpoint directory based on model name
    if args.checkpoint_dir is None:
        model_name = args.model.replace('-', '_')
        args.checkpoint_dir = f'/workspace/data/checkpoints/{model_name}'
        print(f"Auto checkpoint_dir: {args.checkpoint_dir}")

    # Auto-calculate save interval for ~12 checkpoints
    if args.save_interval is None:
        args.save_interval = max(1, args.epochs // 12)
        print(f"Auto save_interval: {args.save_interval} (for ~12 checkpoints)")

    # Create model
    print("\n" + "="*60)
    print(f"Creating model: {args.model}")
    print("="*60)

    if args.model == 'two-stream':
        model = TwoStreamVideoPredictor()
    elif args.model == 'single-stream':
        model = SingleStreamVideoPredictor()
    elif args.model == 'videomae':
        model = VideoMAEPredictor()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create training dataset
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
        train_dataset = EgoDexDataset(
            data_root=args.egodex_root,
            split='train',  # Use train split for training
            max_gap=args.max_gap,
            sample_decay=args.sample_decay,
            loss_decay=args.loss_decay,
            max_videos=args.max_videos,
        )

    # Create evaluation dataset (always use EgoDex test)
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

    # Training configuration summary
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Save interval: {args.save_interval} epochs (~{args.epochs // args.save_interval} checkpoints)")
    print(f"  Eval interval: {args.eval_interval} epochs")
    print(f"  Max gap: {args.max_gap}")
    print(f"  Sample decay: {args.sample_decay}")
    print(f"  Loss decay: {args.loss_decay}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print()

    # Start training
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
    )

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    if history['eval_loss']:
        print(f"Final eval loss: {history['eval_loss'][-1]:.6f}")


if __name__ == '__main__':
    main()
