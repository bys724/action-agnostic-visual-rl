#!/usr/bin/env python
"""
학습 스크립트

사용법:
    # 단일 GPU 학습
    python src/training/train.py --config configs/train_config.yaml

    # Multi-GPU 학습 (DDP)
    torchrun --nproc_per_node=2 src/training/train.py --config configs/train_config.yaml

    # Docker에서 실행
    docker exec dev-env python src/training/train.py --config configs/train_config.yaml
"""

import os
import argparse
import yaml
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 프로젝트 루트 추가
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataloader import create_dataloader, BridgeDataset, TrajectoryDataset
from training.trainer import Trainer
from models.custom_vla import CustomVLA


def setup_distributed():
    """DDP 설정"""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    return 0, 1, 0


def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Action-Agnostic Visual RL Training")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--mode", type=str, default="finetune", choices=["pretrain", "finetune"])

    # Override 옵션
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)

    args = parser.parse_args()

    # DDP 설정
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    if is_main:
        print("="*60)
        print("Action-Agnostic Visual RL Training")
        print("="*60)

    # 설정 로드
    config = load_config(args.config)

    # Override 적용
    if args.lr:
        config["training"]["lr"] = args.lr
    if args.epochs:
        config["training"]["max_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir

    if is_main:
        print(f"Config: {args.config}")
        print(f"Mode: {args.mode}")
        print(f"World size: {world_size}")

    # Device 설정
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # 모델 생성
    model = CustomVLA(
        visual_backbone=config["model"].get("backbone", "dinov2"),
        visual_model_size=config["model"].get("model_size", "base"),
        freeze_visual=config["model"].get("freeze_visual", False),
        partial_finetune_layers=config["model"].get("partial_finetune_layers"),
        use_momentum_encoder=(args.mode == "pretrain"),
    )
    model = model.to(device)

    # DDP 래핑
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")

    # 데이터 로더 생성
    data_config = config["data"]
    dataset_type = data_config.get("type", "bridge")

    if dataset_type == "bridge":
        train_dataset = BridgeDataset(
            data_dir=data_config["data_dir"],
            split="train",
            image_size=tuple(data_config.get("image_size", [224, 224]))
        )
    else:
        train_dataset = TrajectoryDataset(
            data_dir=data_config["data_dir"],
            success_only=data_config.get("success_only", True)
        )

    # DDP용 샘플러
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True
    )

    if is_main:
        print(f"Dataset: {len(train_dataset)} samples")
        print(f"Batch size: {config['training']['batch_size']}")

    # Trainer 생성
    trainer = Trainer(
        model=model.module if world_size > 1 else model,
        train_loader=train_loader,
        lr=config["training"]["lr"],
        max_epochs=config["training"]["max_epochs"],
        checkpoint_dir=config["training"].get("checkpoint_dir", "./data/checkpoints/custom"),
        log_dir=config["training"].get("log_dir", "./logs/training"),
        device=device,
        mixed_precision=config["training"].get("mixed_precision", True)
    )

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 학습 시작
    if is_main:
        print("\nStarting training...")

    trainer.train()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
