"""
학습 트레이너

Action-agnostic visual representation 학습을 위한 트레이너
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    모델 학습 트레이너

    Features:
    - 체크포인트 저장/로드
    - TensorBoard 로깅
    - Mixed precision 학습
    - Gradient accumulation
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        gradient_accumulation_steps: int = 1,
        checkpoint_dir: str = "./data/checkpoints/custom",
        log_dir: str = "./logs/training",
        device: str = "cuda",
        mixed_precision: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision

        # 디렉토리 생성
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer & Scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs * len(train_loader)
        )

        # Mixed precision
        self.scaler = torch.GradScaler('cuda') if mixed_precision else None

        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f"{log_dir}/{timestamp}")

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # 데이터 이동
            images = batch["images"].to(self.device)
            actions = batch["actions"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass
            with torch.autocast(device_type='cuda', enabled=self.mixed_precision):
                loss = self.model.compute_loss(images, actions, mask)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Logging
            if self.global_step % 100 == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)

        return {"loss": total_loss / num_batches}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """검증"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            actions = batch["actions"].to(self.device)
            mask = batch["mask"].to(self.device)

            loss = self.model.compute_loss(images, actions, mask)
            total_loss += loss.item()
            num_batches += 1

        val_loss = total_loss / num_batches
        self.writer.add_scalar("val/loss", val_loss, self.global_step)

        return {"loss": val_loss}

    def save_checkpoint(self, name: str = "latest"):
        """체크포인트 저장"""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Checkpoint loaded: {path} (epoch {self.current_epoch})")

    def train(self):
        """전체 학습 루프"""
        print(f"Starting training for {self.max_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            print(f"Epoch {epoch+1}/{self.max_epochs} - Train loss: {train_metrics['loss']:.4f}")

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                print(f"  Val loss: {val_metrics['loss']:.4f}")

                # Best model 저장
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint("best")

            # 주기적 체크포인트
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch+1}")

            self.save_checkpoint("latest")

        self.writer.close()
        print("Training complete!")
