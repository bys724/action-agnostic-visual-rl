"""
Training utilities for action-agnostic visual representation learning.

This module provides training functions for different learning objectives:
- Pre-training: Self-supervised learning on videos (no action labels)
- Fine-tuning: Supervised learning for action prediction (future work)
"""

from .pretrain import train, train_epoch, evaluate

__all__ = [
    "train",
    "train_epoch",
    "evaluate",
]
