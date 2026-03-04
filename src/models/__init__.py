"""
Models for action-agnostic visual representation learning.

This module provides three models for future frame prediction:
- TwoStreamModel: M/P channel separation with CLS exchange (ours)
- SingleStreamModel: Single ViT baseline
- VideoMAEModel: Masked autoencoding baseline

Each model provides:
- Training interface: model.forward() and model.compute_loss()
- Encoder for VLA integration: <Model>Encoder class
"""

# Core models for training
from .two_stream import TwoStreamModel, TwoStreamEncoder
from .single_stream import SingleStreamModel, SingleStreamEncoder
from .videomae import VideoMAEModel, VideoMAEEncoder

# Common preprocessing
from .common import TwoStreamPreprocessing

__all__ = [
    # Training models
    "TwoStreamModel",
    "SingleStreamModel",
    "VideoMAEModel",
    # VLA encoders
    "TwoStreamEncoder",
    "SingleStreamEncoder",
    "VideoMAEEncoder",
    # Common components
    "TwoStreamPreprocessing",
]
