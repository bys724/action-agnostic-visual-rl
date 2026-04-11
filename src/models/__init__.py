"""
Models for action-agnostic visual representation learning.

- TwoStreamModel: M/P channel separation with CLS exchange (ours)
- VJEPAModel: Feature prediction + EMA teacher (direct competitor)
- VideoMAEModel: Masked autoencoding baseline (legacy)
"""

from .two_stream import TwoStreamModel, TwoStreamEncoder
from .v_jepa import VJEPAModel
from .videomae import VideoMAEModel, VideoMAEEncoderForVLA as VideoMAEEncoder
from .common import TwoStreamPreprocessing

__all__ = [
    "TwoStreamModel",
    "VJEPAModel",
    "VideoMAEModel",
    "TwoStreamEncoder",
    "VideoMAEEncoder",
    "TwoStreamPreprocessing",
]
