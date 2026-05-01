"""
Models for action-agnostic visual representation learning.

- TwoStreamModel: M/P channel separation with CLS exchange (ours, v4~v10)
- TwoStreamV11Model: Motion-guided attention routing + dual-target (v11)
- TwoStreamV12Model: v11 + CLS-level semantic residual + EMA teacher (post-CoRL)
- VideoMAEModel: Masked autoencoding baseline
"""

from .two_stream import TwoStreamModel, TwoStreamEncoder
from .two_stream_v11 import TwoStreamV11Model, TwoStreamV11Encoder
from .two_stream_v12 import TwoStreamV12Model
from .videomae import VideoMAEModel, VideoMAEEncoderForVLA as VideoMAEEncoder
from .common import TwoStreamPreprocessing

__all__ = [
    "TwoStreamModel",
    "TwoStreamV11Model",
    "TwoStreamV12Model",
    "VideoMAEModel",
    "TwoStreamEncoder",
    "TwoStreamV11Encoder",
    "VideoMAEEncoder",
    "TwoStreamPreprocessing",
]
