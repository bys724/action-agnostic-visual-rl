"""
모델 모듈

Core:
- two_stream_vit.py: Two-Stream Interleaved ViT (메인 인코더)
- two_stream_preprocessing.py: M/P 채널 전처리
- action_decoder.py: Action Decoder

Legacy:
- visual_encoder.py: 시각 인코더 (DINOv2, CLIP 기반)
- behavior_encoder.py: Task-conditioned Behavior Encoder
- custom_vla.py: 커스텀 VLA 모델
"""

# Core - Two-Stream Interleaved ViT
from .two_stream_vit import (
    TwoStreamInterleavedViT,
    TwoStreamViTConfig,
    ChannelViT,
)
from .two_stream_preprocessing import (
    TwoStreamPreprocessing,
    TwoStreamViTAdapter,
    TwoStreamProcessor,
)
from .action_decoder import (
    ActionDecoder,
    ActionDecoderConfig,
    FullPipeline,
)

# Legacy
from .visual_encoder import VisualEncoder, MomentumEncoder
from .behavior_encoder import BehaviorEncoder, BehaviorEncoderConfig, TextEncoder
from .custom_vla import CustomVLA

__all__ = [
    # Core
    "TwoStreamInterleavedViT",
    "TwoStreamViTConfig",
    "ChannelViT",
    "TwoStreamPreprocessing",
    "ActionDecoder",
    "ActionDecoderConfig",
    "FullPipeline",
    # Legacy
    "VisualEncoder",
    "BehaviorEncoder",
    "TextEncoder",
    "CustomVLA",
]
