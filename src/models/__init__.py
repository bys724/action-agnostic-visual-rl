"""
모델 모듈

Core:
- two_stream_vit.py: Two-Stream Interleaved ViT (메인 인코더)
- two_stream_preprocessing.py: M/P 채널 전처리
- action_decoder.py: Action Decoder

Optional:
- behavior_encoder.py: Task-conditioned Behavior Encoder (향후 실험용)
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

# Optional - 향후 실험용
from .behavior_encoder import BehaviorEncoder, BehaviorEncoderConfig, TextEncoder

__all__ = [
    # Core
    "TwoStreamInterleavedViT",
    "TwoStreamViTConfig",
    "ChannelViT",
    "TwoStreamPreprocessing",
    "ActionDecoder",
    "ActionDecoderConfig",
    "FullPipeline",
    # Optional
    "BehaviorEncoder",
    "BehaviorEncoderConfig",
    "TextEncoder",
]
