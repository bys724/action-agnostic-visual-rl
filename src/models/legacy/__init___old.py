"""
모델 모듈

Core:
- two_stream.py: Two-Stream Video Predictor (통합 버전)
- baselines.py: Baseline models (Single-Stream)
- videomae_wrapper.py: VideoMAE wrapper
- openvla_encoder.py: OpenVLA 호환 encoder wrappers
- action_head.py: Action head for LIBERO fine-tuning/evaluation

Legacy (호환성):
- two_stream_vit.py: Two-Stream Interleaved ViT
- two_stream_preprocessing.py: M/P 채널 전처리
- action_decoder.py: Action Decoder
- behavior_encoder.py: Task-conditioned Behavior Encoder
"""

# Core - Video Prediction Models
from .two_stream import (
    TwoStreamVideoPredictor,
    TwoStreamPreprocessing as TwoStreamPreprocessingNew,
    InterleavedTwoStreamViT,
    PixelwiseFusion,
    VideoDecoder,
)
from .baselines import SingleStreamVideoPredictor
from .videomae_wrapper import VideoMAEForBridge

# OpenVLA Encoder Wrappers
from .openvla_encoder import (
    TwoStreamEncoderForOpenVLA,
    SingleStreamEncoderForOpenVLA,
    VideoMAEEncoderForOpenVLA,
)

# Action Head for LIBERO evaluation
from .action_head import ActionHead, EncoderWithActionHead

# Legacy - Two-Stream Interleaved ViT
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
    # Core - Video Prediction
    "TwoStreamVideoPredictor",
    "SingleStreamVideoPredictor",
    "VideoMAEForBridge",
    # OpenVLA Encoders
    "TwoStreamEncoderForOpenVLA",
    "SingleStreamEncoderForOpenVLA",
    "VideoMAEEncoderForOpenVLA",
    # Action Head (LIBERO evaluation)
    "ActionHead",
    "EncoderWithActionHead",
    # Legacy
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
