"""Encoder adapters for LIBERO BC-Transformer policy.

각 어댑터는 obs sequence (B, T, 3, H, W) → token sequence (B, T, embed_dim)
변환. 인코더별 native input format 차이 흡수.

사용 패턴:
    adapter = build_adapter(encoder_type, checkpoint_path, **kwargs)
    tokens = adapter(obs_seq)          # 학습 (full sequence)
    adapter.reset()                    # rollout 에피소드 시작
    tokens = adapter(obs_seq[:, t:t+1]) # rollout step (T=1 단일 스텝)
"""

from .base import EncoderAdapter, build_adapter

__all__ = ["EncoderAdapter", "build_adapter"]
