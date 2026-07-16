"""Encoder adapters for the LIBERO BC-Transformer policy.

Each adapter maps an observation sequence (B, T, 3, H, W) to a token sequence
(B, T, embed_dim), absorbing the per-encoder native input format.

Usage:
    adapter = build_adapter(encoder_type, checkpoint_path, **kwargs)
    tokens = adapter(obs_seq)              # training (full sequence)
    adapter.reset()                        # start a rollout episode
    tokens = adapter(obs_seq[:, t:t+1])    # rollout step (T=1)
"""

from .base import EncoderAdapter, build_adapter

__all__ = ["EncoderAdapter", "build_adapter"]
