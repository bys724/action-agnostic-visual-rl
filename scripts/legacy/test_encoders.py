#!/usr/bin/env python
"""
Test all encoder types with action head.

Quick validation that the code runs correctly.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/workspace')

from src.models.openvla_encoder import (
    TwoStreamEncoderForOpenVLA,
    SingleStreamEncoderForOpenVLA,
    VideoMAEEncoderForOpenVLA,
)
from scripts.finetune_libero import EncoderWithActionHead

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_fake_obs(batch_size=4):
    """Simulate LIBERO observation: two consecutive frames."""
    img_prev = torch.rand(batch_size, 3, 224, 224)
    img_curr = torch.rand(batch_size, 3, 224, 224)
    pixel_values = torch.cat([img_prev, img_curr], dim=1)  # [B, 6, H, W]
    return pixel_values.to(device)


def test_encoder(name, encoder):
    """Test encoder forward pass and action prediction."""
    print("\n" + "="*60)
    print(f"Testing: {name}")
    print("="*60)

    # Create model
    model = EncoderWithActionHead(
        encoder=encoder,
        embed_dim=encoder.embed_dim,
        freeze_encoder=True
    ).to(device)

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # Forward pass
    model.eval()
    pixel_values = make_fake_obs(batch_size=4)

    with torch.no_grad():
        action = model(pixel_values)

    print(f"Input shape:  {pixel_values.shape}")
    print(f"Output shape: {action.shape}")
    print(f"Action range: [{action.min().item():.4f}, {action.max().item():.4f}]")

    # Verify output
    assert action.shape == (4, 7), f"Expected (4, 7), got {action.shape}"
    print("OK")

    return True


def main():
    print("="*60)
    print("ENCODER + ACTION HEAD TEST")
    print("="*60)

    # Test 1: Two-Stream (random init)
    encoder = TwoStreamEncoderForOpenVLA().to(device)
    test_encoder("Two-Stream (random)", encoder)

    # Test 2: Two-Stream (pre-trained)
    import os
    checkpoint = "/workspace/data/checkpoints/two_stream_test/20260209_065030/best_model.pt"
    if os.path.exists(checkpoint):
        encoder = TwoStreamEncoderForOpenVLA.from_checkpoint(checkpoint)
        test_encoder("Two-Stream (pre-trained)", encoder)
    else:
        print(f"\nSkipping pre-trained test: {checkpoint} not found")

    # Test 3: Single-Stream (random init)
    encoder = SingleStreamEncoderForOpenVLA().to(device)
    test_encoder("Single-Stream (random)", encoder)

    # Test 4: VideoMAE (random init)
    encoder = VideoMAEEncoderForOpenVLA().to(device)
    test_encoder("VideoMAE (random)", encoder)

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    main()
